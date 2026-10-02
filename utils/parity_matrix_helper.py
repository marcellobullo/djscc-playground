import numpy as np
import os
from pathlib import Path
import scipy.sparse

# Assume project root is parent of 'utils'
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PCM_DIR = ROOT_DIR / "utils" / "parity_matrices"

def get_parity_matrix_path(n, k, pcm_dir=DEFAULT_PCM_DIR):
    """Constructs the path for a given (n, k) parity matrix."""
    return pcm_dir / f"n{n}k{k}.npy"

def load_parity_matrix(n, k, pcm_dir=DEFAULT_PCM_DIR):
    """
    Loads a parity check matrix from a local .npy file.

    Args:
        n (int): Codeword length.
        k (int): Message length.
        pcm_dir (Path, optional): Directory containing the matrix files.
                                  Defaults to utils/parity_matrices.

    Returns:
        scipy.sparse.csr_matrix: The loaded parity check matrix.
    """
    filepath = get_parity_matrix_path(n, k, pcm_dir)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parity matrix file not found: {filepath}. "
                              f"Please run 'python -m utils.parity_matrix_helper' from the project root to download it.")
    H_dense = np.load(filepath)
    return scipy.sparse.csr_matrix(H_dense)

def get_generator_and_info_bits(n, k, pcm_dir=DEFAULT_PCM_DIR):
    """
    Constructs the Generator Matrix in Reduced Row Echelon Form (RREF)
    to guarantee a systematic encoding, and returns the pivot column indices
    which directly map the message bits to the codeword.
    """
    import ldpc.code_util
    H = load_parity_matrix(n, k, pcm_dir)
    G_raw = ldpc.code_util.construct_generator_matrix(H)
    if G_raw is None:
        raise RuntimeError(f"Could not construct generator matrix for LDPC({n},{k})")
    G = G_raw.toarray() if scipy.sparse.issparse(G_raw) else G_raw
    G = G.astype(np.uint8)
    
    # Compute RREF over GF(2)
    rows, cols = G.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        pivot_indices = np.where(G[r:rows, c] == 1)[0]
        if len(pivot_indices) == 0:
            continue
        pivot_row = r + pivot_indices[0]
        if pivot_row != r:
            G[[r, pivot_row]] = G[[pivot_row, r]]
        pivots.append(c)
        mask = G[:, c] == 1
        mask[r] = False
        G[mask] ^= G[r]
        r += 1
        
    return G, np.array(pivots)

if __name__ == '__main__':
    # This script downloads the necessary parity check matrices from kaira
    # and stores them locally to avoid runtime internet dependency.
    try:
        from kaira.models.fec.encoders import LDPCCodeEncoder
    except ImportError:
        print("Error: kaira library not found. Please install it to download matrices (`pip install kaira-fec`).")
        exit(1)

    # Pairs of (n, k) to download, based on project usage.
    ldpc_pairs = [
        (960, 640),
        (1920, 960),
        (2304, 1152),
        (2304, 1728),
        (2304, 1920),
    ]

    print(f"Downloading parity matrices to {DEFAULT_PCM_DIR}...")
    DEFAULT_PCM_DIR.mkdir(parents=True, exist_ok=True)

    for n, k in ldpc_pairs:
        filepath = get_parity_matrix_path(n, k, DEFAULT_PCM_DIR)
        if filepath.exists():
            print(f"Matrix for ({n}, {k}) already exists at {filepath}. Skipping.")
            continue

        print(f"Downloading matrix for ({n}, {k})...")
        try:
            factory = LDPCCodeEncoder(code_length=n, code_dimension=k, rptu_database=True)
            H_dense = factory.parity_check_matrix.cpu().numpy()
            np.save(filepath, H_dense)
            print(f"Saved to {filepath}")
        except Exception as e:
            print(f"Failed to download matrix for ({n}, {k}): {e}")

    print("\nDownload process finished.")
    print("You can now run the transmitter and receiver scripts without an internet connection for LDPC matrices.")
