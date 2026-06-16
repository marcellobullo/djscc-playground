"""Aligner modules + a factory that loads a trained aligner from disk.

The five aligner classes are ported (kept faithful) from
SPAICOM/DJSCC-Semantic-Equalization `alignment/alignment_model.py`. The public call
is `aligner(latent) -> latent` for the single-pass aligners; the zero-shot aligner
exposes `compression()` (transmitter side) and `decompression()` (receiver side)
because it changes the transmitted dimensionality.

This module is pure-torch and self-contained so both codecs and the training script
can import it. The `load_aligner` factory mirrors the reference repo's
`prepare_aligner`: the aligner *type* is inferred from the filename keyword
(`linear`/`neural`, `mlp`, `twoconv`, `conv`, `zeroshot`).

Latent layout in this demo: the channel tensor has shape ``[1, tcn, h, w]`` where
``h = H // 4`` and ``w = W // 4``. The flattened latent dimension is
``m = tcn * h * w``. Unlike the reference repo (which splits a complex latent into
``2 * c`` real channels), here ``tcn`` already counts real+imaginary planes, so the
convolutional aligners operate on ``tcn`` channels directly.
"""

import os
import re

import torch
import torch.nn as nn


def a_inv_times_b(a, b):
    """Compute A^{-1} B efficiently, with a 1x1 fallback (ported verbatim)."""
    try:
        c = torch.linalg.solve(a, b)
    except RuntimeError as e:
        if "The input tensor A must have at least 2 dimensions" in str(e):
            if len(a.shape) == 2 and a.shape[0] == 1 and a.shape[1] == 1:
                c = (1 / a) * b
            else:
                raise e
        else:
            raise e
    return c


class _LinearAlignment(nn.Module):
    """Aligner backed by a single (m x m) matrix. Used for both the closed-form
    least-squares solution and Adam-optimized fitting."""

    def __init__(self, size=None, align_matrix=None):
        super().__init__()
        if align_matrix is not None:
            self.align_matrix = nn.Parameter(align_matrix, requires_grad=False)
        else:
            self.align_matrix = nn.Parameter(torch.empty(size, size))
            nn.init.xavier_uniform_(self.align_matrix)

    def forward(self, x):
        shape = x.shape
        x = x.flatten(start_dim=1)
        x = x @ self.align_matrix
        return x.reshape(shape)


class _MLPAlignment(nn.Module):
    """Aligner backed by a multi-layer perceptron over the flattened latent."""

    def __init__(self, input_dim, hidden_dims, output_dim=None, nonlinearity=nn.PReLU):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(nonlinearity())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x.reshape(shape)


class _ConvolutionalAlignment(nn.Module):
    """Aligner backed by a single convolutional layer (shape-preserving)."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out",
                                nonlinearity="leaky_relu")

    def forward(self, x):
        return self.conv(x)


class _TwoConvAlignment(nn.Module):
    """Aligner backed by two convolutional layers with a PReLU in between."""

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size,
                               padding=kernel_size // 2)
        self.non_linearity = nn.PReLU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size,
                               padding=kernel_size // 2)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out",
                                nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out",
                                nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.conv1(x)
        x = self.non_linearity(x)
        x = self.conv2(x)
        return x


class _ZeroShotAlignment(nn.Module):
    """Zero-shot (training-free) aligner using SVD/Parseval bases + whitening.

    `compression` (transmitter) maps the m-dim latent down to `channel_usage` dims;
    `decompression` (receiver) maps it back to m dims. The transmitted dimensionality
    therefore equals ``F_tilde.shape[0]`` (exposed as `transmitted_dim`).
    """

    def __init__(self, F_tilde, G_tilde, G, L, mean):
        super().__init__()
        self.F_tilde = nn.Parameter(F_tilde, requires_grad=False)
        self.G_tilde = nn.Parameter(G_tilde, requires_grad=False)
        self.G = nn.Parameter(G, requires_grad=False)
        self.L = nn.Parameter(L, requires_grad=False)
        self.mean = nn.Parameter(mean, requires_grad=False)

    def compression(self, input):
        x_hat = input.T
        # go to similarity scores
        x_hat = self.F_tilde @ x_hat
        # prewhitening
        x_hat = a_inv_times_b(self.L, x_hat - self.mean)
        # multiply by F is ignored because it is always 1
        return x_hat

    def decompression(self, input):
        y_hat = input
        # dewhitening
        y_hat = self.L @ y_hat + self.mean
        # go back to image
        y_hat = self.G_tilde @ y_hat
        return y_hat.T


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def _infer_kind(name: str) -> str:
    name = name.lower()
    if "twoconv" in name:
        return "twoconv"
    if "conv" in name:
        return "conv"
    if "mlp" in name:
        return "mlp"
    if "linear" in name or "neural" in name:
        return "linear"
    if "zeroshot" in name:
        return "zeroshot"
    raise ValueError(
        f"Cannot infer aligner type from filename '{name}'. The filename must contain "
        "one of: linear/neural, mlp, conv, twoconv, zeroshot.")


def load_aligner(path: str, tcn: int, h: int, w: int, device="cpu",
                 n_samples: int = None) -> nn.Module:
    """Instantiate the right aligner class for `path` and load its state dict.

    Args:
        path: path to a ``.pth`` state-dict whose filename encodes the type, e.g.
            ``aligner_conv.pth`` or ``aligner_zeroshot_512.pth``.
        tcn, h, w: latent geometry. The flattened dim is ``m = tcn * h * w`` and the
            convolutional channel count is ``tcn``.
        device: torch device for the loaded module.
        n_samples: zero-shot channel usage; if None it is parsed from the filename.

    Returns:
        An ``nn.Module`` with extra attributes ``.kind`` (str) and, for zero-shot,
        ``.transmitted_dim`` (int).
    """
    kind = _infer_kind(os.path.basename(path))
    m = tcn * h * w

    if kind == "linear":
        aligner = _LinearAlignment(size=m)
    elif kind == "mlp":
        aligner = _MLPAlignment(input_dim=m, hidden_dims=[m])
    elif kind == "twoconv":
        aligner = _TwoConvAlignment(in_channels=tcn, hidden_channels=tcn,
                                    out_channels=tcn, kernel_size=5)
    elif kind == "conv":
        aligner = _ConvolutionalAlignment(in_channels=tcn, out_channels=tcn,
                                          kernel_size=5)
    elif kind == "zeroshot":
        if n_samples is None:
            match = re.search(r"_(\d+)\.pth$", os.path.basename(path))
            if match is None:
                raise ValueError(
                    "Zero-shot aligner filename must end with '_<channel_usage>.pth' "
                    f"(e.g. aligner_zeroshot_512.pth); got '{os.path.basename(path)}'.")
            n_samples = int(match.group(1))
        aligner = _ZeroShotAlignment(
            F_tilde=torch.zeros(n_samples, m),
            G_tilde=torch.zeros(m, n_samples),
            G=torch.zeros(1, 1),
            L=torch.zeros(n_samples, n_samples),
            mean=torch.zeros(n_samples, 1),
        )

    state = torch.load(path, map_location=device)
    aligner.load_state_dict(state)
    aligner = aligner.to(device).eval()

    aligner.kind = kind
    if kind == "zeroshot":
        aligner.transmitted_dim = int(aligner.F_tilde.shape[0])
    return aligner
