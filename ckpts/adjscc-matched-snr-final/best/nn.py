"""Self-contained nn modules for AD-JSCC (Attention-based Deep JSCC).

Ported from the gr-deepjscc ``nn_model.py``. Unlike the custom ConvNeXt model,
AD-JSCC is plain-torch + compressai (GDN) and is **SNR-adaptive at both ends**:
attention (AF) modules take the SNR as a side input, so the encoder is NOT
channel-blind. 4x down/up-sampling; latent is [B, tcn, H/4, W/4].

Kept self-contained (no cross-package imports) so HuggingFace push_to_hub copies
a standalone repo. Loading a pushed repo requires ``compressai`` installed.
"""

import os

import torch
import torch.nn as nn
from compressai.layers import GDN


# ── building blocks ──────────────────────────────────────────────────────────

class FL_En_Module(nn.Module):
    """Feature-Learning encoder block: Conv2d + GDN + optional activation."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.GDN = GDN(out_channels)
        if activation == 'sigmoid':
            self.activate_func = nn.Sigmoid()
        elif activation == 'prelu':
            self.activate_func = nn.PReLU()
        else:
            self.activate_func = None

    def forward(self, inputs):
        out = self.GDN(self.conv1(inputs))
        if self.activate_func is not None:
            out = self.activate_func(out)
        return out


class FL_De_Module(nn.Module):
    """Feature-Learning decoder block: ConvTranspose2d + GDN + optional activation."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, out_padding, activation=None):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, output_padding=out_padding)
        self.GDN = GDN(out_channels)
        if activation == 'sigmoid':
            self.activate_func = nn.Sigmoid()
        elif activation == 'prelu':
            self.activate_func = nn.PReLU()
        else:
            self.activate_func = None

    def forward(self, inputs):
        out = self.GDN(self.deconv1(inputs))
        if self.activate_func is not None:
            out = self.activate_func(out)
        return out


class AL_CH_Module(nn.Module):
    """Channel-attention module conditioned on SNR."""

    def __init__(self, channel_size):
        super().__init__()
        self.Ave_Pooling = nn.AdaptiveAvgPool2d(1)
        self.FC_1 = nn.Linear(channel_size + 1, channel_size // 16)
        self.FC_2 = nn.Linear(channel_size // 16, channel_size)

    def forward(self, inputs, attention):
        b = inputs.shape[0]
        out_pooling = self.Ave_Pooling(inputs).view(b, -1)
        c = inputs.shape[1]
        in_fc = torch.cat((attention, out_pooling), dim=1).float()
        out_fc_1 = torch.nn.functional.relu(self.FC_1(in_fc))
        out_fc_2 = torch.sigmoid(self.FC_2(out_fc_1)).view(b, c, 1, 1)
        return out_fc_2 * inputs


class AL_En_Module(nn.Module):
    def __init__(self, channel_in_size):
        super().__init__()
        self.Channel_attention = AL_CH_Module(channel_in_size)

    def forward(self, inputs, attention):
        return self.Channel_attention(inputs, attention)


class AL_De_Module(nn.Module):
    def __init__(self, channel_in_size):
        super().__init__()
        self.Channel_attention = AL_CH_Module(channel_in_size)

    def forward(self, inputs, attention):
        b = inputs.shape[0]
        attention = attention.view(b, -1)
        return self.Channel_attention(inputs, attention)


# ── encoder / decoder networks ───────────────────────────────────────────────

class Attention_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.FL_Module_1 = FL_En_Module(3, 256, 9, 2, 4, 'prelu')
        self.AL_Module_1 = AL_En_Module(256)
        self.FL_Module_2 = FL_En_Module(256, 256, 5, 2, 2, 'prelu')
        self.AL_Module_2 = AL_En_Module(256)
        self.FL_Module_3 = FL_En_Module(256, 256, 5, 1, 2, 'prelu')
        self.AL_Module_3 = AL_En_Module(256)
        self.FL_Module_4 = FL_En_Module(256, 256, 5, 1, 2, 'prelu')
        self.AL_Module_4 = AL_En_Module(256)
        self.FL_Module_5 = FL_En_Module(256, args.tcn, 5, stride=1, padding=2)

    def forward(self, x, attention):
        x = self.AL_Module_1(self.FL_Module_1(x), attention)
        x = self.AL_Module_2(self.FL_Module_2(x), attention)
        x = self.AL_Module_3(self.FL_Module_3(x), attention)
        x = self.AL_Module_4(self.FL_Module_4(x), attention)
        return self.FL_Module_5(x)

    def load_pretrained_weights(self, checkpoint_path):
        return load_pretrained_weights(self, checkpoint_path, "attention_encoder.")


class Attention_Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.FL_De_Module_1 = FL_De_Module(args.tcn, 256, 5, stride=1, padding=2, out_padding=0, activation='prelu')
        self.AL_De_module_1 = AL_De_Module(256)
        self.FL_De_Module_2 = FL_De_Module(256, 256, 5, stride=1, padding=2, out_padding=0, activation='prelu')
        self.AL_De_module_2 = AL_De_Module(256)
        self.FL_De_Module_3 = FL_De_Module(256, 256, 5, stride=1, padding=2, out_padding=0, activation='prelu')
        self.AL_De_module_3 = AL_De_Module(256)
        self.FL_De_Module_4 = FL_De_Module(256, 256, 5, stride=2, padding=2, out_padding=1, activation='prelu')
        self.AL_De_module_4 = AL_De_Module(256)
        self.FL_De_Module_5 = FL_De_Module(256, 3, 9, stride=2, padding=4, out_padding=1, activation='sigmoid')

    def forward(self, x, attention):
        x = self.AL_De_module_1(self.FL_De_Module_1(x), attention)
        x = self.AL_De_module_2(self.FL_De_Module_2(x), attention)
        x = self.AL_De_module_3(self.FL_De_Module_3(x), attention)
        x = self.AL_De_module_4(self.FL_De_Module_4(x), attention)
        return self.FL_De_Module_5(x)

    def load_pretrained_weights(self, checkpoint_path):
        return load_pretrained_weights(self, checkpoint_path, "attention_decoder.")


# ── helpers ──────────────────────────────────────────────────────────────────

class Args:
    """Minimal arg container for the networks (only `tcn` is used)."""

    def __init__(self, tcn=16):
        self.tcn = tcn


def load_pretrained_weights(module, checkpoint_path, prefix):
    """Load weights whose keys start with `prefix` from a training checkpoint
    (``checkpoint['net']``) into `module`, stripping the prefix first."""
    if not os.path.exists(checkpoint_path):
        print(f"adjscc.nn: Checkpoint not found: {checkpoint_path}")
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        full_state = checkpoint["net"] if "net" in checkpoint else checkpoint
        filtered = {k.replace(prefix, ''): v for k, v in full_state.items() if k.startswith(prefix)}
        module.load_state_dict(filtered, strict=False)
        print(f"adjscc.nn: Loaded {len(filtered)} weight tensors (prefix '{prefix}').")
        return True
    except Exception as e:
        print(f"adjscc.nn ERROR loading weights: {e}")
        return False


def powerConstraint(channel_input, P=1):
    """Normalize channel symbols so average power per symbol equals P."""
    import numpy as np
    if len(channel_input) == 0:
        return channel_input
    energy = torch.sum(torch.square(torch.abs(channel_input)))
    if energy == 0:
        return channel_input
    normalization_factor = np.sqrt(len(channel_input) * P) / torch.sqrt(energy)
    return channel_input * normalization_factor
