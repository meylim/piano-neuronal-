"""Loss functions for DDSP-Piano training.

MR-STFT: Multi-resolution short-time Fourier transform loss at 6 scales.
ReverbRegularizer: L1 on reverb IR.
InharmonicityLoss: Penalises negative B coefficients.

Uses torchaudio.transforms.Spectrogram for efficient GPU computation.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
from typing import List

from piano_neuronal.s2_baseline.config import (
    MR_STFT_N_FFTS, MR_STFT_MAG_WEIGHT, MR_STFT_LOGMAG_WEIGHT,
    REVERB_REG_WEIGHT, INHARM_LOSS_WEIGHT, SAMPLE_RATE
)


class SSSLoss(nn.Module):
    """Single-scale spectral loss using STFT magnitude.

    Computes L1 on magnitude and log-magnitude spectrograms.

    Args:
        n_fft: FFT size.
        hop_length: hop size (default: n_fft // 4 for 75% overlap).
        mag_weight: weight for magnitude loss.
        logmag_weight: weight for log-magnitude loss.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = None,
        mag_weight: float = MR_STFT_MAG_WEIGHT,
        logmag_weight: float = MR_STFT_LOGMAG_WEIGHT,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.mag_weight = mag_weight
        self.logmag_weight = logmag_weight

        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=n_fft,
            power=1.0,  # magnitude spectrogram
            normalized=False,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute single-scale spectral loss.

        Args:
            y_pred: (B, T) predicted audio.
            y_true: (B, T) target audio.

        Returns:
            Scalar loss.
        """
        # Move spectrogram to same device as input data
        self.spectrogram.to(y_pred.device)
        S_pred = self.spectrogram(y_pred)  # (B, F, T')
        S_true = self.spectrogram(y_true)

        loss = torch.tensor(0.0, device=y_pred.device)

        if self.mag_weight > 0:
            loss = loss + self.mag_weight * torch.mean(torch.abs(S_pred - S_true))

        if self.logmag_weight > 0:
            log_pred = torch.log(S_pred + 1e-7)
            log_true = torch.log(S_true + 1e-7)
            loss = loss + self.logmag_weight * torch.mean(torch.abs(log_pred - log_true))

        return loss


class MSSLoss(nn.Module):
    """Multi-resolution STFT loss (MR-STFT).

    Sums SSSLoss across multiple FFT sizes: [2048, 1024, 512, 256, 128, 64].
    This is the primary loss function for DDSP-Piano.

    Args:
        n_ffts: list of FFT sizes.
        mag_weight: weight for magnitude loss (applied to all scales).
        logmag_weight: weight for log-magnitude loss (applied to all scales).
    """

    def __init__(
        self,
        n_ffts: List[int] = None,
        mag_weight: float = MR_STFT_MAG_WEIGHT,
        logmag_weight: float = MR_STFT_LOGMAG_WEIGHT,
    ):
        super().__init__()
        n_ffts = n_ffts or MR_STFT_N_FFTS
        self.spectral_losses = nn.ModuleList([
            SSSLoss(n_fft=n, mag_weight=mag_weight, logmag_weight=logmag_weight)
            for n in n_ffts
        ])

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute multi-resolution spectral loss.

        Args:
            y_pred: (B, T) predicted audio.
            y_true: (B, T) target audio.

        Returns:
            Scalar loss summed across all resolutions.
        """
        loss = torch.tensor(0.0, device=y_pred.device)
        for ssl in self.spectral_losses:
            loss = loss + ssl(y_pred, y_true)
        return loss


class ReverbRegularizer(nn.Module):
    """L1 regularizer on reverb impulse response.

    Encourages the learned reverb to be sparse and well-behaved.

    Args:
        weight: regularization weight (default 0.01).
    """

    def __init__(self, weight: float = REVERB_REG_WEIGHT):
        super().__init__()
        self.weight = weight

    def forward(self, reverb_ir: torch.Tensor) -> torch.Tensor:
        """Compute reverb regularization loss.

        Args:
            reverb_ir: (B, ir_length) — learned impulse response.

        Returns:
            Scalar loss.
        """
        # L1 on sorted IR values (encourages sparsity)
        sorted_ir = torch.sort(reverb_ir, dim=-1).values
        return self.weight * torch.mean(torch.abs(sorted_ir))


class InharmonicityLoss(nn.Module):
    """Penalises negative inharmonicity coefficients.

    Physical constraint: B should be >= 0 (inharmonicity makes overtones
    sharper, not flatter). This loss prevents the model from learning
    unphysical negative B values.

    Args:
        weight: loss weight (default 10.0).
    """

    def __init__(self, weight: float = INHARM_LOSS_WEIGHT):
        super().__init__()
        self.weight = weight

    def forward(self, inharm_coef: torch.Tensor) -> torch.Tensor:
        """Compute inharmonicity loss.

        Args:
            inharm_coef: (B, ...) — predicted B coefficients.

        Returns:
            Scalar loss penalising negative B.
        """
        # ReLU-like penalty: only penalise B < 0
        negative_b = torch.clamp(-inharm_coef, min=0.0)
        return self.weight * torch.mean(negative_b)


class HybridLoss(nn.Module):
    """Combined loss for DDSP-Piano training.

    Total loss = MR-STFT + reverb_regulariser + inharmonicity_loss

    Args:
        n_ffts: FFT sizes for MR-STFT.
        mag_weight: magnitude loss weight.
        logmag_weight: log-magnitude loss weight.
        reverb_weight: reverb regularisation weight.
        inharm_weight: inharmonicity loss weight.
    """

    def __init__(
        self,
        n_ffts: List[int] = None,
        mag_weight: float = MR_STFT_MAG_WEIGHT,
        logmag_weight: float = MR_STFT_LOGMAG_WEIGHT,
        reverb_weight: float = REVERB_REG_WEIGHT,
        inharm_weight: float = INHARM_LOSS_WEIGHT,
    ):
        super().__init__()
        self.mr_stft = MSSLoss(n_ffts=n_ffts, mag_weight=mag_weight,
                                logmag_weight=logmag_weight)
        self.reverb_reg = ReverbRegularizer(weight=reverb_weight)
        self.inharm_loss = InharmonicityLoss(weight=inharm_weight)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reverb_ir: torch.Tensor,
        inharm_coef: torch.Tensor,
    ) -> tuple:
        """Compute total hybrid loss.

        Args:
            y_pred: (B, T) predicted audio.
            y_true: (B, T) target audio.
            reverb_ir: (B, ir_length) reverb impulse response.
            inharm_coef: (B, ...) inharmonicity coefficients.

        Returns:
            total_loss: scalar total loss.
            loss_dict: dict of individual losses for logging.
        """
        stft_loss = self.mr_stft(y_pred, y_true)
        reverb_loss = self.reverb_reg(reverb_ir)
        inharm_loss = self.inharm_loss(inharm_coef)

        total = stft_loss + reverb_loss + inharm_loss

        loss_dict = {
            "mr_stft": stft_loss.item(),
            "reverb_reg": reverb_loss.item(),
            "inharm": inharm_loss.item(),
            "total": total.item(),
        }

        return total, loss_dict