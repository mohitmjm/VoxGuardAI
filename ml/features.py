"""Research features; intentionally separate from the raw-waveform AASIST baseline."""
import math
import torch


def log_spectrogram(samples, n_fft=512, hop_length=160):
    """Return log power bins x frames using centered Hann-window STFT."""
    x = torch.as_tensor(samples, dtype=torch.float32)
    if x.ndim != 1 or x.numel() < n_fft or not torch.isfinite(x).all():
        raise ValueError('Expected finite mono audio with at least n_fft samples.')
    spec = torch.stft(x, n_fft, hop_length, window=torch.hann_window(n_fft), return_complex=True)
    return spec.abs().square().clamp_min(1e-10).log()


def lfcc(samples, sample_rate=16000, n_filters=60, n_coefficients=20):
    """Linear-frequency cepstra, DCT-II; no deltas/normalization applied.

    Experimental feature definition, not an implicit replacement for the
    preprocessing of any pretrained checkpoint.
    """
    if sample_rate != 16000 or not 1 <= n_coefficients <= n_filters:
        raise ValueError('Expected 16 kHz and 1 <= n_coefficients <= n_filters.')
    power = log_spectrogram(samples).exp()
    freqs = torch.linspace(0, sample_rate / 2, power.shape[0])
    edges = torch.linspace(0, sample_rate / 2, n_filters + 2)
    bank = torch.stack([torch.minimum((freqs - edges[i]) / (edges[i+1] - edges[i]),
                                      (edges[i+2] - freqs) / (edges[i+2] - edges[i+1])).clamp_min(0)
                        for i in range(n_filters)])
    energies = (bank @ power).clamp_min(1e-10).log()
    k = torch.arange(n_coefficients)[:, None]
    n = torch.arange(n_filters)[None, :]
    dct = torch.cos(math.pi / n_filters * (n + .5) * k) * math.sqrt(2 / n_filters)
    dct[0] /= math.sqrt(2)
    return dct @ energies
