"""Frequency-domain transforms used by ANAF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt


@dataclass(frozen=True)
class DWTTransform:
    """Convert a time window to the legacy ANAF wavelet representation."""

    wavelet: str = "db1"
    level: int = 3

    def __post_init__(self) -> None:
        try:
            wavelet = pywt.Wavelet(self.wavelet)
        except ValueError as exc:
            raise ValueError(f"unknown wavelet: {self.wavelet}") from exc
        if self.level < 1:
            raise ValueError("wavelet level must be positive")
        object.__setattr__(self, "_wavelet", wavelet)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        """Return a float32 array with the same shape as ``window``."""

        values = np.asarray(window)
        if values.ndim != 2:
            raise ValueError(f"DWT input must have shape [time, channels], got {values.shape}")
        if values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("DWT input must contain at least two timestamps and one channel")
        maximum_level = pywt.dwt_max_level(values.shape[0], self._wavelet.dec_len)
        if self.level > maximum_level:
            raise ValueError(
                f"wavelet level {self.level} exceeds maximum {maximum_level} "
                f"for a window of length {values.shape[0]}"
            )

        transformed = np.zeros(values.shape, dtype=np.float32)
        for channel in range(values.shape[1]):
            coefficients = pywt.wavedec(
                values[:, channel], self._wavelet, mode="symmetric", level=self.level
            )
            flattened = np.concatenate(coefficients)
            retained = min(values.shape[0], flattened.shape[0])
            transformed[:retained, channel] = flattened[:retained]
        return transformed
