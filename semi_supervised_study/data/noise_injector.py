"""
Label noise injector for the semi-supervised study.

Flips a controlled fraction of labels in the labeled set to simulate
noisy annotation. Only the labeled portion is affected; unlabeled data
stays untouched (its true labels are hidden anyway).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def inject_label_noise(
    y: NDArray[np.integer],
    noise_rate: float,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.integer], NDArray[np.bool_]]:
    """Flip ``noise_rate`` fraction of labels uniformly at random.

    Parameters
    ----------
    y : array of labels
    noise_rate : float in [0, 1)
    rng : Generator

    Returns
    -------
    y_noisy : modified labels
    flip_mask : boolean array — True where label was changed
    """
    if noise_rate <= 0:
        return y.copy(), np.zeros(len(y), dtype=bool)

    n = len(y)
    classes = np.unique(y)
    n_flip = max(1, int(n * noise_rate))

    flip_idx = rng.choice(n, size=n_flip, replace=False)
    y_noisy = y.copy()
    flip_mask = np.zeros(n, dtype=bool)

    for i in flip_idx:
        other = classes[classes != y[i]]
        if len(other) > 0:
            y_noisy[i] = rng.choice(other)
            flip_mask[i] = True

    return y_noisy, flip_mask


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_n, mask = inject_label_noise(y, 0.3, rng)
    print(f"Original: {y}")
    print(f"Noisy:    {y_n}")
    print(f"Flipped:  {mask.sum()} / {len(y)}")
