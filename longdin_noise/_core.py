import numba
import numpy as np

# <-- FUNCTIONS -->


def _validate_parameters(
    size: tuple[int, int], density: float, iterations: int, radius: int
) -> None:
    if size[0] < 1 or size[1] < 1:
        raise ValueError(
            f"Both dimensions of size must be positive integers, got {size}."
        )

    if density > 1 or density <= 0:
        raise ValueError(
            f"Density value must not be 0 or below and mustn't exceed 1, got {density}."
        )

    if iterations < 1:
        raise ValueError(
            f"Number of iterations cannot be less than 1, got {iterations}."
        )

    if radius < 1:
        raise ValueError(f"Radius cannot be below 1, got {radius}.")


def generate_mono(
    size: tuple[int, int],
    seed: int,
    density: float,
    iterations: int,
    radius: int = 8,
) -> np.ndarray:
    """
    Generates a Longdin Noise array where each pixel carries one float value in the range 0..1.
    Returns an array of shape (height, width).

    Parameters:
        size       - (width, height) dimensions of the output array.
        seed       - Seed for numpy's random number generator. Same seed and parameters always produce the same result.
        density    - Proportion of pixels initially active, in the range 0..1. Low values (e.g. 0.005) produce sparse,
                     sweeping patterns. High values (e.g. 0.5) produce dense, fine-grained noise.
        iterations - Number of times the diffusion algorithm runs. More iterations spread influence further and
                     produce smoother, more blended results.
        radius     - Spatial reach of each active pixel's influence in pixels. Larger values produce broader,
                     softer transitions. Defaults to 8.

    'Mono' means each pixel carries one float value which is propagated by the pixel's influence vector.
    """
    _validate_parameters(size, density, iterations, radius)

    noise = _Noise(size, seed, density, radius, 1)
    result = noise.run(iterations)

    return result.squeeze(axis=-1)


def generate_bi(
    size: tuple[int, int],
    seed: int,
    density: float,
    iterations: int,
    radius: int = 8,
) -> np.ndarray:
    """
    Generates a Longdin Noise array where each pixel carries two float values in the range 0..1.
    Returns an array of shape (height, width, 2).

    Parameters:
        size       - (width, height) dimensions of the output array.
        seed       - Seed for numpy's random number generator. Same seed and parameters always produce the same result.
        density    - Proportion of pixels initially active, in the range 0..1. Low values (e.g. 0.005) produce sparse,
                     sweeping patterns. High values (e.g. 0.5) produce dense, fine-grained noise.
        iterations - Number of times the diffusion algorithm runs. More iterations spread influence further and
                     produce smoother, more blended results.
        radius     - Spatial reach of each active pixel's influence in pixels. Larger values produce broader,
                     softer transitions. Defaults to 8.

    'Bi' means each pixel carries two independent float values, both propagated by the pixel's influence vector.
    """
    _validate_parameters(size, density, iterations, radius)

    noise = _Noise(size, seed, density, radius, 2)
    result = noise.run(iterations)

    return result


def generate_tri(
    size: tuple[int, int],
    seed: int,
    density: float,
    iterations: int,
    radius: int = 8,
) -> np.ndarray:
    """
    Generates a Longdin Noise array where each pixel carries three float values in the range 0..1.
    Returns an array of shape (height, width, 3).

    Parameters:
        size       - (width, height) dimensions of the output array.
        seed       - Seed for numpy's random number generator. Same seed and parameters always produce the same result.
        density    - Proportion of pixels initially active, in the range 0..1. Low values (e.g. 0.005) produce sparse,
                     sweeping patterns. High values (e.g. 0.5) produce dense, fine-grained noise.
        iterations - Number of times the diffusion algorithm runs. More iterations spread influence further and
                     produce smoother, more blended results.
        radius     - Spatial reach of each active pixel's influence in pixels. Larger values produce broader,
                     softer transitions. Defaults to 8.

    'Tri' means each pixel carries three independent float values, all propagated by the pixel's influence vector.
    """
    _validate_parameters(size, density, iterations, radius)

    noise = _Noise(size, seed, density, radius, 3)
    result = noise.run(iterations)

    return result


@numba.njit(parallel=True)
def _process_cells(
    vectors,
    influences,
    vectors_snap,
    influences_snap,
    height,
    width,
    active_positions,
    radius,
):
    for i in numba.prange(len(active_positions)):
        r = active_positions[i, 0]
        c = active_positions[i, 1]

        r_min = max(0, r - radius)
        r_max = min(height, r + radius)
        c_min = max(0, c - radius)
        c_max = min(width, c + radius)

        for lr in range(r_min, r_max):
            for lc in range(c_min, c_max):
                dr = lr - r
                dc = lc - c
                distance = (dr**2 + dc**2) ** 0.5
                dot = dr * influences_snap[r, c, 0] + dc * influences_snap[r, c, 1]
                if dot <= 0 or distance >= radius:
                    continue
                strength = max(0.0, min(1.0, 1 - (distance / radius)))

                for ch in range(vectors.shape[2]):
                    vectors[lr, lc, ch] += strength * (
                        vectors_snap[r, c, ch] - vectors[lr, lc, ch]
                    )

                influences[lr, lc, 0] += strength * (
                    influences_snap[r, c, 0] - influences[lr, lc, 0]
                )
                influences[lr, lc, 1] += strength * (
                    influences_snap[r, c, 1] - influences[lr, lc, 1]
                )


# <-- CLASSES -->


class _Noise:
    def __init__(
        self,
        size: tuple[int, int],
        seed: int,
        density: float = 0.5,
        radius: int = 8,
        channels: int = 2,
    ) -> None:
        self._rng = np.random.default_rng(seed)

        self._active = self._rng.random((size[1], size[0])) < density
        self._influences = self._rng.uniform(-1, 1, (size[1], size[0], 2))
        self._vectors = self._rng.random((size[1], size[0], channels))

        self._size = size
        self._density = density
        self._seed = seed
        self._radius = radius

        self._vectors[~self._active] = 0
        self._influences[~self._active] = 0

    def _iterate(self) -> None:
        height, width = self._active.shape
        active_positions = np.argwhere(self._active)
        vectors_snap = self._vectors.copy()
        influences_snap = self._influences.copy()

        np.random.shuffle(active_positions)

        _process_cells(
            self._vectors,
            self._influences,
            vectors_snap,
            influences_snap,
            height,
            width,
            active_positions,
            self._radius,
        )

        self._active = np.any(self._vectors != 0, axis=-1)

    def run(self, iterations: int = 3):
        for i in range(iterations):
            self._iterate()
        return self._vectors
