"""
Longdin Noise
---
A library for the Longdin Noise noise type.

Longdin Noise uses per-pixel influence vectors along with pixel values to create a diffusion driven noise.
Pixel values can take on 3 different numbers of channels (1 | 2 | 3)
"""

from ._core import generate_bi, generate_mono, generate_tri

__all__ = ["generate_mono", "generate_bi", "generate_tri"]
