"""
Small tensor and harmonics utilities.
"""

import numpy as np

# Levi-Civita symbol
EPS = np.zeros((3, 3, 3))
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    EPS[_i, _j, _k], EPS[_i, _k, _j] = 1, -1


def stf(M):
    """Symmetric trace-free part of the (..., 3, 3) tensor M"""
    M = 0.5 * (M + np.swapaxes(M, -1, -2))
    tr = np.trace(M, axis1=-2, axis2=-1)
    return M - tr[..., None, None] * np.eye(3) / 3


def sYlm2(m, th, ph):
    """Spin-weighted spherical harmonics, s = -2, l = 2"""
    c, s = np.cos(th), np.sin(th)
    amp = {
        2: np.sqrt(5 / (64 * np.pi)) * (1 + c) ** 2,
        1: np.sqrt(5 / (16 * np.pi)) * s * (1 + c),
        0: np.sqrt(15 / (32 * np.pi)) * s**2,
        -1: np.sqrt(5 / (16 * np.pi)) * s * (1 - c),
        -2: np.sqrt(5 / (64 * np.pi)) * (1 - c) ** 2,
    }[m]
    return amp * np.exp(1j * m * ph)


def rot_z(x):
    """Rotation matrices about z by the angles x: shape (N, 3, 3)"""
    c, s, z, o = np.cos(x), np.sin(x), np.zeros_like(x), np.ones_like(x)
    return np.array([[c, -s, z], [s, c, z], [z, z, o]]).transpose(2, 0, 1)


def rot_y(x):
    """Rotation matrices about y by the angles x: shape (N, 3, 3)"""
    c, s, z, o = np.cos(x), np.sin(x), np.zeros_like(x), np.ones_like(x)
    return np.array([[c, z, s], [z, o, z], [-s, z, c]]).transpose(2, 0, 1)
