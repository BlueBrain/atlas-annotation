"""The DEep AtLas module (DEAL)."""
import pathlib

import numpy as np
from warpme.base import DisplacementField

from deal.utils import load_nrrd

from .version import __version__  # noqa: F401


def dfs_to_deltas(dfs):
    """Transform displacement fields to deltas.

    Parameters
    ----------
    dfs : iterable
        The displacement fields

    Returns
    -------
    deltas : np.ndarray
        The array with the deltas with shape (n_dfs, 2, height, width).
    """
    deltas = np.stack([[df.delta_x, df.delta_y] for df in dfs])
    return deltas


def deltas_to_dfs(deltas):
    """Transform deltas to displacement fields.

    Parameters
    ----------
    deltas : np.ndarray
        The array with deltas. Has to have shape (n_dfs, 2, height, width).

    Returns
    -------
    dfs : list
        The resulting displacement fields.
    """
    dfs = [DisplacementField(*delta) for delta in deltas]
    return dfs


def safe_dfs(file, dfs):
    """Save displacement fields to disk.

    Parameters
    ----------
    file : str or pathlib.Path
        The output file path.

    dfs : iterable
        The displacement fields.
    """
    np.save(file, dfs_to_deltas(dfs))


def load_dfs(file):
    """Load displacement fields from disk.

    Parameters
    ----------
    file : str
        The input file.

    Returns
    -------
    dfs : list
        The displacement fields loaded from file.
    """
    deltas = np.load(file)
    dfs = deltas_to_dfs(deltas)
    return dfs


def load_volume(volume_path, normalize=True):
    """Load volume from the given path.

    Parameters
    ----------
    volume_path : str or pathlib.Path
        Path of file to load

    normalize : bool
        If True, the numpy is normalized between 0 and 1.

    Returns
    -------
    img : np.ndarray or None
        Loaded path
    """
    volume_path = pathlib.Path(volume_path)
    if volume_path.exists():
        if volume_path.suffix == ".nrrd":
            img = load_nrrd(str(volume_path), norm=normalize)
        elif volume_path.suffix == ".npy":
            img = np.load(volume_path)
            if volume_path:
                img = img.astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min())
        else:
            raise ValueError(f"The extension {volume_path.suffix} is not supported")
    else:
        img = None
    return img
