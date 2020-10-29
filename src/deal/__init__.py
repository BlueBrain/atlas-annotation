"""The DEep AtLas module (DEAL)."""
import numpy as np
from warpme.base import DisplacementField

from .version import __version__  # noqa


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
