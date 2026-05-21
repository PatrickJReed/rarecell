"""I/O utilities for h5ad handling and checkpoints."""

from pathlib import Path

import anndata as ad
import pandas as pd


def save_h5ad(adata: ad.AnnData, path: str | Path) -> None:
    """Save AnnData object to h5ad file with h5ad-compatible sanitization.

    Sanitizes adata.uns entries and obs/var column dtypes to ensure compatibility
    with h5ad/h5py serialization.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object to save.
    path : str or Path
        The output file path.
    """
    path = Path(path)
    # Sanitize uns: replace the whole dict with stringified version
    adata.uns = _stringify_dict_keys(adata.uns)
    _sanitize_obs_columns(adata)
    adata.write_h5ad(path)


def load_h5ad(path: str | Path) -> ad.AnnData:
    """Load an AnnData object from h5ad file.

    Parameters
    ----------
    path : str or Path
        The input file path.

    Returns
    -------
    adata : ad.AnnData
        The loaded AnnData object.
    """
    return ad.read_h5ad(path)


def setup_figure_dir(out_dir: str | Path) -> Path:
    """Create and return a figure output directory.

    Parameters
    ----------
    out_dir : str or Path
        The directory to create.

    Returns
    -------
    Path
        The created directory path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_checkpoint(adata: ad.AnnData, name: str) -> None:
    """Save a checkpoint to ~/.cache/rarecell/checkpoints/{name}.h5ad.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object to checkpoint.
    name : str
        The checkpoint name (without extension).
    """
    cache_dir = Path.home() / ".cache" / "rarecell" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.h5ad"
    save_h5ad(adata, path)


def load_checkpoint(name: str) -> ad.AnnData | None:
    """Load a checkpoint from ~/.cache/rarecell/checkpoints/{name}.h5ad if it exists.

    Parameters
    ----------
    name : str
        The checkpoint name (without extension).

    Returns
    -------
    adata : ad.AnnData or None
        The loaded AnnData object if found, else None.
    """
    cache_dir = Path.home() / ".cache" / "rarecell" / "checkpoints"
    path = cache_dir / f"{name}.h5ad"
    if path.exists():
        return load_h5ad(path)
    return None


def _stringify_dict_keys(d: dict) -> dict:
    """Recursively convert all non-string dict keys to strings.

    h5ad/h5py requires all mapping keys to be strings. DataFrame.to_dict()
    produces integer row-index keys that trigger AttributeError on write.

    Parameters
    ----------
    d : dict
        The dictionary to convert.

    Returns
    -------
    dict
        Dictionary with string keys throughout, with sanitized DataFrames.
    """
    out = {}
    for k, v in d.items():
        str_k = str(k) if not isinstance(k, str) else k
        if isinstance(v, dict):
            out[str_k] = _stringify_dict_keys(v)
        elif isinstance(v, pd.DataFrame):
            _sanitize_df_columns(v)
            out[str_k] = v
        else:
            out[str_k] = v
    return out


def _sanitize_df_columns(df: pd.DataFrame) -> None:
    """Convert object-dtype columns in a DataFrame to string for h5ad.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to sanitize (modified in place).
    """
    for col in df.columns:
        if df[col].dtype == object:
            try:
                # Try numeric first
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Fall back to string
                df[col] = df[col].astype(str)


def _sanitize_obs_columns(adata: ad.AnnData) -> None:
    """Convert obs columns with non-serializable dtypes for h5ad.

    Handles Float64 (nullable float), categorical with NaN, and object columns.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object to sanitize (modified in place).
    """
    for col in adata.obs.columns:
        dtype_str = str(adata.obs[col].dtype)
        if dtype_str == "Float64":
            adata.obs[col] = adata.obs[col].astype("float64")
        elif dtype_str == "category" and adata.obs[col].isna().any():
            adata.obs[col] = adata.obs[col].astype(str).replace("nan", "Unknown")
        elif dtype_str == "object" and adata.obs[col].isna().any():
            adata.obs[col] = adata.obs[col].fillna("Unknown").astype(str)
