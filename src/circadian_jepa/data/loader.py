import warnings

import anndata
import scanpy as sc


def load_and_preprocess(
    adata: anndata.AnnData,
    gene_list: list[str],
    min_cells_per_gene: int = 10,
    log_normalize: bool = True,
) -> anndata.AnnData:
    """Subset to gene_list, store raw counts, and optionally normalize.

    Raw counts are saved to .layers['counts'] before any normalization.
    .X is set to log1p-normalized values when log_normalize=True.
    """
    available = [g for g in gene_list if g in adata.var_names]
    missing = [g for g in gene_list if g not in adata.var_names]
    if missing:
        warnings.warn(f"{len(missing)} gene(s) not found in adata: {missing}", stacklevel=2)
    if not available:
        raise ValueError("None of the requested genes are present in adata.var_names.")

    adata = adata[:, available].copy()
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    if "counts" not in adata.layers:
        x = adata.X
        adata.layers["counts"] = x.toarray() if hasattr(x, "toarray") else x.copy()

    if log_normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    return adata
