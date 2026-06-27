# Joint Embedding of PBMC Multiome RNA and ATAC Data

Use this workflow to analyze paired RNA and ATAC modalities from the 10x PBMC multiome dataset. The RNA and ATAC data are processed separately, then combined with `snap.tl.multi_spectral` to create a joint embedding.

## Prerequisites

- Use paired multiome data where RNA and ATAC objects have matching cell barcodes in the same order.
- Install `scanpy` for RNA preprocessing.
- Use `features=None` in spectral embedding when the input matrix is already the feature space to analyze.

## Complete Script

```python
import scanpy as sc
import snapatac2 as snap


# 1. Load and preprocess RNA data.
rna = snap.read(snap.datasets.pbmc10k_multiome(modality="RNA"), backed=None)
sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=3000)
rna = rna[:, rna.var.highly_variable]
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)

# 2. Embed RNA cells.
snap.tl.spectral(rna, features=None)
snap.tl.umap(rna)
snap.pl.umap(rna, color="cell_type", interactive=False, height=550)

# 3. Load and embed ATAC data.
atac = snap.read(snap.datasets.pbmc10k_multiome(modality="ATAC"), backed=None)
snap.tl.spectral(atac, features=None)
snap.tl.umap(atac)
snap.pl.umap(atac, color="cell_type", interactive=False, height=550)

# 4. Verify paired cells and compute a joint RNA+ATAC embedding.
assert (rna.obs_names == atac.obs_names).all()
embedding = snap.tl.multi_spectral([rna, atac], features=None)[1]

# 5. Store and visualize the joint embedding on the ATAC object.
atac.obsm["X_joint"] = embedding
snap.tl.umap(atac, use_rep="X_joint")
snap.pl.umap(atac, color="cell_type", interactive=False, height=550)
```

## Workflow Notes

- RNA preprocessing uses highly variable gene selection, total-count normalization, and log transformation.
- ATAC and RNA are embedded independently before joint embedding so each modality can be inspected on its own.
- `snap.tl.multi_spectral([rna, atac], features=None)` returns a joint representation aligned across modalities.
- The joint embedding is stored in `atac.obsm["X_joint"]` only as a convenient host object; the rows correspond to the shared cell order.

## Key Outputs

- `rna.obsm["X_spectral"]` and `rna.obsm["X_umap"]`: RNA-only embeddings.
- `atac.obsm["X_spectral"]` and `atac.obsm["X_umap"]`: ATAC-only embeddings.
- `atac.obsm["X_joint"]`: joint RNA+ATAC embedding.

## Common Mistakes

- Do not run `multi_spectral` until RNA and ATAC cell barcodes are aligned and ordered identically.
- Do not use an ATAC object with filtered cells that no longer match the RNA object.
- Do not skip RNA normalization before computing the RNA spectral embedding.
