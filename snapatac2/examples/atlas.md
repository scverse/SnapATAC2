# Atlas-Scale Human Chromatin Accessibility Analysis

Use this workflow to process many scATAC-seq fragment files into a human chromatin accessibility atlas. The pipeline imports samples, computes tile matrices, removes doublets, builds an AnnDataSet, performs spectral embedding and batch correction, clusters cells, visualizes the atlas, and optionally subclusters one major cluster.

## Prerequisites

- Download fragment files from the Human Cell Atlas-style dataset source before running the script.
- Use backed files for both per-sample AnnData objects and the combined AnnDataSet.
- Use a larger tile size, such as 5000 bp, for atlas-scale workflows to reduce memory and runtime.
- Make cell barcodes unique across samples before combining data.

## Complete Script

```python
import os

import numpy as np
import snapatac2 as snap


# 1. Configure input and output locations.
data_dir = "/path/to/fragment/files"
output_dir = "h5ad_output"
os.makedirs(output_dir, exist_ok=True)

fragment_files = [
    os.path.join(data_dir, filename)
    for filename in os.listdir(data_dir)
    if filename.endswith(".bed.gz")
]

# 2. Import each fragment file into a backed AnnData object.
outputs = []
for fragment_file in fragment_files:
    sample_name = os.path.basename(fragment_file).split(".bed.gz")[0]
    outputs.append(os.path.join(output_dir, sample_name + ".h5ad"))

adatas = snap.pp.import_fragments(
    fragment_files,
    file=outputs,
    genome=snap.genome.hg38,
    min_num_fragments=1000,
    min_tsse=7,
)

# 3. Build sample-level tile matrices, select features, and remove doublets.
snap.pp.add_tile_matrix(adatas, bin_size=5000)
snap.pp.select_features(adatas)
snap.pp.scrublet(adatas)
snap.pp.filter_doublets(adatas)

# 4. Combine samples into an AnnDataSet and make barcodes unique.
adataset = snap.AnnDataSet(
    adatas=[(adata.filename.split("/")[-1].split(".h5ad")[0], adata) for adata in adatas],
    filename="data.h5ads",
)
adataset.obs_names = np.array(adataset.obs["sample"]) + "+" + np.array(adataset.obs_names)

# 5. Select atlas-level features and compute the initial embedding.
snap.pp.select_features(adataset, n_features=250000)
snap.tl.spectral(adataset)

# 6. Correct sample effects while preserving broad tissue groups.
adataset.obs["tissue"] = [sample.split(":")[0] for sample in adataset.obs["sample"]]
snap.pp.mnc_correct(
    adataset,
    batch="sample",
    groupby="tissue",
    key_added="X_spectral",
)

# 7. Cluster and visualize the atlas.
snap.pp.knn(adataset)
snap.tl.leiden(adataset)
snap.tl.umap(adataset, n_comps=3)
snap.pl.umap(adataset, color="leiden", interactive=False)

# 8. Optional: convert to in-memory AnnData for subclustering one major cluster.
adata = adataset.to_adata()
adata_c0 = adata[adata.obs["leiden"] == "0"].copy()

snap.pp.select_features(adata_c0, n_features=250000)
snap.tl.spectral(adata_c0)
snap.pp.mnc_correct(
    adata_c0,
    batch="sample",
    groupby="tissue",
    key_added="X_spectral",
)
snap.pp.knn(adata_c0)
snap.tl.leiden(adata_c0)
snap.tl.umap(adata_c0)
snap.pl.umap(adata_c0, color="leiden", interactive=False)
snap.pl.umap(adata_c0, color="sample", interactive=False)
```

## Workflow Notes

- Atlas-scale analysis should process fragments per sample first, then combine backed objects with `snap.AnnDataSet`.
- `bin_size=5000` creates a coarser tile matrix that is more suitable for hundreds of thousands of cells.
- `min_num_fragments` and `min_tsse` are applied during fragment import to remove low-quality barcodes early.
- `groupby="tissue"` in `mnc_correct` lets correction respect broad biological groups while correcting sample-level effects.
- `adataset.to_adata()` materializes the combined dataset in memory; only do this when enough RAM is available.

## Key Outputs

- Per-sample `*.h5ad` files in `h5ad_output/`.
- `data.h5ads`: combined backed AnnDataSet.
- `adataset.obs["sample"]`: sample labels.
- `adataset.obs["tissue"]`: broad tissue labels parsed from sample names.
- `adataset.obs["leiden"]`: atlas-level clusters.
- `adataset.obsm["X_spectral"]` and UMAP coordinates: atlas embeddings.
- `adata_c0`: optional subclustered AnnData object for cluster 0.

## Common Mistakes

- Do not use duplicate barcodes across samples; prefix or otherwise make them unique.
- Do not call `to_adata()` on atlas-scale data unless memory is sufficient.
- Do not use the default 500-bp tiles for very large atlas-scale data unless runtime and memory are acceptable.
- Do not skip per-sample doublet filtering before constructing the atlas.
- Do not use hard-coded local paths from examples; replace `data_dir` with the actual fragment directory.
