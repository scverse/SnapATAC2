SnapATAC2: A Python/Rust package for single-cell epigenomics analysis
=====================================================================

![PyPI](https://img.shields.io/pypi/v/snapatac2)
![PyPI - Downloads](https://img.shields.io/pypi/dm/snapatac2)
![Continuous integration](https://github.com/scverse/SnapATAC2/workflows/test-python-package/badge.svg)
![GitHub Repo stars](https://img.shields.io/github/stars/scverse/SnapATAC2?style=social)

> [!TIP]
> Got raw fastq files? Check out our new single-cell preprocessing package [precellar](https://github.com/regulatory-genomics/precellar)!

SnapATAC2 is a flexible, versatile, and scalable single-cell omics analysis framework, featuring:

- Scale to more than 10 million cells.
- Blazingly fast preprocessing tools for BAM to fragment files conversion and count matrix generation.
- Matrix-free spectral embedding algorithm that is applicable to a wide range of single-cell omics data, including single-cell ATAC-seq, single-cell RNA-seq, single-cell Hi-C, and single-cell methylation.
- Efficient and scalable co-embedding algorithm for single-cell multi-omics data integration.
- End-to-end analysis pipeline for single-cell ATAC-seq data, including preprocessing, dimension reduction, clustering, data integration, peak calling, differential analysis, motif analysis, regulatory network analysis.
- Seamless integration with other single-cell analysis packages such as Scanpy.
- Implementation of fully backed AnnData.

Documentation
-------------

- **Full Documentation**: https://scverse.org/SnapATAC2/
- **Installation instructions**: https://scverse.org/SnapATAC2/install.html
- **Tutorial/Demo**: https://scverse.org/SnapATAC2/tutorials/index.html

How to cite
-----------

Zhang, K., Zemke, N. R., Armand, E. J. & Ren, B. (2024).
A fast, scalable and versatile tool for analysis of single-cell omics data.
Nature Methods, 1–11. https://doi.org/10.1038/s41592-023-02139-9

[//]: # (numfocus-fiscal-sponsor-attribution)

SnapATAC2 is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>