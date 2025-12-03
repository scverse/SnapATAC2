import snapatac2 as snap
import polars as pl
narrowpeak_cols = [
    "chrom", "start", "end", "name", "score", 
    "strand", "signal_value", "p_value", "q_value", "peak"
]
schema_types = {
    "chrom": pl.String,
    "start": pl.UInt64, 
    "end": pl.UInt64,
    "score": pl.UInt16,  # <--- THIS IS THE FIX
    "peak": pl.UInt64    # Good practice to define this too
}
peak1 = pl.read_csv(
            "/storage/zhangkaiLab/hanlitian/macrophage/script/utils/merge_peaks/test_data/B1-1_ATAC3_peaks.narrowPeak",
            separator="\t",
            has_header=False,
            comment_prefix="#",
            ignore_errors=True, # Skip malformed lines like track lines
            new_columns=narrowpeak_cols,
            schema_overrides=schema_types,
        )
peak2 = pl.read_csv(
            "/storage/zhangkaiLab/hanlitian/macrophage/script/utils/merge_peaks/test_data/B1-2_ATAC3_peaks.narrowPeak",
            separator="\t",
            has_header=False,
            comment_prefix="#",
            ignore_errors=True, # Skip malformed lines like track lines
            new_columns=narrowpeak_cols,
            schema_overrides=schema_types,
        )
def clean_narrowpeak(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        # 1. Clip score to 0-1000 (standard narrowPeak limit) to prevent overflow
        pl.col("score").clip(0, 1000).cast(pl.UInt16),
        
        # 2. Ensure coordinates are UInt64
        pl.col("start").cast(pl.UInt64),
        pl.col("end").cast(pl.UInt64),
        
        # 3. Ensure chrom is String
        pl.col("chrom").cast(pl.String)
    ])

# Apply the cleaning function to your existing loaded dataframes
peak1 = clean_narrowpeak(peak1)
peak2 = clean_narrowpeak(peak2)

# Re-create the dictionary
peaks_dict = {
    "B1-1": peak1,
    "B1-2": peak2,
}
peak_merge = snap.tl.merge_peaks(peaks_dict, snap.genome.hg38)
peak_merge