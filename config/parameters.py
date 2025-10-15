"""Pipeline parameters."""

PARAMETERS = {
    'n_preview_rows': 10,
    'percentile_min': 5,
    'percentile_max': 95,
    'compression': 'snappy',
    'engine': 'pyarrow',
    'chunk_size': 1_000_000,
}
