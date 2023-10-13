from somd2.io import (
    dataframe_to_parquet,
    dict_to_yaml,
    parquet_append,
    parquet_to_dataframe,
)
import pytest


def test_parquet():
    import random
    import pandas as pd
    import tempfile
    from pathlib import Path

    outdir = tempfile.TemporaryDirectory()
    full_fname = Path(outdir.name) / "output.parquet"
    data = {
        "Column1": [random.uniform(0, 1) for _ in range(1000)],
        "Column2": [random.uniform(0, 100) for _ in range(1000)],
        "Column3": [float(random.choice([True, False])) for _ in range(1000)],
        "Column4": [random.uniform(0, 1) for _ in range(1000)],
    }
    df = pd.DataFrame(data)
    meta = {"one": 1, "two": 2}
    dataframe_to_parquet(df, meta, filepath=outdir.name)
    data1 = {
        "Column1": [random.uniform(0, 1) for _ in range(1000)],
        "Column2": [random.uniform(0, 100) for _ in range(1000)],
        "Column3": [float(random.choice([True, False])) for _ in range(1000)],
        "Column4": [random.uniform(0, 1) for _ in range(1000)],
    }
    df1 = pd.DataFrame(data1)
    df_full = pd.concat([df, df1], ignore_index=True)
    parquet_append(full_fname, pd.DataFrame(data1))
    df_back, meta_back = parquet_to_dataframe(full_fname)
    assert df_back.equals(df_full)
    assert meta_back == meta
    outdir.cleanup()
