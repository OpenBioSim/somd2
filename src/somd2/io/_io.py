__all__ = ["dataframe_to_parquet", "dict_to_yaml", "parquet_append"]
import pyarrow
from pathlib import Path as _Path
import pyarrow as _pa
import pyarrow.parquet as _pq
import json
import pandas as _pd


def dataframe_to_parquet(df, metadata, filepath=None):
    """
    Save a dataframe to parquet format with custom metadata.
    Parameters:
    -----------
    df: pandas.DataFrame
        The dataframe to be saved. In this case containing info required for FEP calculation
    metadata: dict
        Dictionary containing metadata to be saved with the dataframe.
        Currently just temperature and lambda value.
    filepath: str or pathlib.PosixPath
        The of the parent directory in to which the parquet file will be saved.
        If None, save to current working directory.
    """

    if filepath is None:
        filepath = _Path.cwd()
    elif isinstance(filepath, str):
        filepath = _Path(filepath)
    custom_meta_key = "somd2"
    table = _pa.Table.from_pandas(df)
    custom_meta_json = json.dumps(metadata)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode(): custom_meta_json.encode(),
        **existing_meta,
    }
    table = table.replace_schema_metadata(combined_meta)
    filename = f"Lam_{metadata['lambda'].replace('.','')[:5]}_T_{metadata['temperature']}.parquet"
    _pq.write_table(table, filepath / filename)


def dict_to_yaml(data_dict, file_path, filename="config.yaml"):
    """
    Write a dictionary to a YAML file.
    Parameters:
    -----------
    data_dict: dict
        The dictionary to be written to a YAML file.
    file_path: str or pathlib.PosixPath
        The path to the YAML file to be written.
    filename: str
        The name of the YAML file to be written (default 'config.yaml').
    """
    import yaml as _yaml

    try:
        file_path = _Path(file_path) / filename
        # Ensure the parent directory for the file exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Open the file in write mode and write the dictionary as YAML
        with file_path.open("w") as yaml_file:
            _yaml.dump(
                data_dict,
                yaml_file,
            )
        print("config written")
    except Exception as e:
        print(f"Error writing the dictionary to {file_path}: {e}")


def parquet_append(filepath: _Path or str, df: _pd.DataFrame) -> None:
    """
    Append to dataframe to existing .parquet file. Reads original .parquet file in, appends new dataframe, writes new .parquet file out.

    Parameters
    ----------
    filepath: Filepath for parquet file.
    df: Pandas dataframe to append. Must be same schema as original.
    """
    import shutil

    try:
        table_original_file = _pq.read_table(
            source=str(filepath), pre_buffer=False, use_threads=True, memory_map=True
        )  # Use memory map for speed.
        table_to_append = _pa.Table.from_pandas(df)
        table_to_append = table_to_append.cast(
            table_original_file.schema
        )  # Attempt to cast new schema to existing, e.g. datetime64[ns] to datetime64[us] (may throw otherwise).

        temp_file = str(filepath) + "_temp"  # Temporary file to write to

        # Writing to a temporary file
        with _pq.ParquetWriter(temp_file, table_original_file.schema) as temp_writer:
            temp_writer.write_table(table_original_file)
            temp_writer.write_table(table_to_append)

        # Atomic operation to ensure data integrity
        shutil.move(temp_file, filepath)

    except (FileNotFoundError, IOError, Exception) as e:
        # Handle specific exceptions that might occur during file operations
        print(f"Error occurred: {e}")
