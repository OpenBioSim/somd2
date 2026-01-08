######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2026
#
# Authors: The OpenBioSim Team <team@openbiosim.org>
#
# SOMD2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SOMD2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SOMD2. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

__all__ = [
    "dataframe_to_parquet",
    "dict_to_yaml",
    "yaml_to_dict",
    "parquet_append",
    "parquet_to_dataframe",
]

from pathlib import Path as _Path

import json as _json
import os as _os
import pyarrow as _pa
import pyarrow.parquet as _pq
import pandas as _pd
import yaml as _yaml


def dataframe_to_parquet(df, metadata, filepath=None, filename=None):
    """
    Save a dataframe to parquet format with custom metadata.

    Parameters
    ----------

    df: pandas.DataFrame
        The dataframe to be saved. In this case containing info required for
        FEP calculation.

    metadata: dict
        Dictionary containing metadata to be saved with the dataframe.

    filepath: str or pathlib.PosixPath
        The of the parent directory in to which the parquet file will be saved.
        If None, save to current working directory.

    filename: str
        The name of the parquet file to be saved. If None, a default name will be used.
    """

    if filepath is None:
        filepath = _Path.cwd()
    elif isinstance(filepath, str):
        filepath = _Path(filepath)
    custom_meta_key = "somd2"
    table = _pa.Table.from_pandas(df)
    custom_meta_json = _json.dumps(metadata)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode(): custom_meta_json.encode(),
        **existing_meta,
    }
    table = table.replace_schema_metadata(combined_meta)
    if filename is None:
        if "lambda" in metadata and "temperature" in metadata:
            filename = f"Lam_{metadata['lambda'].replace('.','')[:5]}_T_{metadata['temperature']}.parquet"
        else:
            filename = "output.parquet"
    if not filename.endswith(".parquet"):
        filename += ".parquet"
    _pq.write_table(table, filepath / filename)
    return filepath / filename


def dict_to_yaml(data_dict, filename="config.yaml", path=None):
    """
    Write a dictionary to a YAML file.

    Parameters
    ----------

    data_dict: dict
        The dictionary to be written to a YAML file.

    filename: str
        The name of the YAML file to be written (default 'config.yaml').

    path: str or pathlib.PosixPath
        The path to the YAML file to be written.
    """
    import yaml as _yaml

    if path is None:
        path = _Path(filename)
    else:
        path = _Path(path) / filename

    try:
        # Ensure the parent directory for the file exists
        path.parent.mkdir(parents=True, exist_ok=True)
        # Open the file in write mode and write the dictionary as YAML
        with path.open("w") as yaml_file:
            _yaml.dump(
                data_dict,
                yaml_file,
            )
    except Exception as e:
        raise IOError(f"Error writing the dictionary to {path}: {e}")


def yaml_to_dict(path):
    """
    Read a YAML file and return the contents as a dictionary.

    Parameters
    ----------

    path: str
        The path to the YAML file to be read.
    """

    if not isinstance(path, str):
        raise TypeError("'path' must be of type 'str'")

    if not _os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(path, "r") as f:
            d = _yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Could not load YAML file: {e}")

    return d


def parquet_append(filepath: _Path or str, df: _pd.DataFrame) -> None:
    """
    Append to dataframe to existing .parquet file. Reads original .parquet file in,
    appends new dataframe, writes new .parquet file out.

    Parameters
    ----------
    filepath: str
        Filepath for parquet file.

    df: pandas.DataFrame
        Pandas dataframe to append. Must be same schema as original.
    """
    try:
        # Use memory map for speed.
        table_original_file = _pq.read_table(
            source=str(filepath), pre_buffer=False, use_threads=True, memory_map=True
        )
        table_to_append = _pa.Table.from_pandas(df)
        # Attempt to cast new schema to existing, e.g. datetime64[ns] to
        # datetime64[us] (may throw otherwise).
        table_to_append = table_to_append.cast(table_original_file.schema)

        # Temporary file to write to.
        temp_file = str(filepath) + "_temp"

        # Writing to a temporary file
        with _pq.ParquetWriter(temp_file, table_original_file.schema) as temp_writer:
            temp_writer.write_table(table_original_file)
            temp_writer.write_table(table_to_append)

        import shutil as _shutil

        # Atomic operation to ensure data integrity
        _shutil.move(temp_file, filepath)

    except Exception as e:
        raise (f"Error occurred append to Parquet file: {e}")


@staticmethod
def parquet_to_dataframe(filepath, meta_key="somd2"):
    """
    Reads a parquet file containing an energy trajectory, extracts the trajectory
    as a dataframe and the metadata as a dictionary.

    Parameters
    ----------

    filepath : str
        Path to the parquet file containing the energy trajectory

    meta_key : str
        Key of the metadata to be used for analysis

    Returns
    -------

    restored_df : pandas dataframe
        Dataframe containing the energy trajectory

    restored_meta : dict
        Dictionary containing the metadata for the simulation"""

    try:
        restored_table = _pq.read_table(filepath)
    except:
        raise ValueError("Unable to read parquet file")
    restored_df = restored_table.to_pandas()
    try:
        restored_meta_json = restored_table.schema.metadata[meta_key.encode()]
    except KeyError:
        raise KeyError("No metadata with key {} found".format(meta_key))
    restored_meta = _json.loads(restored_meta_json)
    return restored_df, restored_meta
