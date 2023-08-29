"""Input : dataframe of energy values """


class analyse_single_lambda:
    """Reads a parquet file containing an energy trajectory, as
    well as metadata describing the details of the system, then
    performs analysis on the trajectory.
    Currently only supports SOMD-style simulations."""

    def __init__(self, data_parquet, custom_meta_key="SOMD2.iot"):
        """
        Constructor

        Parameters
        ----------
        data_parquet : str
            Path to the parquet file containing the energy trajectory
        custom_meta_key : str
            Key of the metadata to be used for analysis
        """
        from pathlib import Path as _Path

        file_path = _Path(data_parquet)
        if file_path.suffix not in [".parquet", ".pqt", ".pq"]:
            raise ValueError("File must be a parquet file")
        self._parquet_file = file_path

        self._meta_key = custom_meta_key

        self._dataframe, self._metadata = self.parquet_to_dataframe(
            self._parquet_file, self._meta_key
        )
        print(self._metadata)

    @staticmethod
    def parquet_to_dataframe(filepath, meta_key="SOMD2.iot"):
        """
        Reads a parquet file containing an energy trajectory,
        extracts the trajectory as a dataframe and the metadata as a
        dictionary

        parameters
        ----------
        filepath : str
            Path to the parquet file containing the energy trajectory

        meta_key : str
            Key of the metadata to be used for analysis

        returns
        -------
        restored_df : pandas dataframe
            Dataframe containing the energy trajectory

        restored_meta : dict
            Dictionary containing the metadata for the simulation"""
        import pyarrow.parquet as _pq
        import json as _json

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
        if not all(key in restored_meta for key in ["lambda", "temperature"]):
            raise KeyError("No lambda and/or temperature found in metadata")
        return restored_df, restored_meta

    def analyse(self):
        """
        Function to call calculate_gradient_and_pert_energies
        for internal data structures.
        """
        self._calculate_beta()
        try:
            self._lambda_array = self._metadata["lambda_array"]
        except KeyError:
            self._lambda_array = None
        self._analysed_df = self.calculate_gradient_and_pert_energies(
            self._dataframe,
            self._metadata["lambda"],
            self._beta,
            self._lambda_array,
        )
        return self._analysed_df

    def _calculate_beta(self):
        from scipy import constants as _const
        from sire import u as _u
        from sire.units import kelvin as _kelvin

        print(self._metadata["temperature"])
        self._beta = 1.0 / (
            (_const.gas_constant / 1000)
            * _u(float(self._metadata["temperature"]) * _kelvin).to("K")
        )

    @staticmethod
    def calculate_gradient_and_pert_energies(df, base_lambda, beta, lambda_array=None):
        """
        Calculates the gradient, along with forward and backward metropolis
          using energy values in the dataframe

        Parameters
        ----------
        df : pandas dataframe
            dataframe of energy values output from sire dynamics

        base_lambda : float
            lambda value of the current simulation

        beta : float
            Thermodynamic beta value

        lambda_array:
            Array of lambda values used accross all simulations,
            required for MBAR analysis

        Returns
        --------
        df : pandas dataframe
            dataframe of energy values with gradient, forward and backward metropolis
            values appended, as well as columns for lambda_array re-scaled to perturbation
        """
        import pandas as _pd
        import numpy as _np

        df = df.copy()
        base_lambda = float(base_lambda)
        columns_lambdas = df.columns[
            _pd.to_numeric(df.columns, errors="coerce").to_series().notnull()
        ]
        if len(columns_lambdas) > 3 and lambda_array is None:
            raise ValueError(
                "More than 3 lambda values in the dataframe..but no lambda array provided"
            )
        try:
            lam_below = max(
                [
                    float(lambda_val)
                    for lambda_val in columns_lambdas
                    if float(lambda_val) < float(base_lambda)
                ]
            )
        except ValueError:
            lam_below = None
        try:
            lam_above = min(
                [
                    float(lambda_val)
                    for lambda_val in columns_lambdas
                    if float(lambda_val) > float(base_lambda)
                ]
            )
        except ValueError:
            lam_above = None
        if lam_below is None:
            double_incr = (lam_above - base_lambda) * 2
            grad = (df[str(lam_above)] - df[str(base_lambda)]) * 2 / double_incr
            back_m = _np.exp(beta * (df[str(lam_above)] - df[str(base_lambda)]))
            forward_m = _np.exp(
                -1 * beta * (df[str(lam_above)] - df[str(base_lambda)])
            )  # just 1/back_m?
        elif lam_above is None:
            double_incr = (base_lambda - lam_below) * 2
            grad = (df[str(base_lambda)] - df[str(lam_below)]) * 2 / double_incr
            back_m = _np.exp(-1 * beta * (df[str(lam_below)] - df[str(base_lambda)]))
            forward_m = _np.exp(beta * (df[str(lam_below)] - df[str(base_lambda)]))

        else:
            double_incr = lam_above - lam_below
            grad = (df[str(lam_above)] - df[str(lam_below)]) / double_incr
            back_m = _np.exp(beta * (df[str(base_lambda)] - df[str(lam_below)]))
            forward_m = _np.exp(beta * (df[str(base_lambda)] - df[str(lam_above)]))

        grad.name = "gradient"
        back_m.name = "backward_mc"
        forward_m.name = "forward_mc"

        if lambda_array is not None:
            df[[str(i) for i in lambda_array]] = df[
                [str(i) for i in lambda_array]
            ].apply(lambda x: x * -1 * beta)

        df = _pd.concat(
            [df, _pd.DataFrame(grad), _pd.DataFrame(back_m), _pd.DataFrame(forward_m)],
            axis=1,
        )
        return df

    def get_metadata(self):
        """
        Returns the metadata for the current simulation
        """
        return self._metadata


class Analyse_all:
    """
    Reads a directory of parquet files containing energy trajectories,
    and analyses them all, then proceeds to perform FEP calculation with
    alchemlyb
    """

    def __init__(self, parquet_folder, method="TI", custom_meta_key="SOMD2.iot"):
        """
        Constructor

        Parameters
        ----------
        parquet_folder : str
            Path to the directory containing the parquet files
        custom_meta_key : str
            Key of the metadata to be used for analysis
        """
        from pathlib import Path as _Path
        import glob as _glob

        folder_path = _Path(parquet_folder)
        if not folder_path.is_dir():
            raise ValueError("Path must be a directory")
        self._parquet_folder = folder_path
        self._parquet_files = _glob.glob(str(folder_path / "*.parquet"))
        self._custom_meta_key = custom_meta_key
        if method not in ["TI", "MBAR"]:
            raise ValueError("Method must be either TI or MBAR")
        else:
            self._method = method

    @staticmethod
    def extract_data_TI(dataframe, metadata):
        """
        Extract gradients from a processed dataframe,
        formats in alchemlyb-compatible dataframe format

        Parameters
        ----------
        dataframe : pandas dataframe
            Dataframe containing the gradient values
            requires a column named 'gradient' and
            either a column named 'time' or for
            time to be the index column

        metadata : dict
            Dictionary containing metadata for the simulation
            minimum requirement is a key 'lambda' containing
            the lambda value of the simulation in question

        Returns
        -------
        df : pandas dataframe
            Dataframe containing the gradient values, formatted for TI
        """
        grads = list(dataframe["gradient"])
        try:
            time = list(dataframe["time"])
        except KeyError:
            time = list(dataframe.index)
        lam = len(time) * [float(metadata["lambda"])]
        import pandas as _pd

        # Create a multi-index from the two lists
        multi_index = _pd.MultiIndex.from_tuples(
            zip(time, lam), names=["time", "fep-lambdas"]
        )

        # Create a DataFrame with the multi-index
        df = _pd.DataFrame({"fep": grads}, index=multi_index)
        return df

    @staticmethod
    def extract_data_MBAR(dataframe, metadata):
        """
        Extract gradients from a processed dataframe,
        formats in alchemlyb-compatible dataframe format

        Parameters
        ----------
        dataframe : pandas dataframe
            Dataframe containing the reduced potential values

        metadata : dict
            Dictionary containing metadata for the simulation
            minimum requirement is a key 'lambda_windows' containing
            a list of all lambda values used across all simulations

        Returns
        -------
        df : pandas dataframe
            Dataframe containing the reduced potential values, formatted for MBAR"""
        try:
            lambda_array = metadata["lambda_array"]
        except KeyError:
            raise KeyError(
                "No lambda_array found in metadata,unable to perform MBAR calculation"
            )
        import pandas as _pd

        temp = dataframe[[str(i) for i in lambda_array]].copy()
        lams = len(temp.index) * [metadata["lambda"]]
        multiindex = _pd.MultiIndex.from_tuples(
            zip(temp.index, lams), names=["time", "lambdas"]
        )
        temp.index = multiindex
        return temp

    def analyse_all(self):
        """
        Function to call analyse on all parquet files in the directory
        returns a list of dataframes that is sorted and ready to be used
        by alchemlyb
        """
        extracted_dict = (
            {}
        )  # dict to store extracted data, needs to be sorted, python 3.7+ required
        lam_array = None  # Used to check that all lambda arrays match
        for parquet_file in self._parquet_files:
            temp = analyse_single_lambda(parquet_file, self._custom_meta_key)
            analysed = temp.analyse()
            meta = temp.get_metadata()
            lam_curr = float(meta["lambda"])
            if self._method == "MBAR":
                if lam_array is None:
                    lam_array = meta["lambda_array"]
                else:
                    if sorted(meta["lambda_array"]) != sorted(lam_array):
                        raise ValueError(
                            "Lambda arrays do not match across all simulations"
                        )
                extracted_dict[lam_curr] = self.extract_data_MBAR(
                    analysed, meta
                ).dropna()
            elif self._method == "TI":
                extracted_dict[lam_curr] = self.extract_data_TI(analysed, meta).dropna()

        s = dict(sorted(extracted_dict.items()))
        data = list(s.values())

        return data
