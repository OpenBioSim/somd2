__all__ = ["_merged_simulation"]


class _merged_simulation:
    """
    Class for controlling the running and bookkeeping of a single lambda value simulation
    Currently just a wrapper around sire dynamics

    Currently the input map handles all simulation options
    """

    def __init__(
        self,
        system,
        map,
        lambda_val,
        increment=0.001,
        minimise=False,
        no_bookkeeping_only=False,
        no_bookkeeping_time=None,
    ):
        """
        Constructor

        Parameters
        ----------
        system : Sire System
            Sire system containing at least one perturbable molecule

        map : Sire Map
            Sire map containing all the simulation options

        lambda_val : float
            Lambda value for the simulation

        increment : float
            Increment of lambda value - used for calculating the gradient

        minimise : bool
            Whether to minimise the system before running the simulation

        no_bookkeeping_only : bool
            Only run the simulation without bookkeeping

        no_bookkeeping_time : str
            Time for which to run no bookkeeping section
        """
        if map is None:
            raise ValueError("No map given")

        try:
            system.molecules("property is_perturbable")
        except KeyError:
            raise KeyError("No perturbable molecules in the system")

        # Cloning might not be needed as its already done by sire dynamics
        self._system = system.clone()
        try:
            self._system.molecules("property is_perturbable")
        except KeyError:
            raise KeyError("No perturbable molecules in the system")

        self._map = map
        self._lambda_val = lambda_val
        self._increment = increment
        self._minimise = minimise
        self._no_bookkeeping_only = no_bookkeeping_only
        self._no_bookeeping_time = no_bookkeeping_time
        # self._setup_dynamics()
        print("Here", flush=True)

    @staticmethod
    def calculate_gradient(df, base_lambda, beta):
        """
        Calculates the gradient, along with forward and backward metropolis
          using energy values in the dataframe

        Parameters
        ----------
        df : pandas dataframe
            dataframe of energy values output from sire dynamics

        base_lambda : float
            lambda value at which gradient is to be calculated

        beta : float
            Thermodynamic beta value

        Returns
        --------
        df_final : pandas dataframe
            dataframe of energy values with gradient, forward and backward metropolis
            values appended
        """

        import pandas as _pd
        import numpy as _np

        df = df.copy()
        columns_lambdas = df.columns[
            _pd.to_numeric(df.columns, errors="coerce").to_series().notnull()
        ]
        if len(columns_lambdas) > 3:
            raise ValueError("More than 3 lambda values in the dataframe")
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
        df_final = _pd.concat(
            [df, _pd.DataFrame(grad), _pd.DataFrame(back_m), _pd.DataFrame(forward_m)],
            axis=1,
        )
        return df_final

    def _setup_dynamics(self, timestep="2fs"):
        """
        Minimise if needed and then setup dynamics object
        """

        if self._minimise:
            self._system.minimisation(lambda_value=self._lambda_val).run().commit()

        self._dyn = self._system.dynamics(
            timestep=timestep,
            lambda_value=self._lambda_val,
            map=self._map,
        )

    def _run_no_bookkeeping(
        self,
        runtime="10ps",
        energy_frequency="0.05ps",
        frame_frequency="1ps",
        save_velocities=False,
    ):
        """
        Run the simulation without bookkeeping

        Parameters
        ----------
        runtime : str
            Runtime of the simulation

        runtime : str
            Runtime of the simulation

        timestep : str
            Timestep of the simulation
        """
        self._dyn.run(
            runtime,
            frame_frequency=frame_frequency,
            save_velocities=save_velocities,
            energy_frequency=energy_frequency,
        )
        self._dyn.commit()

    def _run_with_bookkeeping(
        self,
        runtime="10ps",
        energy_frequency="0.05ps",
        frame_frequency="1ps",
        save_velocities=False,
    ):
        """
        Run the simulation with bookkeeping

        Parameters
        ----------
        runtime : str
            Runtime of the simulation

        runtime : str
            Runtime of the simulation

        timestep : str
            Timestep of the simulation
        """
        self._dyn = self._system.dynamics(
            timestep="2fs",
            lambda_value=self._lambda_val,
            temperature="300K",
            map=self._map,
        )
        if self._no_bookeeping_time is not None:
            self._dyn.run(
                self._no_bookeeping_time,
                frame_frequency=frame_frequency,
                save_velocities=save_velocities,
            )
            self._dyn.commit()

        def generate_lam_vals(lambda_base, increment):
            """Generate lambda values for a given lambda_base and increment"""
            if lambda_base + increment > 1.0 and lambda_base - increment < 0.0:
                raise ValueError("Increment too large")
            if lambda_base + increment > 1.0:
                lam_vals = [lambda_base - increment]
            elif lambda_base - increment < 0.0:
                lam_vals = [lambda_base + increment]
            else:
                lam_vals = [lambda_base - increment, lambda_base + increment]
            return lam_vals

        self._dyn.run(
            runtime,
            energy_frequency=energy_frequency,
            frame_frequency=frame_frequency,
            lambda_windows=generate_lam_vals(self._lambda_val, self._increment),
            save_velocities=save_velocities,
            auto_fix_minimise=False,
        )
        self._dyn.commit()
        df = self._system.property("energy_trajectory").to_pandas()

        from scipy import constants as _const
        from sire import u as _u

        beta = 1.0 / (
            (_const.gas_constant / 1000) * _u(self._map["temperature"]).to("K")
        )
        data = self.calculate_gradient(df, self._lambda_val, beta)
        return data