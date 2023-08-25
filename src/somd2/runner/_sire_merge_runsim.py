__all__ = ["MergedSimulation"]


class MergedSimulation:
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
        lambda_array=None,
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

        lambda_array : list
            List of lambda values to be used for perturbation, if none won't return
            reduced perturbed energies

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

        self._system = system
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
        self._lambda_array = lambda_array

    def _setup_dynamics(self, timestep="2fs", lam_val_min=None):
        """
        Minimise if needed and then setup dynamics object

        Parameters
        ----------
        timestep : str
            Timestep of the simulation

        lam_val_min : float
            Lambda value aat which to run minimisation,
            if None run at pre-set lambda_val
        """

        if self._minimise:
            if lam_val_min is None:
                self._system = (
                    self._system.minimisation(lambda_value=self._lambda_val)
                    .run()
                    .commit()
                )
            else:
                self._system = (
                    self._system.minimisation(lambda_value=lam_val_min).run().commit()
                )

        self._dyn = self._system.dynamics(
            timestep=timestep,
            lambda_value=self._lambda_val,
            map=self._map,
        )

    def _run_no_bookkeeping(
        self,
        frame_frequency="1ps",
        save_velocities=False,
    ):
        """
        Placeholder for per-window equilibration.
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
            self._no_bookeeping_time,
            frame_frequency=frame_frequency,
            save_velocities=save_velocities,
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

        Returns
        -------
        df : pandas dataframe
            Dataframe containing the sire energy
            trajectory
        """
        if self._no_bookeeping_time is not None:
            self._run_no_bookkeeping()

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

        if self._lambda_array is None:
            lam_arr = generate_lam_vals(self._lambda_val, self._increment)
        else:
            lam_arr = self._lambda_array + generate_lam_vals(
                self._lambda_val, self._increment
            )
        try:
            self._dyn.run(
                runtime,
                energy_frequency=energy_frequency,
                frame_frequency=frame_frequency,
                lambda_windows=lam_arr,
                save_velocities=save_velocities,
                auto_fix_minimise=False,
            )
        except Exception:
            raise
        self._system = self._dyn.commit()
        df = self._system.property("energy_trajectory").to_pandas()
        return df
