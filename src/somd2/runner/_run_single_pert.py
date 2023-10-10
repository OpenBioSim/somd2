__all__ = ["MergedSimulation"]
from ..config import Config as _Config


# rename this
class RunSingleWindow:
    """
    Class for controlling the running and bookkeeping of a single lambda value simulation
    Currently just a wrapper around sire dynamics

    Simulation options are held within a Config object
    """

    def __init__(
        self,
        system,
        lambda_val,
        lambda_array,
        config,
        increment=0.001,
        device=None,
    ):
        """
        Constructor

        Parameters
        ----------
        system : Sire System
            Sire system containing at least one perturbable molecule

        lambda_val : float
            Lambda value for the simulation

        lambda_array : list
            List of lambda values to be used for perturbation, if none won't return
            reduced perturbed energies

        increment : float
            Increment of lambda value - used for calculating the gradient

        config : somd2 Config object
            Config object containing simulation options

        device : int
            GPU device number to use  - does nothing if running on CPU (default None)

        """

        try:
            system.molecules("property is_perturbable")
        except KeyError:
            raise KeyError("No perturbable molecules in the system")

        self._system = system
        try:
            self._system.molecules("property is_perturbable")
        except KeyError:
            raise KeyError("No perturbable molecules in the system")

        if not isinstance(config, _Config):
            raise TypeError("config must be a Config object")
        self._config = config
        self._lambda_val = lambda_val
        self._lambda_array = lambda_array
        self._increment = increment
        self._device = device

    # Would prob. be better to just set up the dynamics object here,
    # then run a separate dynamics.minimise
    def _setup_dynamics(self, lam_val_min=None):
        """
        Minimise if needed and then setup dynamics object

        Parameters
        ----------
        lam_val_min : float
            Lambda value at which to run minimisation,
            if None run at pre-set lambda_val
        """

        if self._config.minimise:
            if lam_val_min is None:
                try:
                    m = self._system.minimisation(
                        cutoff_type=self._config.cutoff_type,
                        schedule=self._config.lambda_schedule,
                        lambda_value=self._lambda_val,
                        map=self._config.extra_args,
                    )
                    m.run()
                    self._system = m.commit()
                except:
                    raise
            else:
                try:
                    m = self._system.minimisation(
                        cutoff_type=self._config.cutoff_type,
                        schedule=self._config.lambda_schedule,
                        lambda_value=lam_val_min,
                        map=self._config.extra_args,
                    )
                    m.run()
                    self._system = m.commit()
                except:
                    raise
        self._dyn = self._system.dynamics(
            temperature=self._config.temperature,
            pressure=self._config.pressure,
            timestep=self._config.timestep,
            lambda_value=self._lambda_val,
            cutoff_type=self._config.cutoff_type,
            schedule=self._config.lambda_schedule,
            platform=self._config.platform,
            device=self._device,
            map=self._config.extra_args,
        )

    # combine these - just equil time
    # reset timer to zero when bookeeping starts
    def _equilibration(self):
        """
        Placeholder for per-window equilibration.
        Run the simulation without bookkeeping
        """
        self._dyn.run(
            self._config.equilibration_time,
            frame_frequency=self._config.frame_frequency,
            save_velocities=self._config.save_velocities,
        )
        self._dyn.commit()

    def _run(self):
        """
        Run the simulation with bookkeeping
        Returns
        -------
        df : pandas dataframe
            Dataframe containing the sire energy
            trajectory
        """
        from sire import u as _u

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

        if self._config.equilibrate:
            self._equilibration()

        # Work out the lambda values for finite-difference gradient analysis.
        self._lambda_grad = generate_lam_vals(self._lambda_val, self._increment)

        if self._lambda_array is None:
            lam_arr = self._lambda_grad
        else:
            lam_arr = self._lambda_array + self._lambda_grad
        try:
            self._dyn.run(
                self._config.runtime,
                energy_frequency=self._config.energy_frequency,
                frame_frequency=self._config.frame_frequency,
                lambda_windows=lam_arr,
                save_velocities=self._config.save_velocities,
                auto_fix_minimise=False,
            )
        except Exception:
            raise
        self._system = self._dyn.commit()
        from pathlib import Path as _Path

        if self._config.output_directory is not None:
            _Path(self._config.output_directory).mkdir(parents=True, exist_ok=True)
            outdir = _Path(self._config.output_directory)
        else:
            outdir = _Path.cwd()
        traj_filename = outdir / f"traj_{self._lambda_val}"
        # from sire import save as _save

        # _save(self._system.trajectory(), traj_filename, format=["DCD"])
        df = self._system.energy_trajectory(to_alchemlyb=True)
        return df

    def get_timing(self):
        return self._dyn.time_speed()

    def _cleanup(self):
        del self._dyn
