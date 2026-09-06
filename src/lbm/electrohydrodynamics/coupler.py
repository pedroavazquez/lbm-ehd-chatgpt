from collections.abc import Callable

import numpy as np

from lbm.electrohydrodynamics.force import (
    electric_force_density,
)
from lbm.interfaces import (
    ElectrostaticSolver,
    FieldInterpolator,
    HydrodynamicSolver,
)


class EHDCoupler:
    """
    Coordinate electrostatic and hydrodynamic solvers through
    an interpolation layer.

    The coupler depends only on abstract solver behavior,
    rather than on concrete Cartesian implementations.

    Current coupling
    ----------------
    Coulomb body force:

        F_e = rho_e E

    Charge transport and phase-field coupling will be added
    later.
    """

    def __init__(
        self,
        hydrodynamics: HydrodynamicSolver,
        electrostatics: ElectrostaticSolver,
        interpolator: FieldInterpolator,
    ):
        self.hydrodynamics = hydrodynamics

        self.electrostatics = electrostatics

        self.interpolator = interpolator

        # ====================================================
        # Check compatibility between the hydrodynamic grid
        # and the interpolation target grid.
        # ====================================================

        if self.hydrodynamics.nx != self.interpolator.nx_lbm:
            raise ValueError("LBM nx is inconsistent with the SEM-to-LBM interpolator")

        if self.hydrodynamics.ny != self.interpolator.ny_lbm:
            raise ValueError("LBM ny is inconsistent with the SEM-to-LBM interpolator")

        # ====================================================
        # Coupling state
        # ====================================================

        self.electric_field_lbm = None
        self.electric_force_lbm = None

    def solve_electric_problem(
        self,
        source_function: Callable,
    ) -> np.ndarray:
        """
        Solve the electrostatic problem and reconstruct the
        electric field on the electrostatic solver grid.

        Parameters
        ----------
        source_function : callable
            Source term of the electrostatic equation.

        Returns
        -------
        electric_field_sem : ndarray
            Electric field on the electrostatic solver grid.
        """

        self.electrostatics.solve(source_function)

        electric_field_sem = self.electrostatics.reconstruct_electric_field()

        # New electrostatic solution invalidates previously
        # interpolated quantities.
        self.electric_field_lbm = None
        self.electric_force_lbm = None

        return electric_field_sem

    def interpolate_electric_field(
        self,
    ) -> np.ndarray:
        """
        Interpolate the electric field to the hydrodynamic grid.

        Returns
        -------
        electric_field_lbm : ndarray
            Electric field using the LBM storage convention

                (2, nx, ny).
        """

        field_sem = self.electrostatics.electric_field_grid()

        self.electric_field_lbm = self.interpolator.interpolate_vector_to_lbm_storage(
            field_sem
        )

        # A new electric field invalidates the previously
        # calculated electric body force.
        self.electric_force_lbm = None

        return self.electric_field_lbm

    def update_electric_force(
        self,
        charge_density: np.ndarray,
        preserve_physical_velocity: bool = True,
    ) -> np.ndarray:
        """
        Compute and install the Coulomb body-force density

            F_e = rho_e E.

        Parameters
        ----------
        charge_density : ndarray, shape (nx, ny)
            Free electric charge density on the hydrodynamic
            grid.

        preserve_physical_velocity : bool, default True
            Preserve the physical velocity when the force is
            changed.

            This avoids introducing an artificial half-force
            velocity shift associated with Guo forcing.

        Returns
        -------
        electric_force_lbm : ndarray, shape (2, nx, ny)
            Electric body-force density.
        """

        if self.electric_field_lbm is None:
            raise RuntimeError(
                "Interpolate the electric field before computing the electric force"
            )

        # ====================================================
        # Coulomb body force
        # ====================================================

        self.electric_force_lbm = electric_force_density(
            charge_density,
            self.electric_field_lbm,
        )

        # ====================================================
        # Guo forcing consistency
        #
        # Physical momentum is reconstructed as
        #
        #     rho u =
        #         sum_i f_i c_i
        #         + 1/2 F.
        #
        # Changing F without modifying the populations would
        # produce an artificial instantaneous change
        #
        #     Delta u = Delta F / (2 rho).
        #
        # Therefore recover the physical state using the old
        # force, install the new force, and rebuild the
        # population state consistently.
        # ====================================================

        if preserve_physical_velocity:
            rho, u = self.hydrodynamics.macroscopic()

            self.hydrodynamics.set_force(self.electric_force_lbm)

            self.hydrodynamics.set_state(
                rho,
                u,
            )

        else:
            self.hydrodynamics.set_force(self.electric_force_lbm)

        return self.electric_force_lbm

    def advance_hydrodynamics(
        self,
        nsteps: int = 1,
        stream_operator=None,
    ) -> None:
        """
        Advance the hydrodynamic solver.

        Parameters
        ----------
        nsteps : int
            Number of hydrodynamic time steps.

        stream_operator : callable, optional
            Optional streaming/boundary operator.

            If omitted, the hydrodynamic solver's default
            streaming implementation is used.
        """

        if nsteps < 1:
            raise ValueError("nsteps must be positive")

        for _ in range(nsteps):
            if stream_operator is None:
                self.hydrodynamics.step()

            else:
                self.hydrodynamics.step(stream_operator=(stream_operator))
