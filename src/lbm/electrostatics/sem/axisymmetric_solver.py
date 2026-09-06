from collections.abc import Callable

import numpy as np

from lbm.electrostatics.sem.axisymmetric_mixed_boundary import (
    solve_axisymmetric_mixed_boundary_2d,
)
from lbm.electrostatics.sem.boundary_conditions import (
    BoundaryCondition,
    to_legacy_boundary_conditions,
)
from lbm.electrostatics.sem.field_reconstruction import (
    reconstruct_electric_field_2d,
)
from lbm.electrostatics.sem.mesh import (
    AxisymmetricSEMMesh2D,
)


class AxisymmetricElectrostaticSolver2D:
    """
    High-level axisymmetric SEM solver in (z, r).
    """

    def __init__(
        self,
        mesh: AxisymmetricSEMMesh2D,
        coefficient_function: Callable,
        boundary_conditions: dict[
            str,
            BoundaryCondition,
        ],
    ):

        self.mesh = mesh

        self.coefficient_function = coefficient_function

        self.boundary_conditions = boundary_conditions.copy()

        self._coordinates = None
        self._potential = None
        self._electric_field = None

    @property
    def coordinates(
        self,
    ) -> np.ndarray:

        if self._coordinates is None:
            raise RuntimeError("The electrostatic problem has not been solved")

        return self._coordinates

    @property
    def potential(
        self,
    ) -> np.ndarray:

        if self._potential is None:
            raise RuntimeError("The electrostatic problem has not been solved")

        return self._potential

    @property
    def electric_field(
        self,
    ) -> np.ndarray:

        if self._electric_field is None:
            raise RuntimeError("The electric field has not been reconstructed")

        return self._electric_field

    def solve(
        self,
        source_function: Callable,
    ) -> np.ndarray:

        bc_legacy = to_legacy_boundary_conditions(self.boundary_conditions)

        (
            coordinates,
            potential,
            _,
        ) = solve_axisymmetric_mixed_boundary_2d(
            num_elements_z=(self.mesh.num_elements_z),
            num_elements_r=(self.mesh.num_elements_r),
            order_z=self.mesh.order_z,
            order_r=self.mesh.order_r,
            z_left=self.mesh.z_left,
            z_right=self.mesh.z_right,
            r_bottom=self.mesh.r_bottom,
            r_top=self.mesh.r_top,
            coefficient_function=(self.coefficient_function),
            source_function=source_function,
            boundary_conditions=bc_legacy,
        )

        self._coordinates = coordinates
        self._potential = potential
        self._electric_field = None

        return self._potential

    def reconstruct_electric_field(
        self,
    ) -> np.ndarray:

        if self._potential is None:
            raise RuntimeError(
                "Solve the potential before reconstructing the electric field"
            )

        # Gradient reconstruction is geometrically identical
        # in coordinates (z, r).
        self._electric_field = reconstruct_electric_field_2d(
            phi=self._potential,
            num_elements_x=(self.mesh.num_elements_z),
            num_elements_y=(self.mesh.num_elements_r),
            order_x=self.mesh.order_z,
            order_y=self.mesh.order_r,
            x_left=self.mesh.z_left,
            x_right=self.mesh.z_right,
            y_bottom=self.mesh.r_bottom,
            y_top=self.mesh.r_top,
        )

        return self._electric_field

    def potential_grid(
        self,
    ) -> np.ndarray:

        return self.potential.reshape(self.mesh.array_shape)

    def electric_field_grid(
        self,
    ) -> np.ndarray:

        return self.electric_field.reshape(
            (
                2,
                self.mesh.nr_global,
                self.mesh.nz_global,
            )
        )
