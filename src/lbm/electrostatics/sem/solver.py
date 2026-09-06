from collections.abc import Callable

import numpy as np

from lbm.electrostatics.sem.boundary_conditions import (
    BoundaryCondition,
    to_legacy_boundary_conditions,
)
from lbm.electrostatics.sem.field_reconstruction import (
    reconstruct_electric_field_2d,
)
from lbm.electrostatics.sem.mesh import (
    StructuredSEMMesh2D,
)
from lbm.electrostatics.sem.mixed_boundary import (
    solve_mixed_boundary_2d,
)


class ElectrostaticSolver2D:
    """
    High-level Cartesian spectral-element electrostatic solver.

    The numerical kernels remain in the existing functional
    modules; this class owns configuration and solution state.
    """

    def __init__(
        self,
        mesh: StructuredSEMMesh2D,
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
        """
        Solve the potential problem and return the global
        flattened SEM potential.
        """

        bc_legacy = to_legacy_boundary_conditions(self.boundary_conditions)

        (
            coordinates,
            potential,
            _,
        ) = solve_mixed_boundary_2d(
            num_elements_x=(self.mesh.num_elements_x),
            num_elements_y=(self.mesh.num_elements_y),
            order_x=self.mesh.order_x,
            order_y=self.mesh.order_y,
            x_left=self.mesh.x_left,
            x_right=self.mesh.x_right,
            y_bottom=self.mesh.y_bottom,
            y_top=self.mesh.y_top,
            coefficient_function=(self.coefficient_function),
            source_function=source_function,
            boundary_conditions=bc_legacy,
        )

        self._coordinates = coordinates
        self._potential = potential

        # A new potential invalidates the old field.
        self._electric_field = None

        return self._potential

    def reconstruct_electric_field(
        self,
    ) -> np.ndarray:
        """
        Compute

            E = -grad(phi)

        at global SEM nodes.
        """

        if self._potential is None:
            raise RuntimeError(
                "Solve the potential before reconstructing the electric field"
            )

        self._electric_field = reconstruct_electric_field_2d(
            phi=self._potential,
            num_elements_x=(self.mesh.num_elements_x),
            num_elements_y=(self.mesh.num_elements_y),
            order_x=self.mesh.order_x,
            order_y=self.mesh.order_y,
            x_left=self.mesh.x_left,
            x_right=self.mesh.x_right,
            y_bottom=self.mesh.y_bottom,
            y_top=self.mesh.y_top,
        )

        return self._electric_field

    def potential_grid(
        self,
    ) -> np.ndarray:
        """
        Return potential as

            (ny_global, nx_global).
        """

        return self.potential.reshape(self.mesh.array_shape)

    def electric_field_grid(
        self,
    ) -> np.ndarray:
        """
        Return electric field as

            (2, ny_global, nx_global).
        """

        field = self.electric_field

        return field.reshape(
            (
                2,
                self.mesh.ny_global,
                self.mesh.nx_global,
            )
        )
