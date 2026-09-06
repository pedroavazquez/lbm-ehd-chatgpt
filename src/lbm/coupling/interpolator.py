import numpy as np

from lbm.coupling.sem_to_lbm import (
    element_interpolation_matrices,
    interpolate_element_field,
)
from lbm.electrostatics.sem.mesh import (
    StructuredSEMMesh2D,
)


class SEMToLBMInterpolator:
    """
    Precomputed interpolation from a structured SEM mesh to an
    aligned cell-centred LBM grid.

    Each SEM element covers an integer number of LBM cells.
    """

    def __init__(
        self,
        mesh: StructuredSEMMesh2D,
        lbm_cells_per_element_x: int,
        lbm_cells_per_element_y: int,
    ):

        if lbm_cells_per_element_x < 1:
            raise ValueError("lbm_cells_per_element_x must be positive")

        if lbm_cells_per_element_y < 1:
            raise ValueError("lbm_cells_per_element_y must be positive")

        self.mesh = mesh

        self.lbm_cells_per_element_x = lbm_cells_per_element_x

        self.lbm_cells_per_element_y = lbm_cells_per_element_y

        (
            self.xi_lbm,
            self.eta_lbm,
            self.Px,
            self.Py,
        ) = element_interpolation_matrices(
            order_x=mesh.order_x,
            order_y=mesh.order_y,
            num_lbm_nodes_x=(lbm_cells_per_element_x),
            num_lbm_nodes_y=(lbm_cells_per_element_y),
            include_endpoints=False,
        )

    @property
    def nx_lbm(self) -> int:

        return self.mesh.num_elements_x * self.lbm_cells_per_element_x

    @property
    def ny_lbm(self) -> int:

        return self.mesh.num_elements_y * self.lbm_cells_per_element_y

    @property
    def lbm_shape_yx(
        self,
    ) -> tuple[int, int]:

        return (
            self.ny_lbm,
            self.nx_lbm,
        )

    def interpolate_scalar(
        self,
        field_sem: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        field_sem : ndarray
            Shape (ny_sem_global, nx_sem_global).

        Returns
        -------
        field_lbm : ndarray
            Shape (ny_lbm, nx_lbm).
        """

        if field_sem.shape != self.mesh.array_shape:
            raise ValueError(f"field_sem must have shape {self.mesh.array_shape}")

        result = np.empty(
            self.lbm_shape_yx,
            dtype=np.result_type(
                field_sem,
                float,
            ),
        )

        nx_local = self.mesh.order_x + 1

        ny_local = self.mesh.order_y + 1

        for ey in range(self.mesh.num_elements_y):
            for ex in range(self.mesh.num_elements_x):
                i0_sem = ex * self.mesh.order_x

                j0_sem = ey * self.mesh.order_y

                local_sem = field_sem[
                    j0_sem : j0_sem + ny_local,
                    i0_sem : i0_sem + nx_local,
                ]

                local_lbm = interpolate_element_field(
                    field_sem=local_sem,
                    Px=self.Px,
                    Py=self.Py,
                )

                i0_lbm = ex * self.lbm_cells_per_element_x

                j0_lbm = ey * self.lbm_cells_per_element_y

                result[
                    j0_lbm : (j0_lbm + self.lbm_cells_per_element_y),
                    i0_lbm : (i0_lbm + self.lbm_cells_per_element_x),
                ] = local_lbm

        return result

    def interpolate_vector(
        self,
        field_sem: np.ndarray,
    ) -> np.ndarray:
        """
        Input:
            (ncomponents, ny_sem, nx_sem)

        Output:
            (ncomponents, ny_lbm, nx_lbm)
        """

        if field_sem.ndim != 3:
            raise ValueError("field_sem must have shape (ncomponents, ny_sem, nx_sem)")

        components = [
            self.interpolate_scalar(field_sem[i]) for i in range(field_sem.shape[0])
        ]

        return np.stack(
            components,
            axis=0,
        )

    def interpolate_vector_to_lbm_storage(
        self,
        field_sem: np.ndarray,
    ) -> np.ndarray:
        """
        Convert to the hydrodynamic LBM storage convention:

            (ncomponents, nx, ny).
        """

        field = self.interpolate_vector(field_sem)

        return np.transpose(
            field,
            (
                0,
                2,
                1,
            ),
        )
