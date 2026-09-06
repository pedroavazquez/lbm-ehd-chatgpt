from dataclasses import dataclass


@dataclass(frozen=True)
class StructuredSEMMesh2D:
    """
    Structured Cartesian 2-D spectral-element mesh.
    """

    num_elements_x: int
    num_elements_y: int

    order_x: int
    order_y: int

    x_left: float
    x_right: float

    y_bottom: float
    y_top: float

    def __post_init__(self):

        if self.num_elements_x < 1:
            raise ValueError("num_elements_x must be at least 1")

        if self.num_elements_y < 1:
            raise ValueError("num_elements_y must be at least 1")

        if self.order_x < 1:
            raise ValueError("order_x must be at least 1")

        if self.order_y < 1:
            raise ValueError("order_y must be at least 1")

        if self.x_right <= self.x_left:
            raise ValueError("x_right must be greater than x_left")

        if self.y_top <= self.y_bottom:
            raise ValueError("y_top must be greater than y_bottom")

    @property
    def nx_global(self) -> int:

        return self.num_elements_x * self.order_x + 1

    @property
    def ny_global(self) -> int:

        return self.num_elements_y * self.order_y + 1

    @property
    def global_shape(self) -> tuple[int, int]:
        """
        Shape in the same convention used by the current
        SEM assemblers:

            (nx_global, ny_global)
        """

        return (
            self.nx_global,
            self.ny_global,
        )

    @property
    def array_shape(self) -> tuple[int, int]:
        """
        Shape of a field stored as a 2-D NumPy array:

            (ny_global, nx_global)
        """

        return (
            self.ny_global,
            self.nx_global,
        )

    @property
    def ndof(self) -> int:

        return self.nx_global * self.ny_global

    @property
    def element_width(self) -> float:

        return (self.x_right - self.x_left) / self.num_elements_x

    @property
    def element_height(self) -> float:

        return (self.y_top - self.y_bottom) / self.num_elements_y


@dataclass(frozen=True)
class AxisymmetricSEMMesh2D:
    """
    Structured non-swirling axisymmetric SEM mesh in (z, r).
    """

    num_elements_z: int
    num_elements_r: int

    order_z: int
    order_r: int

    z_left: float
    z_right: float

    r_bottom: float
    r_top: float

    def __post_init__(self):

        if self.num_elements_z < 1:
            raise ValueError("num_elements_z must be at least 1")

        if self.num_elements_r < 1:
            raise ValueError("num_elements_r must be at least 1")

        if self.order_z < 1:
            raise ValueError("order_z must be at least 1")

        if self.order_r < 1:
            raise ValueError("order_r must be at least 1")

        if self.z_right <= self.z_left:
            raise ValueError("z_right must be greater than z_left")

        if self.r_bottom < 0.0:
            raise ValueError("r_bottom must be non-negative")

        if self.r_top <= self.r_bottom:
            raise ValueError("r_top must be greater than r_bottom")

    @property
    def nz_global(self) -> int:

        return self.num_elements_z * self.order_z + 1

    @property
    def nr_global(self) -> int:

        return self.num_elements_r * self.order_r + 1

    @property
    def global_shape(self) -> tuple[int, int]:

        return (
            self.nz_global,
            self.nr_global,
        )

    @property
    def array_shape(self) -> tuple[int, int]:

        return (
            self.nr_global,
            self.nz_global,
        )

    @property
    def ndof(self) -> int:

        return self.nz_global * self.nr_global
