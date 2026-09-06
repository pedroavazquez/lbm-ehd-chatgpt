from dataclasses import dataclass

from lbm.lattices.d2q9 import CS2


@dataclass(frozen=True)
class LatticeScaling:
    """
    Conversion between physical and lattice units.

    The scaling is defined by:

    - a physical reference length,
    - the number of lattice cells across that length,
    - the physical kinematic viscosity,
    - the desired BGK relaxation time tau.

    Keeping tau fixed while refining the grid produces
    diffusive scaling:

        dt ~ dx^2.
    """

    length_phys: float
    n_length: int
    nu_phys: float
    tau: float

    def __post_init__(self):

        if self.length_phys <= 0.0:
            raise ValueError("length_phys must be positive")

        if self.n_length <= 0:
            raise ValueError("n_length must be positive")

        if self.nu_phys <= 0.0:
            raise ValueError("nu_phys must be positive")

        if self.tau <= 0.5:
            raise ValueError("tau must be greater than 0.5")

    @property
    def dx(self) -> float:
        """
        Physical length represented by one lattice spacing.
        """

        return self.length_phys / self.n_length

    @property
    def nu_lattice(self) -> float:
        """
        Kinematic viscosity in lattice units.
        """

        return CS2 * (self.tau - 0.5)

    @property
    def dt(self) -> float:
        """
        Physical time represented by one lattice time step.
        """

        return self.nu_lattice * self.dx**2 / self.nu_phys

    # ========================================================
    # Length
    # ========================================================

    def length_to_lattice(
        self,
        length_phys: float,
    ) -> float:

        return length_phys / self.dx

    def length_to_physical(
        self,
        length_lattice: float,
    ) -> float:

        return length_lattice * self.dx

    # ========================================================
    # Time
    # ========================================================

    def time_to_lattice(
        self,
        time_phys: float,
    ) -> float:

        return time_phys / self.dt

    def time_to_physical(
        self,
        time_lattice: float,
    ) -> float:

        return time_lattice * self.dt

    # ========================================================
    # Velocity
    # ========================================================

    def velocity_to_lattice(
        self,
        velocity_phys: float,
    ) -> float:
        """
        u_LB = u_phys dt / dx
        """

        return velocity_phys * self.dt / self.dx

    def velocity_to_physical(
        self,
        velocity_lattice: float,
    ) -> float:
        """
        u_phys = u_LB dx / dt
        """

        return velocity_lattice * self.dx / self.dt

    # ========================================================
    # Acceleration
    # ========================================================

    def acceleration_to_lattice(
        self,
        acceleration_phys: float,
    ) -> float:
        """
        a_LB = a_phys dt^2 / dx
        """

        return acceleration_phys * self.dt**2 / self.dx

    def acceleration_to_physical(
        self,
        acceleration_lattice: float,
    ) -> float:

        return acceleration_lattice * self.dx / self.dt**2

    # ========================================================
    # Frequency
    # ========================================================

    def frequency_to_lattice(
        self,
        omega_phys: float,
    ) -> float:
        """
        Angular frequency conversion:

            omega_LB = omega_phys dt
        """

        return omega_phys * self.dt

    def frequency_to_physical(
        self,
        omega_lattice: float,
    ) -> float:

        return omega_lattice / self.dt

    # ========================================================
    # Diagnostics
    # ========================================================

    def mach_number(
        self,
        velocity_phys: float,
    ) -> float:
        """
        Mach number associated with a physical velocity.
        """

        velocity_lattice = self.velocity_to_lattice(velocity_phys)

        cs = CS2**0.5

        return abs(velocity_lattice) / cs

    def reynolds_number(
        self,
        velocity_phys: float,
        length_phys: float | None = None,
    ) -> float:
        """
        Reynolds number in physical units.
        """

        if length_phys is None:
            length_phys = self.length_phys

        return velocity_phys * length_phys / self.nu_phys
