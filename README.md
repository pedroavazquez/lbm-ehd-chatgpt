# LBM

Research-oriented Lattice Boltzmann solver developed incrementally for hydrodynamics, electrohydrodynamics, electrokinetics, and multiphase flows.

The current version implements and verifies the hydrodynamic foundation of the code in both Cartesian and non-swirling axisymmetric geometries.

## Current capabilities

### Cartesian hydrodynamics

* D2Q9 lattice
* second-order equilibrium distribution
* BGK collision operator
* periodic streaming
* Guo forcing
* halfway bounce-back walls
* physical macroscopic variable reconstruction
* steady-state convergence monitoring

### Axisymmetric hydrodynamics

The axisymmetric solver uses a D2Q9 formulation in cylindrical coordinates

$$
(z,r)
$$

for non-swirling flows with

$$
\mathbf{u}=(u_z,u_r).
$$

It includes:

* symmetry treatment at \(r=0\)
* halfway bounce-back at solid radial walls
* axisymmetric source terms
* axisymmetric macroscopic reconstruction
* external body forces
* physical-to-lattice unit conversion

## Numerical verification

The code currently includes automated tests for:

* D2Q9 quadrature identities
* equilibrium moments
* mass and momentum conservation during collision
* periodic streaming
* Guo forcing
* uniform acceleration
* halfway bounce-back
* planar Poiseuille flow
* grid convergence of planar Poiseuille flow
* shear-wave decay
* viscosity recovery
* shear-wave grid convergence
* axisymmetric Hagen-Poiseuille flow
* Womersley flow
* manufactured non-swirling axisymmetric flow with nonzero \(u_r\)
* physical/lattice unit scaling

The manufactured axisymmetric solution demonstrates approximately second-order convergence in the axial velocity and somewhat reduced convergence in the radial velocity, with the dominant radial error localized near the symmetry axis.

## Installation

The project uses `uv` for Python environment and dependency management.

Clone the repository and run:

```bash
uv sync
```

## Running tests

Run the standard test suite:

```bash
uv run pytest -m "not slow" -v
```

Run the complete verification suite:

```bash
uv run pytest -v
```

Run a specific test:

```bash
uv run pytest tests/test_poiseuille.py -v
```

Show diagnostic output:

```bash
uv run pytest tests/test_axisymmetric_mms_convergence.py -v -s
```

## Running examples

Planar Poiseuille flow:

```bash
uv run python examples/poiseuille.py
```

Axisymmetric Hagen-Poiseuille flow:

```bash
uv run python examples/axisymmetric_poiseuille.py
```

Axisymmetric Womersley flow:

```bash
uv run python examples/womersley.py
```

## Lattice units

For D2Q9,

$$
c_s^2=\frac13
$$

and the BGK kinematic viscosity is

$$
\nu_{\mathrm{LB}}
=
c_s^2
\left(
\tau-\frac12
\right).
$$

Physical problems are converted to lattice units using diffusive scaling,

$$
\Delta t\propto\Delta x^2,
$$

which keeps \(\tau\) fixed under grid refinement and reduces the lattice Mach number as the grid is refined.

## Project structure

```text
src/lbm/
    lattices/       lattice definitions
    hydro/          collision, equilibrium, forcing and macroscopic kernels
    boundary/       streaming and boundary-condition operators
    utils/          numerical utilities
    verification/   analytical and manufactured solutions
    simulation.py
    axisymmetric_simulation.py
    units.py

examples/           executable physical examples
tests/              unit and verification tests
docs/               mathematical and architectural documentation
```

## Development philosophy

The project follows several principles:

1. Numerical kernels are kept as small, testable functions.
2. Optimization is postponed until reference implementations are verified.
3. Every new physical model should first be validated against analytical solutions, benchmark data, or manufactured solutions.
4. Cartesian and axisymmetric physics are kept separate where their kinetic formulations differ.
5. Physical-to-lattice scaling is explicit rather than hidden inside solvers.

## Planned development

The next major stage is electrostatics:

* Laplace equation
* Poisson equation
* electrostatic boundary conditions
* electric-field reconstruction
* independent verification
* coupling of electric body forces to the hydrodynamic solver

Later stages will include:

* charge transport and electrohydrodynamics
* electrokinetics
* two-phase / phase-field models
* performance optimization with Numba
* eventually parallel CPU/GPU implementations
