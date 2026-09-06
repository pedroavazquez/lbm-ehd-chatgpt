# LBM-EHD

Research-oriented numerical framework combining the Lattice Boltzmann Method (LBM) with high-order spectral-element methods for hydrodynamics, electrohydrodynamics, electrokinetics, and multiphase flows.

The project is being developed incrementally with analytical benchmarks, manufactured solutions, convergence studies, and automated regression tests.

## Current capabilities

### Cartesian LBM hydrodynamics

* D2Q9 lattice
* second-order equilibrium distribution
* BGK collision operator
* periodic streaming
* Guo forcing
* halfway bounce-back walls
* macroscopic density and velocity reconstruction
* steady-state convergence monitoring

### Axisymmetric LBM hydrodynamics

Non-swirling axisymmetric flow in cylindrical coordinates

$$
(z,r),
\qquad
\mathbf u=(u_z,u_r).
$$

Implemented features include:

* D2Q9 axisymmetric formulation
* symmetry treatment at \(r=0\)
* halfway bounce-back at solid radial walls
* axisymmetric source terms
* axisymmetric macroscopic reconstruction
* external body forcing
* physical-to-lattice scaling

### Spectral-element electrostatics

High-order continuous spectral-element solver based on Gauss-Lobatto-Legendre nodes.

Implemented features include:

* GLL nodes and quadrature
* nodal Lagrange interpolation
* spectral differentiation matrices
* 1-D element mass and stiffness matrices
* multi-element 1-D assembly
* 2-D tensor-product rectangular elements
* structured multi-element 2-D assembly
* high-order Poisson solver
* variable-coefficient elliptic operator
* real and complex coefficients
* Dirichlet boundary conditions
* Neumann boundary conditions
* Robin boundary conditions
* Cartesian electrostatic problems
* non-swirling axisymmetric electrostatic problems
* natural symmetry-axis treatment
* electric-field reconstruction
* SEM-to-LBM interpolation on aligned grids

The general electrostatic problem is

$$
-\nabla\cdot
\left(
\kappa\nabla\phi
\right)
=
f.
$$

For dielectric electrostatics,

$$
\kappa=\epsilon.
$$

For harmonic AC problems,

$$
\kappa
=
\sigma+i\omega\epsilon,
$$

and both the potential and electric field may be complex:

$$
\tilde{\mathbf E}
=
-\nabla\tilde\phi.
$$

### Axisymmetric electrostatics

In cylindrical coordinates,

$$
-\left[
\frac{\partial}{\partial z}
\left(
\kappa\frac{\partial\phi}{\partial z}
\right)
+
\frac1r
\frac{\partial}{\partial r}
\left(
r\kappa\frac{\partial\phi}{\partial r}
\right)
\right]
=
f.
$$

The spectral-element weak form uses the cylindrical volume weight \(r\), avoiding explicit division by \(r\) at the symmetry axis.

### Physical-to-lattice scaling

The code includes explicit conversion between physical and lattice units.

For D2Q9,

$$
c_s^2=\frac13,
$$

and

$$
\nu_{\rm LB}
=
c_s^2
\left(
\tau-\frac12
\right).
$$

Grid refinement uses diffusive scaling,

$$
\Delta t\propto\Delta x^2,
$$

which keeps the relaxation time fixed and reduces the lattice Mach number under refinement.

## Verification

The hydrodynamic implementation is verified using:

* equilibrium moment identities
* collision conservation tests
* streaming tests
* uniform acceleration
* shear-wave decay
* shear-wave grid convergence
* planar Poiseuille flow
* planar Poiseuille grid convergence
* axisymmetric Hagen-Poiseuille flow
* Womersley flow
* manufactured axisymmetric flow with nonzero radial velocity

The spectral-element implementation is verified using:

* GLL quadrature exactness
* polynomial differentiation
* element mass and stiffness identities
* 1-D Poisson analytical solutions
* multi-element 1-D assembly
* \(h\)-convergence
* \(p\)-convergence
* 2-D Poisson analytical solutions
* 2-D multi-element assembly
* variable-coefficient manufactured solutions
* complex AC-potential tests
* Dirichlet, Neumann, and Robin boundary conditions
* axisymmetric Poisson solutions
* axisymmetric mixed-boundary solutions
* spectral electric-field reconstruction
* exact polynomial SEM-to-LBM interpolation

## SEM/LBM coupling strategy

The electrostatic and hydrodynamic solvers use different numerical grids.

The current coupling strategy uses geometrically aligned grids:

* each spectral element covers an integer block of LBM cells;
* the SEM solution is evaluated at GLL nodes;
* electric potential and electric field are interpolated to the LBM grid;
* interpolation matrices are precomputed and reused.

For a tensor-product element,

$$
F_{\rm LB}
=
P_y F_{\rm SEM}P_x^T.
$$

This interpolation can be applied to

$$
\phi,\qquad
E_x,\qquad
E_y,
$$

and equally to complex AC fields.

## Planned multiphase model

Multiphase flows will use a diffuse-interface phase-field formulation.

Material properties will therefore vary smoothly through the interface, for example

$$
\epsilon=\epsilon(\psi),
\qquad
\sigma=\sigma(\psi).
$$

This is particularly compatible with the continuous high-order spectral-element electrostatic discretization.

## Installation

The project uses `uv`.

```bash
uv sync
```

## Running tests

Fast tests:

```bash
uv run pytest -m "not slow" -v
```

Complete verification suite:

```bash
uv run pytest -v
```

Show diagnostic output from convergence tests:

```bash
uv run pytest -v -s
```

Static analysis:

```bash
uv run ruff check .
```

Formatting:

```bash
uv run ruff format .
```

## Examples

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

## Project structure

```text
src/lbm/
    lattices/
    hydro/
    boundary/
    electrostatics/
        sem/
    coupling/
    utils/
    verification/

    simulation.py
    axisymmetric_simulation.py
    units.py

tests/
examples/
docs/
```

## Development principles

1. Numerical kernels remain small and independently testable.
2. New physical models are verified before optimization.
3. Analytical benchmarks and manufactured solutions are preferred whenever available.
4. Cartesian and axisymmetric formulations remain separated when their kinetic or weak forms differ.
5. Physical-to-lattice conversions are explicit.
6. Electrostatics and hydrodynamics are coupled through clearly defined interpolation and force interfaces.
7. Complex arithmetic is supported directly for AC electrohydrodynamics.
8. Performance optimization will follow numerical verification rather than precede it.

## Current milestone

### v0.2.0

Verified Cartesian and axisymmetric LBM hydrodynamics together with a high-order spectral-element electrostatic solver supporting real and complex potentials and aligned-grid SEM-to-LBM interpolation.

The next development stage will focus on coupling electric fields to the hydrodynamic solver and introducing charge transport.
