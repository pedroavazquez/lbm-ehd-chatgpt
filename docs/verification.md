# Verification

The project uses automated numerical verification as part of the development process.

## Hydrodynamics

### Equilibrium and collision

Tests verify:

* density recovery;
* momentum recovery;
* equilibrium fixed-point behavior;
* collision conservation of density;
* collision conservation of momentum.

### Streaming

Tests verify:

* propagation along every D2Q9 direction;
* periodic wrapping;
* global mass conservation.

### Guo forcing

Tests verify:

$$
\sum_i S_i=0,
$$

and the expected first moment of the forcing term.

Uniform forcing is verified against

$$
u(t)=at.
$$

### Shear-wave decay

The analytical solution is

$$
u_x(y,t)
=
U_0
e^{-\nu k^2t}
\sin(ky).
$$

The measured viscosity is compared with

$$
\nu
=
c_s^2
\left(
\tau-\frac12
\right).
$$

Grid refinement demonstrates approximately second-order behavior.

### Planar Poiseuille flow

For halfway bounce-back walls,

$$
u_x(y)
=
\frac{g}{2\nu}
y(H-y).
$$

The numerical solution reproduces the parabolic profile and shows approximately second-order grid convergence.

### Axisymmetric Hagen-Poiseuille flow

The analytical profile is

$$
u_z(r)
=
\frac{g}{4\nu}
\left(
R^2-r^2
\right).
$$

The test verifies the axisymmetric hydrodynamic formulation and symmetry-axis treatment.

### Womersley flow

Oscillatory pipe flow is compared with the analytical Bessel-function solution.

This verifies:

* unsteady axisymmetric behavior;
* amplitude response;
* phase response.

### Axisymmetric manufactured solution

A manufactured divergence-free velocity field with both

$$
u_z\neq0,
\qquad
u_r\neq0
$$

tests the full non-swirling cylindrical formulation.

The weighted axisymmetric norm uses the cylindrical measure

$$
r\,dr\,dz.
$$

The axial velocity converges close to second order.

The radial velocity shows somewhat reduced global order, with the largest error localized close to the symmetry axis. This behavior is retained as a documented characteristic of the current formulation.

## Spectral-element verification

### GLL quadrature

Gauss-Lobatto-Legendre quadrature with polynomial order \(p\) is verified to integrate polynomials through degree

$$
2p-1.
$$

### Spectral differentiation

The nodal differentiation matrix exactly differentiates polynomials representable by the interpolation space.

### Element matrices

Tests verify:

$$
M=\operatorname{diag}(w_i),
$$

and

$$
K=D^TMD.
$$

The stiffness matrix is symmetric and annihilates constant fields.

### 1-D Poisson equation

The problem

$$
-\phi''=
\pi^2\sin(\pi x)
$$

is tested against

$$
\phi=\sin(\pi x).
$$

Both single-element and multi-element formulations are verified.

### 1-D convergence

Two refinement strategies are tested.

#### h-refinement

Polynomial order is fixed while the number of elements increases.

#### p-refinement

The mesh is fixed while polynomial order increases.

For analytic solutions, rapid spectral convergence is observed under p-refinement.

### 2-D Poisson equation

The manufactured solution

$$
\phi(x,y)
=
\sin(\pi x)\sin(\pi y)
$$

satisfies

$$
-\nabla^2\phi
=
2\pi^2
\sin(\pi x)
\sin(\pi y).
$$

Both single-element and structured multi-element solvers are tested.

### 2-D convergence

Both h- and p-refinement tests verify high-order convergence.

### Variable coefficient

The operator

$$
-\nabla\cdot
\left(
\epsilon\nabla\phi
\right)
$$

is tested with

$$
\epsilon=1+x+y
$$

and a manufactured analytical solution.

This exercises both

$$
-\epsilon\nabla^2\phi
$$

and

$$
-\nabla\epsilon\cdot\nabla\phi.
$$

### Complex potential

Complex coefficients of the form

$$
\kappa
=
\sigma+i\omega\epsilon
$$

are tested using analytical linear-potential solutions.

The tests verify that:

* stiffness matrices become complex;
* complex Dirichlet data are supported;
* complex solutions are recovered correctly.

### Mixed boundary conditions

Tests cover:

* Dirichlet;
* zero and nonzero Neumann flux;
* real Robin conditions;
* complex Robin conditions.

### Axisymmetric electrostatics

The weak form uses the cylindrical weight \(r\).

Manufactured solutions verify the axisymmetric Poisson operator.

Mixed-boundary tests verify:

* natural symmetry at \(r=0\);
* nonzero radial flux;
* complex AC potential.

### Electric-field reconstruction

The electric field is reconstructed spectrally:

$$
\mathbf E
=
-\nabla\phi.
$$

Tests include:

* exact linear fields;
* exact polynomial fields;
* complex fields;
* electric fields reconstructed from numerical Poisson solutions.

### SEM-to-LBM interpolation

Tensor-product Lagrange interpolation is verified exactly for representable polynomials.

The interpolation supports:

* real fields;
* complex fields;
* endpoint targets;
* cell-centred targets.

The coupling relation is

$$
F_{\rm LB}
=
P_yF_{\rm SEM}P_x^T.
$$

## Regression policy

Before each milestone commit:

```bash
uv run ruff check .
uv run pytest -v
```

should pass without warnings or failures.

Slow convergence tests remain part of the full verification suite.
