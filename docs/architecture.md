# Architecture

## Overview

The code uses different numerical methods for different physical subsystems.

```text
        Electrostatic / AC problem
                  |
                  v
      Spectral-element solver
                  |
             phi, E
                  |
                  v
        SEM -> LBM interpolation
                  |
                  v
          Electric body force
                  |
                  v
       Cartesian / axisymmetric
             LBM solver
                  |
                  v
             velocity
```

The long-term multiphysics architecture will add charge transport and a phase-field subsystem.

## Hydrodynamic solver

Hydrodynamics uses D2Q9 LBM.

The main numerical kernels are kept as independent functions:

```text
equilibrium
macroscopic reconstruction
collision
forcing
streaming
boundary treatment
```

State management is handled by simulation classes.

### Cartesian

```text
LBMSimulation
```

supports standard BGK hydrodynamics and external forcing.

Boundary treatment is injected through

```python
sim.step(stream_operator=...)
```

rather than being hard-coded into the simulation object.

### Axisymmetric

```text
AxisymmetricLBMSimulation
```

uses a separate kinetic formulation because cylindrical source terms and macroscopic reconstruction differ from the Cartesian case.

Coordinates are

$$
(z,r).
$$

## Spectral-element electrostatic solver

The electrostatic solver uses continuous nodal spectral elements based on Gauss-Lobatto-Legendre points.

For one-dimensional elements,

$$
M=\operatorname{diag}(w_i),
$$

and

$$
K=D^TMD.
$$

For two-dimensional tensor-product elements,

$$
M_{2D}
=
M_y\otimes M_x,
$$

and for constant coefficients,

$$
K_{2D}
=
M_y\otimes K_x
+
K_y\otimes M_x.
$$

For a variable coefficient,

$$
-\nabla\cdot
(\kappa\nabla\phi)=f,
$$

the operator is assembled using nodal quadrature of

$$
\int_\Omega
\kappa
\nabla N_i\cdot\nabla N_j\,d\Omega.
$$

The coefficient may be real or complex.

## AC formulation

For harmonic electric fields,

$$
\kappa
=
\sigma+i\omega\epsilon.
$$

The solver uses native complex arithmetic.

The potential phasor satisfies

$$
\nabla\cdot
\left[
(\sigma+i\omega\epsilon)
\nabla\tilde\phi
\right]
=
0.
$$

The electric-field phasor is

$$
\tilde{\mathbf E}
=
-\nabla\tilde\phi.
$$

## Boundary conditions

The SEM solver currently supports:

### Dirichlet

$$
\phi=\phi_D.
$$

### Neumann

$$
\mathbf n\cdot\kappa\nabla\phi
=
g_N.
$$

### Robin

$$
\mathbf n\cdot\kappa\nabla\phi
+
\beta\phi
=
g_R.
$$

All boundary data may be real or complex.

## Axisymmetric electrostatics

In cylindrical coordinates,

$$
-\left[
\partial_z
\left(
\kappa\partial_z\phi
\right)
+
\frac1r
\partial_r
\left(
r\kappa\partial_r\phi
\right)
\right]
=
f.
$$

The weak form is

$$
\int_\Omega
r\kappa
\nabla\phi\cdot\nabla v
\,dz\,dr
=
\int_\Omega
rfv
\,dz\,dr
+
\text{boundary terms}.
$$

No explicit division by \(r\) appears in the assembled operator.

The symmetry axis \(r=0\) is therefore handled naturally.

## SEM-to-LBM coupling

The SEM and LBM grids are geometrically aligned, but they do not use the same nodal distribution.

Each SEM element covers an integer rectangular block of LBM cells.

Interpolation is performed using tensor-product Lagrange interpolation matrices.

For a scalar field,

$$
F_{\rm LB}
=
P_yF_{\rm SEM}P_x^T.
$$

The matrices \(P_x\) and \(P_y\) depend only on:

* SEM polynomial order;
* target LBM node positions inside the element.

They are precomputed once and reused.

The same operator is applied independently to:

$$
\phi,
\quad
E_x,
\quad
E_y.
$$

## Phase-field compatibility

Future two-fluid simulations will use a diffuse phase field \(\psi\).

Properties such as

$$
\epsilon(\psi),
\qquad
\sigma(\psi),
$$

will be smooth across the diffuse interface.

This avoids introducing discontinuous electrostatic coefficients inside continuous spectral elements and allows the existing variable-coefficient formulation to be reused directly.

## Planned coupling sequence

The next multiphysics development stage is:

```text
SEM potential solve
       |
       v
spectral E-field reconstruction
       |
       v
SEM -> LBM interpolation
       |
       v
electric force evaluation
       |
       v
LBM collision + streaming
```

Later this will be extended by charge transport and phase-field evolution.
