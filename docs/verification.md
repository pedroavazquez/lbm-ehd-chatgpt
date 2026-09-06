| Test                   | Purpose                    | Expected result              |
| ---------------------- | -------------------------- | ---------------------------- |
| D2Q9 quadrature        | lattice isotropy           | exact to roundoff            |
| Equilibrium moments    | mass/momentum flux         | exact to roundoff            |
| BGK collision          | conservation               | exact to roundoff            |
| Uniform force          | forcing integration        | \(u=at\)                     |
| Shear wave             | viscosity                  | \(\nu=c_s^2(\tau-1/2)\)      |
| Planar Poiseuille      | wall + forcing             | parabolic profile            |
| Poiseuille convergence | spatial order              | approximately 2              |
| Hagen-Poiseuille       | axisymmetric steady flow   | \(u_z\propto R^2-r^2\)       |
| Womersley              | unsteady axisymmetric flow | analytical Bessel solution   |
| Axisymmetric MMS       | general \(u_z,u_r\)        | convergence under refinement |
