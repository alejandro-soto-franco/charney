# charney

Rotating shallow water on a sphere and on an Earth-fitted oblate spheroid, in
Rust, with no dependencies.

Geometry enters through one trait. `Mesh` builds an icosahedral graph on the
unit sphere, pushes it onto whichever surface it is given, and measures every
length and area in the ambient embedding. Swapping `Sphere` for `Wgs84` leaves
the operators, the tendencies, the integrator and the diagnostics untouched.

## Discretisation

A C-grid finite-volume scheme after Thuburn et al. (2009) and Ringler et al.
(2010), the one MPAS-Atmosphere uses. Thickness sits on the cells of a
spherical centroidal Voronoi tessellation, velocity normal to each edge sits on
the edges, and vorticity sits on the dual triangles.

```text
dh/dt = -div(h u)
du/dt =  q F_perp - grad( g (h + b) + K )
```

`q` is the potential vorticity, `F_perp` the tangential mass flux reconstructed
from the TRiSK weights, `K` the kinetic energy and `b` the topography.
Integration is classical RK4.

Two conventions matter. The discrete manifold is the polyhedron inscribed in
the surface, so lengths are chords, areas are planar, and the three kites of a
triangle tile it exactly, which is what makes mass conservation exact. And the
Coriolis parameter is `2 omega n_z`, the axial component of the true surface
normal, so on WGS 84 it follows the geodetic latitude rather than the
geocentric one.

## Results

Williamson case 2, the steady zonal geostrophic flow, after five days
(`cargo run --release --example williamson2`):

| level | cells | dx (km) | l2(h) | linf(h) | order | mass | energy |
|---|---|---|---|---|---|---|---|
| 3 | 642 | 848 | 2.256e-3 | 6.197e-3 | | 1.7e-15 | -8.2e-9 |
| 4 | 2562 | 412 | 5.919e-4 | 1.569e-3 | 1.93 | 3.0e-15 | -1.7e-9 |
| 5 | 10242 | 200 | 1.675e-4 | 6.277e-4 | 1.82 | 1.8e-14 | -4.5e-10 |

Case 5, zonal flow over an isolated mountain, fifteen days at level 5
(`cargo run --release --example williamson5`): mass holds to 5e-15, total
energy drifts by 1e-9 and potential enstrophy by 5e-4. Output goes to a Zarr v3
store that `zarr-python` and `xarray` open directly.

Sphere against spheroid (`cargo run --release --example spheroid`): case 2 is
an exact steady solution on a sphere and not on WGS 84, so the spheroid run
adjusts. Reading one resolution settles nothing, since on a coarse mesh the
sphere's own drift is numerical error of the same size. Refining separates
them, because only one of the two converges away. Root-mean-square height
change after three days, in metres, on a grid and a solver shared by both, with
the sphere given the spheroid's authalic radius:

| level | cells | sphere | wgs84 | ratio |
|---|---|---|---|---|
| 3 | 642 | 4.069 | 4.245 | 1.04 |
| 4 | 2562 | 1.056 | 1.401 | 1.33 |
| 5 | 10242 | 0.293 | 0.801 | 2.73 |

## Grid relaxation

The bisected icosahedron is not centroidal, and the tangential reconstruction
is exact only on a uniform mesh. Sixty sweeps of Lloyd relaxation on the unit
sphere take the reconstruction from half-order convergence to second order and
cut its error at level 4 by a factor of seventeen. Relaxation runs on the unit
sphere and the result is embedded afterwards, so a sphere and a spheroid share
one grid.

```rust
use charney::{Mesh, Sphere, Wgs84};

let on_a_sphere = Mesh::scvt(5, 60, &Sphere::EARTH);
let on_wgs84 = Mesh::scvt(5, 60, &Wgs84::WGS84);
```

## Verification

`cargo test` runs 29 tests. The ladder is: mesh counts and Euler
characteristic; second-order convergence of the polyhedral area; exact kite
tiling; orthonormal edge frames; `curl grad = 0` to rounding; net divergence
and net circulation zero to rounding; antisymmetric TRiSK weights; second-order
tangential reconstruction on both surfaces; the case-2 state discretely
balanced; then the four end-to-end runs.

## Scope

One layer, no tracers, no physics, no parallelism. The scheme conserves mass
exactly and energy nearly, and potential enstrophy only approximately. Grid
imprinting at the twelve pentagons is visible in the momentum residual and
grows as the mesh refines, which is the known behaviour of this scheme on an
icosahedral grid and would want explicit dissipation in a longer run.

## References

- D. L. Williamson, J. B. Drake, J. J. Hack, R. Jakob and P. N. Swarztrauber.
  A standard test set for numerical approximations to the shallow water
  equations in spherical geometry. *J. Comput. Phys.* 102 (1992) 211.
- J. Thuburn, T. D. Ringler, W. C. Skamarock and J. B. Klemp. Numerical
  representation of geostrophic modes on arbitrarily structured C-grids.
  *J. Comput. Phys.* 228 (2009) 8321.
- T. D. Ringler, J. Thuburn, J. B. Klemp and W. C. Skamarock. A unified
  approach to energy conservation and potential vorticity dynamics for
  arbitrarily-structured C-grids. *J. Comput. Phys.* 229 (2010) 3065.

## Licence

MIT or Apache-2.0, at your option.
