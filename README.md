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
dh/dt       = -div( h u )
du/dt       =  q F_perp - grad( g (h + b) + K ) - nu4 lap lap u
d(h q_k)/dt = -div( h u q_k )
```

`q` is the potential vorticity, `F_perp` the tangential mass flux reconstructed
from the TRiSK weights, `K` the kinetic energy, `b` the topography and `q_k` a
passive tracer. Integration is classical RK4.

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

Case 1, a cosine bell carried once around the sphere under a frozen wind, at
level 5 (`cargo run --release --example williamson1`). The three flux forms and
what each costs:

| flux | l2(q) | order | min | max |
|---|---|---|---|---|
| centred | 6.184e-1 | 0.75 | -0.222 | 0.865 |
| upwind | 3.122e-1 | 0.93 | -0.062 | 0.801 |
| limited upwind | 2.555e-1 | 1.15 | -0.000 | 0.755 |

The bell's first derivative jumps at its rim, and that is what keeps the order
near one. On a tracer with no kink the same operator converges at 1.94.

Case 5, zonal flow over an isolated mountain, fifteen days at level 5
(`cargo run --release --example williamson5`): mass is conserved to 5e-15,
total energy drifts by 1e-9 and potential enstrophy by 5e-4.

Case 6, the Rossby-Haurwitz wave of zonal wavenumber four, fourteen days at
level 5 (`cargo run --release --example williamson6`): the pattern travels east
at 11.15 degrees a day against the 12.20 the nondivergent theory gives, 8.5 per
cent low, which is the divergence the shallow-water system has and the theory
does not. Mass is conserved to 7e-15, energy drifts by 1.8e-4 and potential
enstrophy by 4.1e-3.

Sphere against spheroid (`cargo run --release --example spheroid`): case 2 is an
exact steady solution on a sphere and not on WGS 84, so the spheroid run
adjusts. Reading one resolution settles nothing, since on a coarse mesh the
sphere's own drift is numerical error of the same size. Refining separates them,
because only one of the two converges away. Root-mean-square height change after
three days, in metres, on a grid and a solver shared by both, with the sphere
given the spheroid's authalic radius:

| level | cells | sphere | wgs84 | ratio |
|---|---|---|---|---|
| 3 | 642 | 4.069 | 4.245 | 1.04 |
| 4 | 2562 | 1.056 | 1.401 | 1.33 |
| 5 | 10242 | 0.293 | 0.801 | 2.73 |

## Tracer transport

Tracers are stored as column mass `h q`, so their transport is a flux
divergence like the thickness equation and every tracer total is conserved to
rounding. The mixing ratio reaches an edge one of three ways: a centred average,
which has no dissipation at all and rings; the upwind cell's value extrapolated
with a least-squares gradient fitted in its tangent plane; or the same limited
after Barth and Jespersen, which removes the undershoot outright and pays by
clipping smooth extrema. The last is the default.

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

## Damping

`del^4` on the velocity removes the grid imprint that the scheme leaves at the
twelve pentagons. Setting the coefficient from a damping time alone is unsafe:
explicit RK4 is stable only out to about `2.78 / lambda`, and on a coarse mesh
the gravity-wave time step is long enough to cross that. `Model::nu4_stable`
measures the spectral radius by power iteration and returns a fraction of what
the step allows; `Model::damping_time` reports what that buys at the grid scale.
Between a fifth and a half of the limit changes the case-6 pattern speed by one
part in a hundred, which is the operator being properly scale-selective.

## Output

Runs write a Zarr v3 store, laid out by hand, that `zarr-python` opens and that
`xarray` opens with longitude and latitude already attached as coordinates.
Chunks are uncompressed; a codec would mean a compression dependency, and the
trade is not worth it for output a post-processing step reads once.

## Command line

```text
charney <case> [--level N] [--days D] [--every HOURS] [--surface sphere|wgs84]
               [--damping F] [--out DIR]
```

```console
$ charney case5 --level 5 --days 15 --out out/mountain.zarr
$ charney case2 --surface wgs84 --level 5
```

## Verification

`cargo test` runs 41 tests in about sixteen seconds. The ladder is: mesh counts
and Euler characteristic; second-order convergence of the polyhedral area; exact
kite tiling; orthonormal edge frames and wind basis; `curl grad = 0` to
rounding; net divergence and net circulation zero to rounding; antisymmetric
TRiSK weights; second-order tangential reconstruction on both surfaces; a
self-adjoint, negative semi-definite vector Laplacian; scale-selective damping;
a convergent least-squares gradient and a limiter that bounds a step and leaves
a smooth field alone; a uniform tracer that stays uniform under a divergent
flow; the case-2 state discretely balanced; then six end-to-end runs.

## Scope

One layer, no physics, no parallelism. The scheme conserves mass and tracer mass
exactly and energy nearly, and potential enstrophy only approximately. Grid
imprinting at the twelve pentagons shows in the momentum residual and grows as
the mesh refines, which is the known behaviour of this scheme on an icosahedral
grid and is what the biharmonic term is there for.

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
- T. J. Barth and D. C. Jespersen. The design and application of upwind schemes
  on unstructured meshes. AIAA paper 89-0366 (1989).

## Licence

MIT or Apache-2.0, at your option.
