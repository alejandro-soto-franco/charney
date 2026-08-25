# charney

Rotating shallow water on a sphere and on an Earth-fitted oblate spheroid, in
Rust, with no dependencies. `cargo tree` prints one line.

Geometry enters through one trait. `Mesh` builds an icosahedral graph on the
unit sphere, pushes it onto whichever surface it is given, and measures every
length and area in the ambient embedding. Swapping `Sphere` for `Wgs84` leaves
the operators, the tendencies, the integrator and the diagnostics untouched.

## Discretisation

A C-grid finite-volume scheme after Thuburn et al. [\[3\]](#ref-3) and Ringler
et al. [\[2\]](#ref-2), the one MPAS-Atmosphere uses. Thickness sits on the cells of a
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

Velocity reaches a cell centre, for output and for a Courant number, through
the Perot reconstruction [\[7\]](#ref-7).

Two conventions matter. The discrete manifold is the polyhedron inscribed in
the surface, so lengths are chords, areas are planar, and the three kites of a
triangle tile it exactly, which is what makes mass conservation exact. And the
Coriolis parameter is `2 omega n_z`, the axial component of the true surface
normal, so on WGS 84 it follows the geodetic latitude rather than the
geocentric one.

## Results

Williamson [\[4\]](#ref-4) case 2, the steady zonal geostrophic flow, after
five days
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
after Barth and Jespersen [\[1\]](#ref-1), which removes the undershoot
outright and pays by clipping smooth extrema. The last is the default.

## Grid relaxation

The bisected icosahedron is not centroidal, and the tangential reconstruction
is exact only on a uniform mesh. Sixty sweeps of Lloyd relaxation
[\[5\]](#ref-5), [\[6\]](#ref-6) on the unit sphere take the reconstruction
from half-order convergence to second order and cut its error at level 4 by a
factor of seventeen. Relaxation runs on the unit sphere and the result is
embedded afterwards, so a sphere and a spheroid share one grid. Moving a cell
centre changes no incidence, so the topology is built once and a sweep touches
positions alone: sixty sweeps at level 6 take a quarter of a second.

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

## Parallelism

`--features parallel` splits the loops with `rayon`. It is off by default and
the crate then has no dependencies at all. Nothing else changes: the same
answers come out, bit for bit.

```console
$ cargo run --release --features parallel --bin charney -- case5 --level 6
```

## Command line

```text
charney <case> [--level N] [--days D] [--every HOURS] [--surface sphere|wgs84]
               [--damping F] [--out DIR]
```

```console
$ charney case5 --level 5 --days 15 --out out/mountain.zarr
$ charney case2 --surface wgs84 --level 5
```

## Performance

An AMD Ryzen 9 8940HX, sixteen cores, one core per step unless the `parallel`
feature is on:

| level | cells | mesh (s) | step, 1 thread (ms) | step, 8 threads (ms) | a model day (s) |
|---|---|---|---|---|---|
| 4 | 2562 | 0.01 | 0.40 | 0.44 | 0.03 |
| 5 | 10242 | 0.06 | 1.64 | 1.78 | 0.27 |
| 6 | 40962 | 0.24 | 8.89 | 5.82 | 1.93 |
| 7 | 163842 | 1.20 | 48.85 | 27.38 | 18.17 |

The step is four RK4 stages, each a handful of passes over the edges, the cells
and the duals. Every pass gathers through an index, so the loops are bound by
memory rather than by arithmetic: the parallel speedup saturates near 1.8 at
eight threads and then falls back. That ceiling, not the pool, is also why the
last two columns are close on a coarse mesh, where a slice shorter than sixteen
thousand entries stays on the calling thread.

What bought the serial time was layout. The incidence arrays are packed rows in
one block rather than a `Vec` per cell; the orientation signs are folded into
the lengths they multiply, so an operator reads one contiguous stream where it
used to gather a length and a sign; the average onto the duals gathers where it
used to scatter; and the potential vorticity sits beside the mass flux in one
array, since the Coriolis loop gathers both for all ten stencil neighbours of
every edge and one stream halves the cache lines it touches.

`examples/checksum` prints a digest of a run, and the serial and parallel builds
print the same digits, so chunking changes no answer.

```console
$ cargo run --release --example bench
$ cargo run --release --features parallel --example checksum
```

## Verification

`cargo test` runs 45 tests in about ten seconds, and the same 45 pass under
`--features parallel`. The ladder is: mesh counts
and Euler characteristic; second-order convergence of the polyhedral area; exact
kite tiling; orthonormal edge frames and wind basis; `curl grad = 0` to
rounding; net divergence and net circulation zero to rounding; antisymmetric
TRiSK weights; second-order tangential reconstruction on both surfaces; a
self-adjoint, negative semi-definite vector Laplacian; scale-selective damping;
a convergent least-squares gradient and a limiter that bounds a step and leaves
a smooth field alone; a uniform tracer that stays uniform under a divergent
flow; packed rows that come back as they went in; chunk boundaries that cover
every index once; the case-2 state discretely balanced; then six end-to-end
runs.

## Scope

One layer, no physics, no parallelism. The scheme conserves mass and tracer mass
exactly and energy nearly, and potential enstrophy only approximately. Grid
imprinting at the twelve pentagons shows in the momentum residual and grows as
the mesh refines, which is the known behaviour of this scheme on an icosahedral
grid and is what the biharmonic term is there for.

## References

<a id="ref-1"></a>[1] T. J. Barth and D. C. Jespersen. The design and
application of upwind schemes on unstructured meshes. *27th Aerospace Sciences
Meeting*, AIAA paper 89-0366, 1989.
[doi:10.2514/6.1989-366](https://doi.org/10.2514/6.1989-366)

<a id="ref-2"></a>[2] T. D. Ringler, J. Thuburn, J. B. Klemp and W. C.
Skamarock. A unified approach to energy conservation and potential vorticity
dynamics for arbitrarily-structured C-grids. *Journal of Computational Physics*
229(9), 3065-3090, 2010.
[doi:10.1016/j.jcp.2009.12.007](https://doi.org/10.1016/j.jcp.2009.12.007)

<a id="ref-3"></a>[3] J. Thuburn, T. D. Ringler, W. C. Skamarock and J. B.
Klemp. Numerical representation of geostrophic modes on arbitrarily structured
C-grids. *Journal of Computational Physics* 228(22), 8321-8335, 2009.
[doi:10.1016/j.jcp.2009.08.006](https://doi.org/10.1016/j.jcp.2009.08.006)

<a id="ref-4"></a>[4] D. L. Williamson, J. B. Drake, J. J. Hack, R. Jakob and
P. N. Swarztrauber. A standard test set for numerical approximations to the
shallow water equations in spherical geometry. *Journal of Computational
Physics* 102(1), 211-224, 1992.
[doi:10.1016/S0021-9991(05)80016-6](https://doi.org/10.1016/S0021-9991(05)80016-6)

<a id="ref-5"></a>[5] Q. Du, V. Faber and M. Gunzburger. Centroidal Voronoi
tessellations: applications and algorithms. *SIAM Review* 41(4), 637-676, 1999.
[doi:10.1137/S0036144599352836](https://doi.org/10.1137/S0036144599352836)

<a id="ref-6"></a>[6] S. P. Lloyd. Least squares quantization in PCM. *IEEE
Transactions on Information Theory* 28(2), 129-137, 1982.
[doi:10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489)

<a id="ref-7"></a>[7] J. B. Perot. Conservation properties of unstructured
staggered mesh schemes. *Journal of Computational Physics* 159(1), 58-89, 2000.
[doi:10.1006/jcph.2000.6424](https://doi.org/10.1006/jcph.2000.6424)

## Licence

MIT or Apache-2.0, at your option.
