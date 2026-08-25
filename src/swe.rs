//! Rotating shallow water in vector-invariant form on a C grid.
//!
//! Thickness sits on cells, the velocity component normal to each edge sits on
//! edges, and vorticity sits on duals. The system is
//!
//! ```text
//! dh/dt      = -div( h u )
//! du/dt      =  q F_perp - grad( g (h + b) + K ) - nu4 lap lap u
//! d(h q_k)/dt = -div( h u q_k )
//! ```
//!
//! where `q` is the potential vorticity, `F_perp` the tangential mass flux
//! from the TRiSK weights, `K` the kinetic energy, `b` the topography and
//! `q_k` a passive tracer. Thickness and tracer mass both move by a
//! divergence of edge fluxes, which telescopes, so their totals are conserved
//! to rounding.

use crate::dec::{self, Signs};
use crate::geometry::Surface;
use crate::mesh::Mesh;
use crate::transport::{Gradients, TracerFlux};
use crate::trisk::Weights;
use crate::vec3::Vec3;

/// Prognostic fields.
///
/// Tracers are stored as column mass `h q` rather than as mixing ratio `q`,
/// so their transport is a flux divergence like the thickness equation and
/// every tracer total is conserved to rounding. [`Model::mixing_ratio`]
/// recovers `q`.
#[derive(Clone)]
pub struct State {
    /// Fluid thickness on cells.
    pub h: Vec<f64>,
    /// Velocity normal to each edge, along `Mesh::edge_normal`.
    pub u: Vec<f64>,
    /// Column mass `h q` of each passive tracer, on cells.
    pub tracer_mass: Vec<Vec<f64>>,
}

impl State {
    pub fn zeros(mesh: &Mesh) -> State {
        State::with_tracers(mesh, 0)
    }

    pub fn with_tracers(mesh: &Mesh, tracers: usize) -> State {
        State {
            h: vec![0.0; mesh.n_cells()],
            u: vec![0.0; mesh.n_edges()],
            tracer_mass: vec![vec![0.0; mesh.n_cells()]; tracers],
        }
    }

    pub fn n_tracers(&self) -> usize {
        self.tracer_mass.len()
    }

    fn axpy(&mut self, a: f64, d: &State) {
        for (x, y) in self.h.iter_mut().zip(&d.h) {
            *x += a * y;
        }
        for (x, y) in self.u.iter_mut().zip(&d.u) {
            *x += a * y;
        }
        for (t, dt) in self.tracer_mass.iter_mut().zip(&d.tracer_mass) {
            for (x, y) in t.iter_mut().zip(dt) {
                *x += a * y;
            }
        }
    }

    fn set_combination(&mut self, base: &State, a: f64, d: &State) {
        self.h.copy_from_slice(&base.h);
        self.u.copy_from_slice(&base.u);
        for (t, b) in self.tracer_mass.iter_mut().zip(&base.tracer_mass) {
            t.copy_from_slice(b);
        }
        self.axpy(a, d);
    }
}

/// Conserved sums, for watching a run.
#[derive(Clone, Debug, PartialEq)]
pub struct Invariants {
    pub mass: f64,
    pub energy: f64,
    pub potential_enstrophy: f64,
    /// Total column mass of each tracer.
    pub tracer_mass: Vec<f64>,
}

/// Mesh, constants and scratch space for one configuration.
pub struct Model {
    pub mesh: Mesh,
    pub signs: Signs,
    pub weights: Weights,
    /// Gravity.
    pub g: f64,
    /// Coriolis parameter at each dual, `2 omega sin(geodetic latitude)`.
    pub f_dual: Vec<f64>,
    /// Bottom height above the reference surface, per cell.
    pub bottom: Vec<f64>,
    /// Biharmonic viscosity on the velocity, m^4/s. Zero disables it.
    ///
    /// The scheme leaves a grid imprint at the twelve pentagons of the
    /// icosahedral mesh, and it grows as the mesh refines, so a long run wants
    /// something to remove it. `del^4` damps the shortest waves and leaves
    /// solid-body rotation untouched, that flow having neither divergence nor
    /// a vorticity gradient.
    pub nu4: f64,
    /// Hold the flow fixed and advect the tracers alone. For transport tests.
    pub frozen_flow: bool,
    /// How the mixing ratio reaches an edge for the tracer flux.
    pub tracer_flux_scheme: TracerFlux,

    h_edge: Vec<f64>,
    flux: Vec<f64>,
    zeta: Vec<f64>,
    h_dual: Vec<f64>,
    q_dual: Vec<f64>,
    q_edge: Vec<f64>,
    kinetic: Vec<f64>,
    phi: Vec<f64>,
    grad: Vec<f64>,
    div_u: Vec<f64>,
    lap_u: Vec<f64>,
    lap2_u: Vec<f64>,
    tracer_flux: Vec<f64>,
    scratch_cell: Vec<f64>,
    gradients: Gradients,
    tracer_gradient: Vec<(f64, f64)>,
    mixing: Vec<f64>,
}

/// Earth's rotation rate, rad/s.
pub const OMEGA: f64 = 7.292e-5;
/// Standard gravity used throughout the Williamson suite, m/s^2.
pub const GRAVITY: f64 = 9.806_16;

impl Model {
    pub fn new<S: Surface + ?Sized>(mesh: Mesh, surface: &S, g: f64, omega: f64) -> Model {
        let signs = Signs::new(&mesh);
        let weights = Weights::new(&mesh);
        let gradients = Gradients::new(&mesh);
        let f_dual = (0..mesh.n_duals())
            .map(|v| 2.0 * omega * surface.sin_geodetic_latitude(mesh.dual_site[v]))
            .collect();
        let (nc, ne, nv) = (mesh.n_cells(), mesh.n_edges(), mesh.n_duals());
        Model {
            mesh,
            signs,
            weights,
            g,
            f_dual,
            bottom: vec![0.0; nc],
            nu4: 0.0,
            frozen_flow: false,
            tracer_flux_scheme: TracerFlux::default(),
            h_edge: vec![0.0; ne],
            flux: vec![0.0; ne],
            zeta: vec![0.0; nv],
            h_dual: vec![0.0; nv],
            q_dual: vec![0.0; nv],
            q_edge: vec![0.0; ne],
            kinetic: vec![0.0; nc],
            phi: vec![0.0; nc],
            grad: vec![0.0; ne],
            div_u: vec![0.0; nc],
            lap_u: vec![0.0; ne],
            lap2_u: vec![0.0; ne],
            tracer_flux: vec![0.0; ne],
            scratch_cell: vec![0.0; nc],
            gradients,
            tracer_gradient: vec![(0.0, 0.0); nc],
            mixing: vec![0.0; nc],
        }
    }

    /// Time step from the gravity-wave speed and the shortest edge.
    pub fn stable_step(&self, mean_depth: f64, courant: f64) -> f64 {
        courant * self.mesh.min_dc() / (self.g * mean_depth).sqrt()
    }

    /// Largest eigenvalue of the discrete biharmonic operator, by power
    /// iteration. Sets what the explicit time step can tolerate.
    pub fn biharmonic_spectral_radius(&mut self) -> f64 {
        let n = self.mesh.n_edges();
        let mut v: Vec<f64> = (0..n)
            .map(|e| if e % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut lap = vec![0.0; n];
        let mut out = vec![0.0; n];
        let mut lambda = 0.0;
        for _ in 0..40 {
            self.vector_laplacian(&v, &mut lap);
            self.vector_laplacian(&lap, &mut out);
            lambda = (out.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt()
                / (v.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
            let scale = out.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            for (x, &y) in v.iter_mut().zip(out.iter()) {
                *x = y / scale;
            }
        }
        lambda
    }

    /// Biharmonic viscosity as a fraction of what the time step allows.
    ///
    /// Explicit RK4 is stable on the real axis out to about `2.78 / lambda`,
    /// with `lambda` the spectral radius of `nu4 lap lap`. Setting `nu4` from
    /// a damping time alone ignores that and blows up on a coarse mesh, where
    /// the gravity-wave time step is long. A fraction around a half damps the
    /// grid scale hard and leaves margin.
    pub fn nu4_stable(&mut self, dt: f64, fraction: f64) -> f64 {
        fraction * 2.78 / (dt * self.biharmonic_spectral_radius())
    }

    /// Time constant the given `nu4` imposes on the shortest resolved wave.
    pub fn damping_time(&mut self, nu4: f64) -> f64 {
        1.0 / (nu4 * self.biharmonic_spectral_radius())
    }

    fn diagnose(&mut self, s: &State) {
        let m = &self.mesh;
        for e in 0..m.n_edges() {
            let [c0, c1] = m.cells_on_edge[e];
            self.h_edge[e] = 0.5 * (s.h[c0] + s.h[c1]);
            self.flux[e] = self.h_edge[e] * s.u[e];
        }
        dec::curl(m, &self.signs, &s.u, &mut self.zeta);
        dec::cell_to_dual(m, &s.h, &mut self.h_dual);
        for v in 0..m.n_duals() {
            self.q_dual[v] = (self.zeta[v] + self.f_dual[v]) / self.h_dual[v];
        }
        for e in 0..m.n_edges() {
            let [v0, v1] = m.duals_on_edge[e];
            self.q_edge[e] = 0.5 * (self.q_dual[v0] + self.q_dual[v1]);
        }
        for c in 0..m.n_cells() {
            let mut acc = 0.0;
            for &e in &m.edges_on_cell[c] {
                acc += m.dc[e] * m.dv[e] * s.u[e] * s.u[e];
            }
            self.kinetic[c] = 0.25 * acc / m.area_cell[c];
            self.phi[c] = self.g * (s.h[c] + self.bottom[c]) + self.kinetic[c];
        }
    }

    /// Vector Laplacian of an edge field, as `grad div - curl curl`.
    fn vector_laplacian(&mut self, u: &[f64], out: &mut [f64]) {
        let m = &self.mesh;
        dec::divergence(m, &self.signs, u, &mut self.div_u);
        dec::curl(m, &self.signs, u, &mut self.zeta);
        for (e, o) in out.iter_mut().enumerate() {
            let [c0, c1] = m.cells_on_edge[e];
            let [v0, v1] = m.duals_on_edge[e];
            *o = (self.div_u[c1] - self.div_u[c0]) / m.dc[e]
                - (self.zeta[v1] - self.zeta[v0]) / m.dv[e];
        }
    }

    /// Right-hand side of the system at `s`.
    pub fn tendency(&mut self, s: &State, out: &mut State) {
        self.diagnose(s);

        if self.frozen_flow {
            out.h.iter_mut().for_each(|x| *x = 0.0);
            out.u.iter_mut().for_each(|x| *x = 0.0);
        } else {
            self.flow_tendency(s, out);
        }
        self.tracer_tendency(s, out);
    }

    fn flow_tendency(&mut self, s: &State, out: &mut State) {
        dec::divergence(&self.mesh, &self.signs, &self.flux, &mut out.h);
        out.h.iter_mut().for_each(|x| *x = -*x);

        dec::gradient(&self.mesh, &self.phi, &mut self.grad);
        for e in 0..self.mesh.n_edges() {
            let (es, ws) = self.weights.stencil(e);
            let mut coriolis = 0.0;
            for (&eoe, &wt) in es.iter().zip(ws) {
                let eoe = eoe as usize;
                coriolis += wt * 0.5 * (self.q_edge[e] + self.q_edge[eoe]) * self.flux[eoe];
            }
            out.u[e] = coriolis - self.grad[e];
        }

        if self.nu4 != 0.0 {
            let mut lap = std::mem::take(&mut self.lap_u);
            let mut lap2 = std::mem::take(&mut self.lap2_u);
            self.vector_laplacian(&s.u, &mut lap);
            self.vector_laplacian(&lap, &mut lap2);
            for (o, &d) in out.u.iter_mut().zip(lap2.iter()) {
                *o -= self.nu4 * d;
            }
            self.lap_u = lap;
            self.lap2_u = lap2;
        }
    }

    fn tracer_tendency(&mut self, s: &State, out: &mut State) {
        for k in 0..s.n_tracers() {
            // Interpolating the mixing ratio, rather than the column mass,
            // is what keeps a uniform tracer uniform under a divergent flow.
            for (q, (&hq, &h)) in self
                .mixing
                .iter_mut()
                .zip(s.tracer_mass[k].iter().zip(&s.h))
            {
                *q = hq / h;
            }
            if self.tracer_flux_scheme != TracerFlux::Centered {
                self.gradients
                    .compute(&self.mesh, &self.mixing, &mut self.tracer_gradient);
                if self.tracer_flux_scheme == TracerFlux::LimitedUpwind {
                    self.gradients
                        .limit(&self.mesh, &self.mixing, &mut self.tracer_gradient);
                }
            }

            let m = &self.mesh;
            for e in 0..m.n_edges() {
                let [c0, c1] = m.cells_on_edge[e];
                let q_edge = match self.tracer_flux_scheme {
                    TracerFlux::Centered => 0.5 * (self.mixing[c0] + self.mixing[c1]),
                    _ => {
                        // A positive normal velocity runs from the first cell
                        // to the second, so the first cell is upwind.
                        let from = if self.flux[e] >= 0.0 { c0 } else { c1 };
                        let slot = Gradients::slot_of(m, from, e);
                        self.gradients.at_edge(
                            from,
                            slot,
                            self.mixing[from],
                            self.tracer_gradient[from],
                        )
                    }
                };
                self.tracer_flux[e] = self.flux[e] * q_edge;
            }
            dec::divergence(m, &self.signs, &self.tracer_flux, &mut self.scratch_cell);
            for (o, &d) in out.tracer_mass[k].iter_mut().zip(self.scratch_cell.iter()) {
                *o = -d;
            }
        }
    }

    /// One classical fourth-order Runge-Kutta step.
    pub fn step(&mut self, s: &mut State, dt: f64, work: &mut Work) {
        assert_eq!(
            s.n_tracers(),
            work.k1.n_tracers(),
            "the work space must be built from a state with the same tracers"
        );
        self.tendency(s, &mut work.k1);
        work.tmp.set_combination(s, 0.5 * dt, &work.k1);
        self.tendency(&work.tmp, &mut work.k2);
        work.tmp.set_combination(s, 0.5 * dt, &work.k2);
        self.tendency(&work.tmp, &mut work.k3);
        work.tmp.set_combination(s, dt, &work.k3);
        self.tendency(&work.tmp, &mut work.k4);

        s.axpy(dt / 6.0, &work.k1);
        s.axpy(dt / 3.0, &work.k2);
        s.axpy(dt / 3.0, &work.k3);
        s.axpy(dt / 6.0, &work.k4);
    }

    /// Advance `steps` steps.
    pub fn integrate(&mut self, s: &mut State, dt: f64, steps: usize, work: &mut Work) {
        for _ in 0..steps {
            self.step(s, dt, work);
        }
    }

    /// Potential vorticity at the duals, as of the last tendency or
    /// invariants call.
    pub fn potential_vorticity(&self) -> &[f64] {
        &self.q_dual
    }

    /// Relative vorticity at the duals, as of the last tendency or
    /// invariants call.
    pub fn relative_vorticity(&self) -> &[f64] {
        &self.zeta
    }

    /// Mixing ratio of tracer `k`.
    pub fn mixing_ratio(&self, s: &State, k: usize, out: &mut [f64]) {
        for (o, (&hq, &h)) in out.iter_mut().zip(s.tracer_mass[k].iter().zip(&s.h)) {
            *o = hq / h;
        }
    }

    pub fn invariants(&mut self, s: &State) -> Invariants {
        self.diagnose(s);
        let m = &self.mesh;
        let mut mass = 0.0;
        let mut energy = 0.0;
        for c in 0..m.n_cells() {
            mass += m.area_cell[c] * s.h[c];
            energy += m.area_cell[c]
                * (s.h[c] * self.kinetic[c]
                    + 0.5 * self.g * s.h[c] * (s.h[c] + 2.0 * self.bottom[c]));
        }
        let mut enstrophy = 0.0;
        for v in 0..m.n_duals() {
            enstrophy += 0.5 * m.area_dual[v] * self.h_dual[v] * self.q_dual[v] * self.q_dual[v];
        }
        let tracer_mass = s
            .tracer_mass
            .iter()
            .map(|hq| (0..m.n_cells()).map(|c| m.area_cell[c] * hq[c]).sum())
            .collect();
        Invariants {
            mass,
            energy,
            potential_enstrophy: enstrophy,
            tracer_mass,
        }
    }

    /// Velocity at each cell centre by the Perot reconstruction.
    ///
    /// On an orthogonal mesh the arm from the cell centre to an edge midpoint
    /// runs along that edge's outward normal, so the two orientation signs in
    /// the general formula cancel and none appears here.
    pub fn cell_velocity(&self, s: &State, out: &mut [Vec3]) {
        let m = &self.mesh;
        for (c, o) in out.iter_mut().enumerate() {
            let mut v = Vec3::ZERO;
            for &e in &m.edges_on_cell[c] {
                let arm = (m.edge_pos[e] - m.cell_pos[c]).norm();
                v += m.edge_normal[e] * (m.dv[e] * arm * s.u[e]);
            }
            *o = v / m.area_cell[c];
        }
    }

    /// Speed at each cell centre.
    pub fn cell_speed(&self, s: &State, out: &mut [f64]) {
        let mut v = vec![Vec3::ZERO; out.len()];
        self.cell_velocity(s, &mut v);
        for (o, w) in out.iter_mut().zip(v) {
            *o = w.norm();
        }
    }
}

/// Scratch state for the integrator, allocated once.
pub struct Work {
    k1: State,
    k2: State,
    k3: State,
    k4: State,
    tmp: State,
}

impl Work {
    /// Work space matching a state's tracer count.
    pub fn for_state(mesh: &Mesh, s: &State) -> Work {
        let blank = || State::with_tracers(mesh, s.n_tracers());
        Work {
            k1: blank(),
            k2: blank(),
            k3: blank(),
            k4: blank(),
            tmp: blank(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;

    /// Perot reconstruction against the field it was built from.
    #[test]
    fn cell_velocity_recovers_a_solid_body_flow() {
        let surface = Sphere::EARTH;
        let mesh = Mesh::scvt(4, 60, &surface);
        let spin = Vec3::new(0.0, 0.0, 1.0e-5);
        let mut state = State::zeros(&mesh);
        for e in 0..mesh.n_edges() {
            state.u[e] = spin.cross(mesh.edge_site[e]).dot(mesh.edge_normal[e]);
        }
        let exact: Vec<Vec3> = (0..mesh.n_cells())
            .map(|c| spin.cross(mesh.cell_pos[c]))
            .collect();

        let model = Model::new(mesh, &surface, GRAVITY, OMEGA);
        let mut got = vec![Vec3::ZERO; model.mesh.n_cells()];
        model.cell_velocity(&state, &mut got);

        let num: f64 = got
            .iter()
            .zip(&exact)
            .map(|(a, b)| (*a - *b).norm().powi(2))
            .sum();
        let den: f64 = exact.iter().map(|b| b.norm().powi(2)).sum();
        assert!(
            (num / den).sqrt() < 0.02,
            "relative error {}",
            (num / den).sqrt()
        );
    }

    /// The discrete vector Laplacian is self-adjoint in the energy inner
    /// product and negative semi-definite. Together those give the biharmonic
    /// term `-nu4 lap lap u` a sign: it can only remove kinetic energy.
    #[test]
    fn the_vector_laplacian_dissipates() {
        let surface = Sphere::EARTH;
        let mesh = Mesh::scvt(3, 60, &surface);
        let mut model = Model::new(mesh, &surface, GRAVITY, OMEGA);
        let n = model.mesh.n_edges();

        let wiggle =
            |seed: f64| -> Vec<f64> { (0..n).map(|e| ((e as f64) * seed).sin()).collect() };
        let (a, b) = (wiggle(0.618_033_988_749), wiggle(1.324_717_957_244));
        let mut la = vec![0.0; n];
        let mut lb = vec![0.0; n];
        model.vector_laplacian(&a, &mut la);
        model.vector_laplacian(&b, &mut lb);

        let inner = |x: &[f64], y: &[f64]| -> f64 {
            (0..n)
                .map(|e| model.mesh.dc[e] * model.mesh.dv[e] * x[e] * y[e])
                .sum()
        };
        let (ab, ba) = (inner(&a, &lb), inner(&b, &la));
        assert!(
            (ab - ba).abs() / (ab.abs() + ba.abs()) < 1e-12,
            "not self-adjoint: {ab} against {ba}"
        );
        assert!(inner(&a, &la) < 0.0, "the Laplacian added energy");
    }

    /// Damping must bite at the grid scale and barely touch a smooth flow.
    #[test]
    fn hyperdiffusion_is_scale_selective() {
        let surface = Sphere::EARTH;
        let mesh = Mesh::scvt(4, 60, &surface);
        let spin = Vec3::new(0.0, 0.0, 1.0e-5);
        let smooth: Vec<f64> = (0..mesh.n_edges())
            .map(|e| spin.cross(mesh.edge_site[e]).dot(mesh.edge_normal[e]))
            .collect();
        let rough: Vec<f64> = (0..mesh.n_edges())
            .map(|e| if e % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let mut model = Model::new(mesh, &surface, GRAVITY, OMEGA);
        let decay = |model: &mut Model, u: &[f64]| {
            let mut l1 = vec![0.0; u.len()];
            let mut l2 = vec![0.0; u.len()];
            model.vector_laplacian(u, &mut l1);
            model.vector_laplacian(&l1, &mut l2);
            rms(&l2) / rms(u)
        };
        let ratio = decay(&mut model, &rough) / decay(&mut model, &smooth);
        assert!(
            ratio > 5e3,
            "grid-scale damping only {ratio} times the smooth rate"
        );
    }

    /// A uniform tracer must stay uniform under any flow, however divergent.
    #[test]
    fn a_uniform_tracer_stays_uniform() {
        let surface = Sphere::EARTH;
        let mesh = Mesh::scvt(3, 60, &surface);
        let mut state = State::with_tracers(&mesh, 1);
        for c in 0..mesh.n_cells() {
            state.h[c] = 1000.0 + 40.0 * mesh.cell_pos[c].z / surface.radius;
            state.tracer_mass[0][c] = state.h[c];
        }
        for e in 0..mesh.n_edges() {
            state.u[e] = 12.0 * (mesh.edge_site[e].x / surface.radius).sin();
        }

        let mut model = Model::new(mesh, &surface, GRAVITY, OMEGA);
        let mut work = Work::for_state(&model.mesh, &state);
        model.integrate(&mut state, 60.0, 200, &mut work);

        let mut q = vec![0.0; model.mesh.n_cells()];
        model.mixing_ratio(&state, 0, &mut q);
        let worst = q.iter().map(|x| (x - 1.0).abs()).fold(0.0f64, f64::max);
        assert!(worst < 1e-12, "mixing ratio strayed by {worst}");
    }

    /// Second order on a tracer with no kink in it. Williamson case 1 gets
    /// nearer first order, and that is the cosine bell's corner rather than
    /// the operator: the bell's first derivative jumps at its rim.
    #[test]
    fn transport_is_second_order_on_a_smooth_tracer() {
        let surface = Sphere::EARTH;
        let error = |level: u32| {
            let mesh = Mesh::scvt(level, 60, &surface);
            let u0 = 2.0 * std::f64::consts::PI * surface.radius / (12.0 * 86_400.0);
            let centre = Vec3::new(1.0, 0.0, 0.0);
            let mut state = State::with_tracers(&mesh, 1);
            for c in 0..mesh.n_cells() {
                state.h[c] = 1000.0;
                let arc = mesh.cell_pos[c]
                    .normalized()
                    .dot(centre)
                    .clamp(-1.0, 1.0)
                    .acos();
                state.tracer_mass[0][c] = 1000.0 * (1.0 + (-(arc / 0.5).powi(2)).exp());
            }
            for e in 0..mesh.n_edges() {
                let spin = Vec3::new(0.0, 0.0, u0 / surface.radius);
                state.u[e] = spin.cross(mesh.edge_site[e]).dot(mesh.edge_normal[e]);
            }

            let initial = state.clone();
            let dx = mesh.min_dc();
            let mut model = Model::new(mesh, &surface, GRAVITY, OMEGA);
            model.frozen_flow = true;
            let mut work = Work::for_state(&model.mesh, &state);
            let period = 12.0 * 86_400.0;
            let steps = (period / (0.4 * dx / u0)).ceil() as usize;
            model.integrate(&mut state, period / steps as f64, steps, &mut work);

            let m = &model.mesh;
            let (mut num, mut den) = (0.0, 0.0);
            for c in 0..m.n_cells() {
                let d = state.tracer_mass[0][c] - initial.tracer_mass[0][c];
                num += m.area_cell[c] * d * d;
                den += m.area_cell[c] * initial.tracer_mass[0][c].powi(2);
            }
            (num / den).sqrt()
        };
        let (coarse, fine) = (error(3), error(4));
        let rate = (coarse / fine).log2();
        assert!(
            rate > 1.4,
            "convergence rate {rate}, from {coarse} to {fine}"
        );
    }

    fn rms(v: &[f64]) -> f64 {
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
    }
}
