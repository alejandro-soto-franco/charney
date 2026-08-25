//! Rotating shallow water in vector-invariant form on a C grid.
//!
//! Thickness sits on cells, the velocity component normal to each edge sits on
//! edges, and vorticity sits on duals. The momentum equation is
//!
//! ```text
//! du/dt = q F_perp - grad( g (h + b) + K )
//! dh/dt = -div( h u )
//! ```
//!
//! where `q` is the potential vorticity and `F_perp` the tangential mass flux
//! from the TRiSK weights. The divergence is a telescoping sum of edge fluxes,
//! so total mass is conserved to rounding.

use crate::dec::{self, Signs};
use crate::geometry::Surface;
use crate::mesh::Mesh;
use crate::trisk::Weights;

/// Prognostic fields.
#[derive(Clone)]
pub struct State {
    /// Fluid thickness on cells.
    pub h: Vec<f64>,
    /// Velocity normal to each edge, along `Mesh::edge_normal`.
    pub u: Vec<f64>,
}

impl State {
    pub fn zeros(mesh: &Mesh) -> State {
        State {
            h: vec![0.0; mesh.n_cells()],
            u: vec![0.0; mesh.n_edges()],
        }
    }

    fn axpy(&mut self, a: f64, d: &State) {
        for (x, y) in self.h.iter_mut().zip(&d.h) {
            *x += a * y;
        }
        for (x, y) in self.u.iter_mut().zip(&d.u) {
            *x += a * y;
        }
    }

    fn set_combination(&mut self, base: &State, a: f64, d: &State) {
        self.h.copy_from_slice(&base.h);
        self.u.copy_from_slice(&base.u);
        self.axpy(a, d);
    }
}

/// Conserved sums, for watching a run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Invariants {
    pub mass: f64,
    pub energy: f64,
    pub potential_enstrophy: f64,
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

    h_edge: Vec<f64>,
    flux: Vec<f64>,
    zeta: Vec<f64>,
    h_dual: Vec<f64>,
    q_dual: Vec<f64>,
    q_edge: Vec<f64>,
    kinetic: Vec<f64>,
    phi: Vec<f64>,
    grad: Vec<f64>,
}

/// Earth's rotation rate, rad/s.
pub const OMEGA: f64 = 7.292e-5;
/// Standard gravity used throughout the Williamson suite, m/s^2.
pub const GRAVITY: f64 = 9.806_16;

impl Model {
    pub fn new<S: Surface>(mesh: Mesh, surface: &S, g: f64, omega: f64) -> Model {
        let signs = Signs::new(&mesh);
        let weights = Weights::new(&mesh);
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
            h_edge: vec![0.0; ne],
            flux: vec![0.0; ne],
            zeta: vec![0.0; nv],
            h_dual: vec![0.0; nv],
            q_dual: vec![0.0; nv],
            q_edge: vec![0.0; ne],
            kinetic: vec![0.0; nc],
            phi: vec![0.0; nc],
            grad: vec![0.0; ne],
        }
    }

    /// Time step from the gravity-wave speed and the shortest edge.
    pub fn stable_step(&self, mean_depth: f64, courant: f64) -> f64 {
        courant * self.mesh.min_dc() / (self.g * mean_depth).sqrt()
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

    /// Right-hand side of the system at `s`.
    pub fn tendency(&mut self, s: &State, out: &mut State) {
        self.diagnose(s);
        let m = &self.mesh;

        dec::divergence(m, &self.signs, &self.flux, &mut out.h);
        out.h.iter_mut().for_each(|x| *x = -*x);

        dec::gradient(m, &self.phi, &mut self.grad);
        for e in 0..m.n_edges() {
            let (es, ws) = self.weights.stencil(e);
            let mut coriolis = 0.0;
            for (&eoe, &wt) in es.iter().zip(ws) {
                let eoe = eoe as usize;
                coriolis += wt * 0.5 * (self.q_edge[e] + self.q_edge[eoe]) * self.flux[eoe];
            }
            out.u[e] = coriolis - self.grad[e];
        }
    }

    /// One classical fourth-order Runge-Kutta step.
    pub fn step(&mut self, s: &mut State, dt: f64, work: &mut Work) {
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

    /// Potential vorticity at the duals, as of the last tendency or
    /// invariants call.
    pub fn potential_vorticity(&self) -> &[f64] {
        &self.q_dual
    }

    /// Advance `steps` steps.
    pub fn integrate(&mut self, s: &mut State, dt: f64, steps: usize, work: &mut Work) {
        for _ in 0..steps {
            self.step(s, dt, work);
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
        Invariants {
            mass,
            energy,
            potential_enstrophy: enstrophy,
        }
    }

    /// Velocity magnitude at each cell, reconstructed from the edge normals
    /// by the Perot formula. For output and for the Courant number only.
    pub fn cell_speed(&self, s: &State, out: &mut [f64]) {
        let m = &self.mesh;
        for (c, o) in out.iter_mut().enumerate() {
            let mut v = crate::vec3::Vec3::ZERO;
            for (k, &e) in m.edges_on_cell[c].iter().enumerate() {
                let sign = self.signs.out_of_cell[c][k];
                let arm = (m.edge_pos[e] - m.cell_pos[c]).norm();
                v += m.edge_normal[e] * (sign * sign * m.dv[e] * arm * s.u[e]);
            }
            *o = (v / m.area_cell[c]).norm();
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
    pub fn new(mesh: &Mesh) -> Work {
        Work {
            k1: State::zeros(mesh),
            k2: State::zeros(mesh),
            k3: State::zeros(mesh),
            k4: State::zeros(mesh),
            tmp: State::zeros(mesh),
        }
    }
}
