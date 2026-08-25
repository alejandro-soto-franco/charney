//! Initial states from the standard test suite for the shallow-water
//! equations on a sphere, Williamson et al., J. Comput. Phys. 102 (1992).
//!
//! Case 2 is a steady zonal geostrophic flow, so the exact solution at any
//! time is the initial state and the error after five days measures the
//! scheme. Case 5 adds an isolated mountain and has no closed form; it is run
//! for the conservation record and against a reference solution.

use crate::geometry::Surface;
use crate::mesh::Mesh;
use crate::swe::{GRAVITY, OMEGA, State};
use crate::vec3::Vec3;

/// Longitude and geodetic latitude of a point on a surface of revolution.
pub fn lonlat<S: Surface>(surface: &S, p: Vec3) -> (f64, f64) {
    let lon = p.y.atan2(p.x);
    let lat = surface.sin_geodetic_latitude(p).asin();
    (lon, lat)
}

/// Case 2: solid-body zonal flow in geostrophic balance, flowing along the
/// equator.
pub struct SteadyZonalFlow {
    /// Peak zonal wind, `2 pi a / 12 days`.
    pub u0: f64,
    /// Geopotential at the equator.
    pub gh0: f64,
    pub omega: f64,
    pub g: f64,
}

impl SteadyZonalFlow {
    pub fn standard(radius: f64) -> SteadyZonalFlow {
        let twelve_days = 12.0 * 86_400.0;
        SteadyZonalFlow {
            u0: 2.0 * std::f64::consts::PI * radius / twelve_days,
            gh0: 2.94e4,
            omega: OMEGA,
            g: GRAVITY,
        }
    }

    /// Velocity at a point: rotation about the polar axis at rate `u0 / a`.
    pub fn velocity(&self, radius: f64, p: Vec3) -> Vec3 {
        Vec3::new(0.0, 0.0, self.u0 / radius).cross(p)
    }

    /// Thickness at a point.
    pub fn thickness(&self, radius: f64, sin_lat: f64) -> f64 {
        let balance = radius * self.omega * self.u0 + 0.5 * self.u0 * self.u0;
        (self.gh0 - balance * sin_lat * sin_lat) / self.g
    }

    pub fn state<S: Surface>(&self, mesh: &Mesh, surface: &S) -> State {
        let a = surface.authalic_radius();
        let mut s = State::zeros(mesh);
        for c in 0..mesh.n_cells() {
            let sin_lat = surface.sin_geodetic_latitude(mesh.cell_pos[c]);
            s.h[c] = self.thickness(a, sin_lat);
        }
        for e in 0..mesh.n_edges() {
            s.u[e] = self.velocity(a, mesh.edge_site[e]).dot(mesh.edge_normal[e]);
        }
        s
    }

    /// Mean thickness, for the time step and for relative error norms.
    pub fn mean_thickness(&self, radius: f64) -> f64 {
        let balance = radius * self.omega * self.u0 + 0.5 * self.u0 * self.u0;
        (self.gh0 - balance / 3.0) / self.g
    }
}

/// Case 5: zonal flow over an isolated conical mountain.
pub struct MountainFlow {
    pub u0: f64,
    /// Free-surface height at the equator, m.
    pub h0: f64,
    /// Mountain peak height, m.
    pub peak: f64,
    /// Mountain base radius, radians.
    pub radius: f64,
    /// Mountain centre, longitude and latitude in radians.
    pub center: (f64, f64),
    pub omega: f64,
    pub g: f64,
}

impl Default for MountainFlow {
    fn default() -> Self {
        MountainFlow {
            u0: 20.0,
            h0: 5960.0,
            peak: 2000.0,
            radius: std::f64::consts::PI / 9.0,
            center: (-std::f64::consts::PI / 2.0, std::f64::consts::PI / 6.0),
            omega: OMEGA,
            g: GRAVITY,
        }
    }
}

impl MountainFlow {
    /// Mountain height at a point.
    pub fn bottom(&self, lon: f64, lat: f64) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut dlon = (lon - self.center.0).rem_euclid(two_pi);
        if dlon > std::f64::consts::PI {
            dlon -= two_pi;
        }
        let dlat = lat - self.center.1;
        let r = (dlon * dlon + dlat * dlat).sqrt().min(self.radius);
        self.peak * (1.0 - r / self.radius)
    }

    /// Initial state, and the topography it sits on.
    pub fn state<S: Surface>(&self, mesh: &Mesh, surface: &S) -> (State, Vec<f64>) {
        let a = surface.authalic_radius();
        let balance = a * self.omega * self.u0 + 0.5 * self.u0 * self.u0;
        let mut s = State::zeros(mesh);
        let mut bottom = vec![0.0; mesh.n_cells()];
        for (c, b) in bottom.iter_mut().enumerate() {
            let (lon, lat) = lonlat(surface, mesh.cell_pos[c]);
            *b = self.bottom(lon, lat);
            let free_surface = self.h0 - balance * lat.sin() * lat.sin() / self.g;
            s.h[c] = free_surface - *b;
        }
        for e in 0..mesh.n_edges() {
            let v = Vec3::new(0.0, 0.0, self.u0 / a).cross(mesh.edge_site[e]);
            s.u[e] = v.dot(mesh.edge_normal[e]);
        }
        (s, bottom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;
    use crate::swe::Model;

    /// The balance the case-2 state is built on must survive discretisation:
    /// the two large terms in the momentum equation cancel to a small
    /// residual, and the height tendency nearly vanishes. A sign error
    /// anywhere in the Coriolis path doubles the residual instead.
    #[test]
    fn case_two_starts_close_to_balance() {
        let surface = Sphere::EARTH;
        let mesh = Mesh::scvt(4, 60, &surface);
        let case = SteadyZonalFlow::standard(surface.radius);
        let state = case.state(&mesh, &surface);
        let mut model = Model::new(mesh, &surface, case.g, case.omega);

        let mut tend = State::zeros(&model.mesh);
        model.tendency(&state, &mut tend);

        // Compare the residual against the size of the pressure gradient it
        // is supposed to cancel.
        let scale = {
            let mut grad = vec![0.0; model.mesh.n_edges()];
            let phi: Vec<f64> = (0..model.mesh.n_cells())
                .map(|c| case.g * state.h[c])
                .collect();
            crate::dec::gradient(&model.mesh, &phi, &mut grad);
            rms(&grad)
        };
        let residual = rms(&tend.u);
        assert!(
            residual / scale < 0.01,
            "momentum residual {residual} against {scale}"
        );

        let dh = rms(&tend.h);
        let h_scale = rms(&state.h);
        assert!(
            dh / h_scale < 1e-8,
            "height tendency {dh} against {h_scale}"
        );
    }

    fn rms(v: &[f64]) -> f64 {
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
    }
}
