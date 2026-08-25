//! Initial states from the standard test suite for the shallow-water
//! equations on a sphere, Williamson et al., J. Comput. Phys. 102 (1992).
//!
//! Case 1 advects a cosine bell once around the sphere under a prescribed
//! wind, which tests the transport operator on its own. Case 2 is a steady
//! zonal geostrophic flow, so the exact solution at any time is the initial
//! state and the error after five days measures the scheme. Case 5 adds an
//! isolated mountain and case 6 is the Rossby-Haurwitz wave; neither has a
//! closed form, and both are judged on their invariants and on whether the
//! structure survives.

use crate::geometry::Surface;
use crate::mesh::Mesh;
use crate::swe::{GRAVITY, OMEGA, State};
use crate::vec3::Vec3;
use std::f64::consts::PI;

/// Longitude and geodetic latitude of a point on a surface of revolution.
pub fn lonlat<S: Surface + ?Sized>(surface: &S, p: Vec3) -> (f64, f64) {
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

    pub fn state<S: Surface + ?Sized>(&self, mesh: &Mesh, surface: &S) -> State {
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
    pub fn state<S: Surface + ?Sized>(&self, mesh: &Mesh, surface: &S) -> (State, Vec<f64>) {
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

/// Phase of the `wavenumber` Fourier component of a cell field, over the
/// cells whose latitude lies between `band` in radians.
///
/// Following that phase over a run gives the speed the pattern travels at,
/// which is the number [`RossbyHaurwitz::pattern_speed`] predicts.
pub fn wave_phase(
    field: &[f64],
    coords: &[(f64, f64)],
    area: &[f64],
    wavenumber: f64,
    band: (f64, f64),
) -> f64 {
    let (mut cos, mut sin) = (0.0, 0.0);
    for (c, &(lon, lat)) in coords.iter().enumerate() {
        if lat < band.0 || lat > band.1 {
            continue;
        }
        cos += area[c] * field[c] * (wavenumber * lon).cos();
        sin += area[c] * field[c] * (wavenumber * lon).sin();
    }
    sin.atan2(cos)
}

/// Smallest signed difference between two angles.
pub fn angle_step(from: f64, to: f64) -> f64 {
    let mut d = to - from;
    while d > PI {
        d -= 2.0 * PI;
    }
    while d < -PI {
        d += 2.0 * PI;
    }
    d
}

/// Case 1: a cosine bell carried once around the sphere in twelve days by a
/// prescribed solid-body wind.
///
/// The flow is frozen, so what the run measures is the transport operator
/// alone. After one revolution the exact answer is the initial bell, and the
/// scheme's dispersion shows as a smeared peak and as under- and overshoot
/// outside the bell.
pub struct CosineBell {
    /// Advecting wind speed at the equator, m/s.
    pub u0: f64,
    /// Bell peak.
    pub peak: f64,
    /// Bell radius, radians.
    pub radius: f64,
    /// Bell centre, longitude and latitude in radians.
    pub center: (f64, f64),
    /// Uniform layer thickness the bell rides on, m.
    pub depth: f64,
}

impl CosineBell {
    /// The standard settings: one revolution in twelve days, a bell of radius
    /// `a / 3` centred on the equator.
    pub fn standard(radius: f64) -> CosineBell {
        CosineBell {
            u0: 2.0 * PI * radius / (12.0 * 86_400.0),
            peak: 1000.0,
            radius: radius / 3.0,
            center: (-PI / 2.0, 0.0),
            depth: 1000.0,
        }
    }

    /// Mixing ratio at a great-circle distance `arc` from the centre.
    pub fn mixing_ratio(&self, arc: f64) -> f64 {
        if arc >= self.radius {
            0.0
        } else {
            0.5 * self.peak * (1.0 + (PI * arc / self.radius).cos())
        }
    }

    /// Initial state, with the bell as the single tracer.
    pub fn state<S: Surface + ?Sized>(&self, mesh: &Mesh, surface: &S) -> State {
        let a = surface.authalic_radius();
        let center = {
            let (lon, lat) = self.center;
            Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin())
        };
        let mut s = State::with_tracers(mesh, 1);
        for c in 0..mesh.n_cells() {
            s.h[c] = self.depth;
            let unit = mesh.cell_pos[c].normalized();
            let arc = a * unit.dot(center).clamp(-1.0, 1.0).acos();
            s.tracer_mass[0][c] = self.depth * self.mixing_ratio(arc);
        }
        for e in 0..mesh.n_edges() {
            let v = Vec3::new(0.0, 0.0, self.u0 / a).cross(mesh.edge_site[e]);
            s.u[e] = v.dot(mesh.edge_normal[e]);
        }
        s
    }

    /// Time for one revolution.
    pub fn period(&self, radius: f64) -> f64 {
        2.0 * PI * radius / self.u0
    }
}

/// Case 6: the Rossby-Haurwitz wave of zonal wavenumber four.
///
/// A steady solution of the nondivergent barotropic equations, and only an
/// approximate one here, so it drifts and eventually breaks down. What a
/// scheme is judged on is how long the four-lobed structure survives with its
/// phase speed roughly right.
pub struct RossbyHaurwitz {
    /// Angular velocity of the wave pattern, 1/s.
    pub omega_wave: f64,
    /// Amplitude of the wave, 1/s.
    pub k: f64,
    /// Zonal wavenumber.
    pub wavenumber: f64,
    /// Mean layer thickness, m.
    pub h0: f64,
    pub omega: f64,
    pub g: f64,
}

impl Default for RossbyHaurwitz {
    fn default() -> Self {
        RossbyHaurwitz {
            omega_wave: 7.848e-6,
            k: 7.848e-6,
            wavenumber: 4.0,
            h0: 8000.0,
            omega: OMEGA,
            g: GRAVITY,
        }
    }
}

impl RossbyHaurwitz {
    /// Zonal and meridional wind at a point.
    pub fn wind(&self, a: f64, lon: f64, lat: f64) -> (f64, f64) {
        let (r, w, k) = (self.wavenumber, self.omega_wave, self.k);
        let (c, s) = (lat.cos(), lat.sin());
        let zonal = a * w * c + a * k * c.powf(r - 1.0) * (r * s * s - c * c) * (r * lon).cos();
        let meridional = -a * k * r * c.powf(r - 1.0) * s * (r * lon).sin();
        (zonal, meridional)
    }

    /// Thickness at a point.
    pub fn thickness(&self, a: f64, lon: f64, lat: f64) -> f64 {
        let (r, w, k) = (self.wavenumber, self.omega_wave, self.k);
        let c = lat.cos();
        let big_a = 0.5 * w * (2.0 * self.omega + w) * c * c
            + 0.25
                * k
                * k
                * c.powf(2.0 * r)
                * ((r + 1.0) * c * c + (2.0 * r * r - r - 2.0) - 2.0 * r * r / (c * c));
        let big_b = 2.0 * (self.omega + w) * k / ((r + 1.0) * (r + 2.0))
            * c.powf(r)
            * ((r * r + 2.0 * r + 2.0) - (r + 1.0) * (r + 1.0) * c * c);
        let big_c = 0.25 * k * k * c.powf(2.0 * r) * ((r + 1.0) * c * c - (r + 2.0));
        self.h0 + a * a / self.g * (big_a + big_b * (r * lon).cos() + big_c * (2.0 * r * lon).cos())
    }

    /// Angular speed at which the pattern travels east, from the
    /// nondivergent theory. Shallow water is divergent, so a solver reads a
    /// few per cent below this.
    pub fn pattern_speed(&self) -> f64 {
        let r = self.wavenumber;
        (r * (3.0 + r) * self.omega_wave - 2.0 * self.omega) / ((1.0 + r) * (2.0 + r))
    }

    pub fn state<S: Surface + ?Sized>(&self, mesh: &Mesh, surface: &S) -> State {
        let a = surface.authalic_radius();
        let mut s = State::zeros(mesh);
        for c in 0..mesh.n_cells() {
            let (lon, lat) = lonlat(surface, mesh.cell_pos[c]);
            s.h[c] = self.thickness(a, lon, lat);
        }
        for e in 0..mesh.n_edges() {
            let p = mesh.edge_site[e];
            let (lon, lat) = lonlat(surface, p);
            let (zonal, meridional) = self.wind(a, lon, lat);
            let (east, north) = surface.wind_basis(p);
            s.u[e] = (east * zonal + north * meridional).dot(mesh.edge_normal[e]);
        }
        s
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
