//! Williamson case 1: a cosine bell carried once around the sphere.
//!
//! The wind is prescribed and frozen, so the run measures the transport
//! operator on its own. After twelve days the bell has come back to where it
//! started and the exact answer is the initial field.
//!
//! Three flux reconstructions are compared. The centred average carries no
//! dissipation and rings; the upwind extrapolation damps the ringing; the
//! limited form removes the undershoot outright and pays for it by clipping
//! the peak, which is what Barth-Jespersen does at a smooth extremum.
//!
//! Run with `cargo run --release --example williamson1`.

use charney::Sphere;
use charney::mesh::Mesh;
use charney::swe::{GRAVITY, Model, OMEGA, Work};
use charney::transport::TracerFlux;
use charney::williamson::CosineBell;

fn main() {
    let surface = Sphere::EARTH;
    println!("Williamson case 1, one revolution in 12 days, Courant 0.4 on the wind");
    println!(
        "{:>14} {:>5} {:>7} {:>7} {:>11} {:>6} {:>9} {:>9} {:>10}",
        "flux", "level", "cells", "steps", "l2(q)", "order", "min", "max", "mass"
    );

    for scheme in [
        TracerFlux::Centered,
        TracerFlux::Upwind,
        TracerFlux::LimitedUpwind,
    ] {
        run(scheme, &surface);
    }
}

fn run(scheme: TracerFlux, surface: &Sphere) {
    let mut previous: Option<f64> = None;
    for level in [3u32, 4, 5] {
        let mesh = Mesh::scvt(level, 60, surface);
        let case = CosineBell::standard(surface.radius);
        let initial = case.state(&mesh, surface);

        let cells = mesh.n_cells();
        let dx = mesh.min_dc();
        let mut state = initial.clone();
        let mut model = Model::new(mesh, surface, GRAVITY, OMEGA);
        model.frozen_flow = true;
        model.tracer_flux_scheme = scheme;
        let mut work = Work::for_state(&model.mesh, &state);

        let period = case.period(surface.radius);
        let steps = (period / (0.4 * dx / case.u0)).ceil() as usize;
        let before = model.invariants(&state);
        model.integrate(&mut state, period / steps as f64, steps, &mut work);
        let after = model.invariants(&state);

        let m = &model.mesh;
        let mut q = vec![0.0; cells];
        let mut q0 = vec![0.0; cells];
        model.mixing_ratio(&state, 0, &mut q);
        model.mixing_ratio(&initial, 0, &mut q0);

        let mut num = 0.0;
        let mut den = 0.0;
        for c in 0..cells {
            let d = q[c] - q0[c];
            num += m.area_cell[c] * d * d;
            den += m.area_cell[c] * q0[c] * q0[c];
        }
        let l2 = (num / den).sqrt();
        let order = previous.map(|p| (p / l2).log2());
        previous = Some(l2);

        println!(
            "{:>14} {level:>5} {cells:>7} {steps:>7} {l2:>11.3e} {:>6} {:>9.3} {:>9.3} {:>10.2e}",
            if level == 3 {
                format!("{scheme:?}")
            } else {
                String::new()
            },
            order
                .map(|o| format!("{o:.2}"))
                .unwrap_or_else(|| "-".into()),
            q.iter().copied().fold(f64::INFINITY, f64::min) / case.peak,
            q.iter().copied().fold(f64::NEG_INFINITY, f64::max) / case.peak,
            (after.tracer_mass[0] - before.tracer_mass[0]).abs() / before.tracer_mass[0],
        );
    }
}
