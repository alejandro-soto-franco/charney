//! Williamson case 2, the steady zonal geostrophic flow, run for five days at
//! three resolutions.
//!
//! The exact solution is the initial state, so the height error after five
//! days is the scheme's own. Run with `cargo run --release --example williamson2`.

use charney::Sphere;
use charney::mesh::Mesh;
use charney::swe::{Model, State, Work};
use charney::williamson::SteadyZonalFlow;

fn main() {
    let surface = Sphere::EARTH;
    let days = 5.0;
    println!("Williamson case 2, {days} days, RK4, Courant 0.4");
    println!(
        "{:>5} {:>7} {:>9} {:>7} {:>11} {:>11} {:>6} {:>11} {:>11}",
        "level", "cells", "dx (km)", "steps", "l2(h)", "linf(h)", "order", "mass", "energy"
    );

    let mut previous: Option<f64> = None;
    for level in [3u32, 4, 5] {
        let mesh = Mesh::scvt(level, 60, &surface);
        let case = SteadyZonalFlow::standard(surface.radius);
        let exact = case.state(&mesh, &surface);
        let mean_depth = case.mean_thickness(surface.radius);

        let mut state = exact.clone();
        let dx = mesh.min_dc();
        let cells = mesh.n_cells();
        let mut model = Model::new(mesh, &surface, case.g, case.omega);
        let mut work = Work::new(&model.mesh);

        let steps = (days * 86_400.0 / model.stable_step(mean_depth, 0.4)).ceil() as usize;
        let dt = days * 86_400.0 / steps as f64;

        let before = model.invariants(&state);
        model.integrate(&mut state, dt, steps, &mut work);
        let after = model.invariants(&state);

        let (l2, linf) = height_error(&model, &state, &exact);
        let order = previous.map(|p| (p / l2).log2());
        previous = Some(l2);

        println!(
            "{level:>5} {cells:>7} {:>9.1} {steps:>7} {l2:>11.3e} {linf:>11.3e} {:>6} {:>11.2e} {:>11.2e}",
            dx / 1000.0,
            order
                .map(|o| format!("{o:.2}"))
                .unwrap_or_else(|| "-".into()),
            (after.mass - before.mass).abs() / before.mass,
            (after.energy - before.energy) / before.energy,
        );
    }
}

fn height_error(model: &Model, got: &State, exact: &State) -> (f64, f64) {
    let m = &model.mesh;
    let mut num = 0.0;
    let mut den = 0.0;
    let mut worst: f64 = 0.0;
    let mut scale: f64 = 0.0;
    for c in 0..m.n_cells() {
        let d = got.h[c] - exact.h[c];
        num += m.area_cell[c] * d * d;
        den += m.area_cell[c] * exact.h[c] * exact.h[c];
        worst = worst.max(d.abs());
        scale = scale.max(exact.h[c].abs());
    }
    ((num / den).sqrt(), worst / scale)
}
