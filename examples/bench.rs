//! Timings for the mesh build and the time step.
//!
//! Run with `cargo run --release --example bench`.

use std::time::Instant;

use charney::Sphere;
use charney::mesh::Mesh;
use charney::swe::{GRAVITY, Model, OMEGA, State, Work};
use charney::williamson::SteadyZonalFlow;

fn main() {
    let surface = Sphere::EARTH;
    println!("threads: {}", charney::parallel::threads());
    println!(
        "{:>5} {:>8} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "level", "cells", "scvt (s)", "model (s)", "step (ms)", "cell-steps/s", "day (s)"
    );
    for level in [4u32, 5, 6, 7] {
        let t = Instant::now();
        let mesh = Mesh::scvt(level, 60, &surface);
        let scvt = t.elapsed().as_secs_f64();

        let cells = mesh.n_cells();
        let case = SteadyZonalFlow::standard(surface.radius);
        let state = case.state(&mesh, &surface);
        let mean_depth = case.mean_thickness(surface.radius);

        let t = Instant::now();
        let mut model = Model::new(mesh, &surface, GRAVITY, OMEGA);
        let build = t.elapsed().as_secs_f64();

        let mut work = Work::for_state(&model.mesh, &state);
        let dt = model.stable_step(mean_depth, 0.4);

        let mut s = state.clone();
        model.integrate(&mut s, dt, 5, &mut work);
        let mut s = state.clone();
        let n = 40;
        let t = Instant::now();
        model.integrate(&mut s, dt, n, &mut work);
        let per_step = t.elapsed().as_secs_f64() / n as f64;

        println!(
            "{level:>5} {cells:>8} {scvt:>10.2} {build:>10.2} {:>10.2} {:>12.3e} {:>10.2}",
            per_step * 1e3,
            cells as f64 / per_step,
            per_step * 86_400.0 / dt,
        );
        let _ = State::zeros(&model.mesh);
    }
}
