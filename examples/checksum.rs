//! A digest of a short run, for comparing builds.
//!
//! Chunking a loop must not change its answer, so the serial build and the
//! `parallel` build have to print the same digits:
//!
//! ```console
//! $ cargo run --release --example checksum
//! $ cargo run --release --features parallel --example checksum
//! ```

use charney::Sphere;
use charney::mesh::Mesh;
use charney::swe::{Model, Work};
use charney::williamson::MountainFlow;

fn main() {
    let surface = Sphere::EARTH;
    let mesh = Mesh::scvt(5, 60, &surface);
    let case = MountainFlow::default();
    let (mut state, bottom) = case.state(&mesh, &surface);
    let cells = mesh.n_cells();

    let mut model = Model::new(mesh, &surface, case.g, case.omega);
    model.bottom = bottom;
    let mean_depth = state.h.iter().sum::<f64>() / cells as f64;
    let dt = model.stable_step(mean_depth, 0.4);
    model.nu4 = model.nu4_stable(dt, 0.2);
    let mut work = Work::for_state(&model.mesh, &state);
    model.integrate(&mut state, dt, 200, &mut work);

    println!("threads {}", charney::parallel::threads());
    println!("h    {:.17e}", digest(&state.h));
    println!("u    {:.17e}", digest(&state.u));
    let invariants = model.invariants(&state);
    println!("mass {:.17e}", invariants.mass);
    println!("pe   {:.17e}", invariants.potential_enstrophy);
}

/// Order-sensitive sum, so a chunk boundary in the wrong place would show.
fn digest(v: &[f64]) -> f64 {
    v.iter()
        .enumerate()
        .map(|(i, x)| x * (i as f64 + 1.0).sqrt())
        .sum()
}
