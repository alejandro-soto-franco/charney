//! The same case on a sphere and on WGS 84.
//!
//! Williamson case 2 is an exact steady solution of the shallow-water
//! equations on a sphere. On an oblate spheroid it is not: the Coriolis
//! parameter follows the geodetic latitude, which leads the geocentric one by
//! up to 0.19 degrees, and the metric is no longer a sphere's. The state is
//! out of balance from the start and adjusts.
//!
//! Grid, solver, initial data, gravity and rotation rate are shared, and the
//! sphere is given the spheroid's authalic radius, so the two surfaces have
//! equal area and only their shape differs.
//!
//! One resolution settles nothing, since on a coarse mesh the sphere's own
//! drift is numerical error of the same size. Refining separates them: the
//! sphere's drift falls with the mesh and the spheroid's does not, so the
//! ratio grows.
//!
//! Run with `cargo run --release --example spheroid`.

use charney::mesh::Mesh;
use charney::swe::{Model, State, Work};
use charney::williamson::SteadyZonalFlow;
use charney::{Sphere, Surface, Wgs84};

const DAYS: f64 = 3.0;

fn main() {
    let spheroid = Wgs84::WGS84;
    let sphere = Sphere {
        radius: spheroid.authalic_radius(),
    };

    println!("Williamson case 2 after {DAYS} days, equal-area sphere against WGS 84");
    println!(
        "flattening {:.6e}, shared authalic radius {:.1} m\n",
        spheroid.flattening(),
        sphere.radius
    );
    println!(
        "{:>5} {:>7} {:>12} {:>12} {:>7}",
        "level", "cells", "sphere", "wgs84", "ratio"
    );

    for level in [3u32, 4, 5] {
        let a = drift(level, &sphere);
        let b = drift(level, &spheroid);
        println!(
            "{level:>5} {:>7} {:>12.3e} {:>12.3e} {:>7.2}",
            10 * 4usize.pow(level) + 2,
            a,
            b,
            b / a
        );
    }

    println!("\nThe sphere column is discretisation error and converges away. The wgs84");
    println!("column has that plus the adjustment the ellipticity forces, which does not,");
    println!("so the ratio grows with resolution.");
}

/// Root-mean-square height change from the initial state, in metres.
fn drift<S: Surface>(level: u32, surface: &S) -> f64 {
    let mesh = Mesh::scvt(level, 60, surface);
    let case = SteadyZonalFlow::standard(surface.authalic_radius());
    let initial = case.state(&mesh, surface);
    let mean_depth = case.mean_thickness(surface.authalic_radius());

    let mut state = initial.clone();
    let mut model = Model::new(mesh, surface, case.g, case.omega);
    let mut work = Work::for_state(&model.mesh, &state);
    let steps = (DAYS * 86_400.0 / model.stable_step(mean_depth, 0.4)).ceil() as usize;
    model.integrate(&mut state, DAYS * 86_400.0 / steps as f64, steps, &mut work);

    rms_difference(&state, &initial)
}

fn rms_difference(a: &State, b: &State) -> f64 {
    let n = a.h.len() as f64;
    (a.h.iter()
        .zip(&b.h)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        / n)
        .sqrt()
}
