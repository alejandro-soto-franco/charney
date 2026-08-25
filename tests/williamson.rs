//! End-to-end runs of the standard test cases.

use charney::mesh::Mesh;
use charney::swe::{GRAVITY, Model, OMEGA, State, Work};
use charney::williamson::{
    CosineBell, MountainFlow, RossbyHaurwitz, SteadyZonalFlow, angle_step, lonlat, wave_phase,
};
use charney::{Sphere, Surface, Wgs84};

struct Outcome {
    /// Relative l2 height error against the exact solution.
    l2: f64,
    /// Root-mean-square height change from the initial state, in metres.
    drift: f64,
    mass_drift: f64,
    energy_drift: f64,
}

fn steady_zonal<S: Surface>(level: u32, days: f64, surface: &S) -> Outcome {
    let mesh = Mesh::scvt(level, 60, surface);
    let case = SteadyZonalFlow::standard(surface.authalic_radius());
    let exact = case.state(&mesh, surface);
    let mean_depth = case.mean_thickness(surface.authalic_radius());

    let mut state = exact.clone();
    let mut model = Model::new(mesh, surface, case.g, case.omega);
    let mut work = Work::for_state(&model.mesh, &state);
    let steps = (days * 86_400.0 / model.stable_step(mean_depth, 0.4)).ceil() as usize;
    let dt = days * 86_400.0 / steps as f64;

    let before = model.invariants(&state);
    model.integrate(&mut state, dt, steps, &mut work);
    let after = model.invariants(&state);

    let n = state.h.len() as f64;
    let drift = (state
        .h
        .iter()
        .zip(&exact.h)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        / n)
        .sqrt();

    Outcome {
        l2: relative_l2(&model, &state, &exact),
        drift,
        mass_drift: (after.mass - before.mass).abs() / before.mass,
        energy_drift: (after.energy - before.energy).abs() / before.energy,
    }
}

fn relative_l2(model: &Model, got: &State, reference: &State) -> f64 {
    let m = &model.mesh;
    let mut num = 0.0;
    let mut den = 0.0;
    for c in 0..m.n_cells() {
        let d = got.h[c] - reference.h[c];
        num += m.area_cell[c] * d * d;
        den += m.area_cell[c] * reference.h[c] * reference.h[c];
    }
    (num / den).sqrt()
}

/// Case 2 is steady, so halving the mesh spacing should quarter the height
/// error. `examples/williamson2` runs the same study to five days at three
/// resolutions and reports orders of 1.93 and 1.82.
#[test]
fn case_two_converges_at_second_order() {
    let surface = Sphere::EARTH;
    let coarse = steady_zonal(2, 2.0, &surface);
    let fine = steady_zonal(3, 2.0, &surface);
    let order = (coarse.l2 / fine.l2).log2();
    assert!(
        order > 1.7,
        "convergence order {order} from {} to {}",
        coarse.l2,
        fine.l2
    );
    assert!(fine.l2 < 5e-3, "level 3 height error {}", fine.l2);
}

/// The divergence is a telescoping sum of edge fluxes, so total mass cannot
/// drift beyond rounding however long the run.
#[test]
fn mass_is_conserved_to_rounding() {
    let out = steady_zonal(3, 2.0, &Sphere::EARTH);
    assert!(out.mass_drift < 1e-12, "mass drift {}", out.mass_drift);
    assert!(out.energy_drift < 1e-6, "energy drift {}", out.energy_drift);
}

/// The mountain case has no closed form. What it must not do is leak mass or
/// grow an energy trend.
#[test]
fn case_five_holds_its_invariants() {
    let surface = Sphere::EARTH;
    let mesh = Mesh::scvt(3, 60, &surface);
    let case = MountainFlow::default();
    let (mut state, bottom) = case.state(&mesh, &surface);
    let mean_depth = state.h.iter().sum::<f64>() / mesh.n_cells() as f64;

    let mut model = Model::new(mesh, &surface, case.g, case.omega);
    model.bottom = bottom;
    let mut work = Work::for_state(&model.mesh, &state);

    let days = 5.0;
    let steps = (days * 86_400.0 / model.stable_step(mean_depth, 0.4)).ceil() as usize;
    let dt = days * 86_400.0 / steps as f64;

    let before = model.invariants(&state);
    model.integrate(&mut state, dt, steps, &mut work);
    let after = model.invariants(&state);

    assert!(
        (after.mass - before.mass).abs() / before.mass < 1e-12,
        "mass drift {}",
        (after.mass - before.mass).abs() / before.mass
    );
    assert!(
        (after.energy - before.energy).abs() / before.energy < 1e-5,
        "energy drift {}",
        (after.energy - before.energy).abs() / before.energy
    );
    assert!(
        (after.potential_enstrophy - before.potential_enstrophy).abs() / before.potential_enstrophy
            < 1e-2,
        "potential enstrophy drift {}",
        (after.potential_enstrophy - before.potential_enstrophy).abs() / before.potential_enstrophy
    );
    assert!(state.h.iter().all(|&h| h > 0.0), "the layer went dry");
}

/// The state that stands still on a sphere does not stand still on WGS 84.
/// Grid, solver and initial data are shared, and the sphere is given the
/// spheroid's authalic radius, so only the shape of the surface differs.
///
/// One resolution proves nothing here: on a coarse mesh the sphere's own
/// drift is numerical error of the same size as the spheroid's adjustment.
/// Refining separates them, because only one of the two converges away.
/// `examples/spheroid` prints the table.
#[test]
fn refining_separates_the_spheroid_from_the_sphere() {
    let spheroid = Wgs84::WGS84;
    let equal_area = Sphere {
        radius: spheroid.authalic_radius(),
    };
    let ratio = |level: u32| {
        let days = 3.0;
        steady_zonal(level, days, &spheroid).drift / steady_zonal(level, days, &equal_area).drift
    };
    let coarse = ratio(3);
    let fine = ratio(4);
    assert!(
        fine > coarse,
        "the gap closed instead of opening: {coarse} then {fine}"
    );
    assert!(
        fine > 1.2,
        "spheroid drifts only {fine} times as far at level 4"
    );
}

/// Case 1 with the limited upwind flux: the tracer total must not move, and
/// the reconstruction must not manufacture a negative mixing ratio anywhere.
/// `examples/williamson1` runs the same case for all three flux forms and
/// prints what each costs.
#[test]
fn case_one_conserves_the_tracer_and_keeps_it_positive() {
    let surface = Sphere::EARTH;
    let mesh = Mesh::scvt(3, 60, &surface);
    let case = CosineBell::standard(surface.radius);
    let mut state = case.state(&mesh, &surface);

    let cells = mesh.n_cells();
    let dx = mesh.min_dc();
    let mut model = Model::new(mesh, &surface, GRAVITY, OMEGA);
    model.frozen_flow = true;
    let mut work = Work::for_state(&model.mesh, &state);

    let period = case.period(surface.radius);
    let steps = (period / (0.4 * dx / case.u0)).ceil() as usize;
    let before = model.invariants(&state);
    model.integrate(&mut state, period / steps as f64, steps, &mut work);
    let after = model.invariants(&state);

    let drift = (after.tracer_mass[0] - before.tracer_mass[0]).abs() / before.tracer_mass[0];
    assert!(drift < 1e-12, "tracer mass drift {drift}");

    let mut q = vec![0.0; cells];
    model.mixing_ratio(&state, 0, &mut q);
    let lowest = q.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(lowest > -1e-9 * case.peak, "mixing ratio went to {lowest}");

    // The flow was frozen, so nothing else may have moved.
    assert!((after.mass - before.mass).abs() / before.mass < 1e-14);
}

/// Case 6 must travel at close to the speed the nondivergent theory gives.
/// Shallow water is divergent, so the solver reads a few per cent low; a
/// staggering or Coriolis error would put it somewhere else entirely.
#[test]
fn case_six_travels_at_close_to_the_predicted_speed() {
    let surface = Sphere::EARTH;
    let days = 4usize;
    let mesh = Mesh::scvt(3, 60, &surface);
    let case = RossbyHaurwitz::default();
    let mut state = case.state(&mesh, &surface);
    let coords: Vec<(f64, f64)> = (0..mesh.n_cells())
        .map(|c| lonlat(&surface, mesh.cell_pos[c]))
        .collect();

    let mean_depth = state.h.iter().sum::<f64>() / mesh.n_cells() as f64;
    let mut model = Model::new(mesh, &surface, case.g, case.omega);
    let mut work = Work::for_state(&model.mesh, &state);
    let per_day = (86_400.0 / model.stable_step(mean_depth, 0.4)).ceil() as usize;
    let dt = 86_400.0 / per_day as f64;
    model.nu4 = model.nu4_stable(dt, 0.2);

    let area = model.mesh.area_cell.clone();
    let band = (20f64.to_radians(), 70f64.to_radians());
    let mut phase = wave_phase(&state.h, &coords, &area, case.wavenumber, band);
    let mut travelled = 0.0;
    for _ in 0..days {
        model.integrate(&mut state, dt, per_day, &mut work);
        let now = wave_phase(&state.h, &coords, &area, case.wavenumber, band);
        travelled += angle_step(phase, now);
        phase = now;
    }

    let measured = travelled / case.wavenumber / (days as f64 * 86_400.0);
    let predicted = case.pattern_speed();
    let short = 1.0 - measured / predicted;
    assert!(
        (0.0..0.25).contains(&short),
        "pattern speed {} deg/day against {} deg/day",
        measured.to_degrees() * 86_400.0,
        predicted.to_degrees() * 86_400.0
    );
    assert!(
        state.h.iter().all(|&h| h.is_finite() && h > 0.0),
        "the run came apart"
    );
}
