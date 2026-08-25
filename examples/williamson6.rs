//! Williamson case 6: the Rossby-Haurwitz wave of zonal wavenumber four.
//!
//! An exact steady solution of the nondivergent barotropic equations and only
//! an approximate one for shallow water, so it drifts westward, slowly loses
//! symmetry and eventually breaks down. A scheme is judged on how long the
//! four lobes survive and on what its invariants do meanwhile.
//!
//! Output goes to a Zarr v3 store under `out/williamson6.zarr`.
//!
//! Run with `cargo run --release --example williamson6`.

use charney::Sphere;
use charney::mesh::Mesh;
use charney::swe::{Model, Work};
use charney::williamson::{RossbyHaurwitz, angle_step, lonlat, wave_phase};
use charney::zarr::{Attribute, Store};

fn main() -> std::io::Result<()> {
    let surface = Sphere::EARTH;
    let level = 5;
    let days = 14.0;
    let output_every = 12.0 * 3600.0;

    let mesh = Mesh::scvt(level, 60, &surface);
    let case = RossbyHaurwitz::default();
    let mut state = case.state(&mesh, &surface);

    let coords: Vec<(f64, f64)> = (0..mesh.n_cells())
        .map(|c| lonlat(&surface, mesh.cell_pos[c]))
        .collect();
    let cells = mesh.n_cells();
    let duals = mesh.n_duals();

    let mut model = Model::new(mesh, &surface, case.g, case.omega);
    let mut work = Work::for_state(&model.mesh, &state);

    let mean_depth = state.h.iter().sum::<f64>() / cells as f64;
    let per_slice = (output_every / model.stable_step(mean_depth, 0.4)).ceil() as usize;
    let dt = output_every / per_slice as f64;
    // A fortnight is long enough for the grid imprint at the pentagons to
    // show, so damp the grid scale as hard as the time step allows.
    model.nu4 = model.nu4_stable(dt, 0.2);
    let slices = (days * 86_400.0 / output_every) as usize + 1;

    let store = Store::create(
        "out/williamson6.zarr",
        &[
            Attribute::text("title", "Williamson case 6, Rossby-Haurwitz wave"),
            Attribute::text("surface", "sphere"),
            Attribute::number("level", level as f64),
            Attribute::text("integrator", "rk4"),
            Attribute::number("nu4", model.nu4),
        ],
    )?;
    store.coordinate(
        "lon",
        "cell",
        "radians_east",
        &coords.iter().map(|c| c.0).collect::<Vec<_>>(),
    )?;
    store.coordinate(
        "lat",
        "cell",
        "radians_north",
        &coords.iter().map(|c| c.1).collect::<Vec<_>>(),
    )?;
    store.coordinate(
        "time",
        "time",
        "seconds",
        &(0..slices)
            .map(|i| i as f64 * output_every)
            .collect::<Vec<_>>(),
    )?;
    let mut h_out = store.array("h", cells, slices, "cell", "m", "fluid thickness")?;
    let mut speed_out = store.array("speed", cells, slices, "cell", "m s-1", "wind speed")?;
    let mut vort_out = store.array(
        "vorticity",
        duals,
        slices,
        "dual",
        "s-1",
        "relative vorticity",
    )?;

    let start = model.invariants(&state);
    let mut speed = vec![0.0; cells];
    let band = (20f64.to_radians(), 70f64.to_radians());
    let area = model.mesh.area_cell.clone();
    let mut phase = wave_phase(&state.h, &coords, &area, case.wavenumber, band);
    let mut travelled = 0.0;

    println!("Williamson case 6, {days} days, {cells} cells, dt {dt:.0} s");
    println!(
        "{:>6} {:>13} {:>13} {:>13}",
        "day", "mass", "energy", "pot. enstrophy"
    );

    for slice in 0..slices {
        model.cell_speed(&state, &mut speed);
        let inv = model.invariants(&state);
        h_out.push(&state.h)?;
        speed_out.push(&speed)?;
        vort_out.push(model.relative_vorticity())?;

        if slice % 4 == 0 {
            println!(
                "{:>6.1} {:>13.2e} {:>13.2e} {:>13.2e}",
                slice as f64 * output_every / 86_400.0,
                (inv.mass - start.mass) / start.mass,
                (inv.energy - start.energy) / start.energy,
                (inv.potential_enstrophy - start.potential_enstrophy) / start.potential_enstrophy,
            );
        }
        if slice + 1 < slices {
            model.integrate(&mut state, dt, per_slice, &mut work);
            let now = wave_phase(&state.h, &coords, &area, case.wavenumber, band);
            travelled += angle_step(phase, now);
            phase = now;
        }
    }

    let elapsed = (slices - 1) as f64 * output_every;
    let measured = travelled / case.wavenumber / elapsed;
    let predicted = case.pattern_speed();
    println!("\npattern speed over {days} days, from the phase of the height field:");
    println!(
        "  measured {:.2} deg/day, nondivergent theory {:.2} deg/day, {:.1}% below",
        measured.to_degrees() * 86_400.0,
        predicted.to_degrees() * 86_400.0,
        100.0 * (1.0 - measured / predicted)
    );
    println!("wrote out/williamson6.zarr ({slices} slices)");
    Ok(())
}
