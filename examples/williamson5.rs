//! Williamson case 5, zonal flow over an isolated mountain, fifteen days.
//!
//! No closed form exists, so the run is judged on its invariants and on the
//! wave train the mountain sheds downstream. Output goes to a Zarr v3 store
//! under `out/williamson5.zarr`, one time slice every six hours.
//!
//! Run with `cargo run --release --example williamson5`.

use charney::Sphere;
use charney::mesh::Mesh;
use charney::swe::{Model, Work};
use charney::williamson::{MountainFlow, lonlat};
use charney::zarr::{Attribute, Store};

fn main() -> std::io::Result<()> {
    let surface = Sphere::EARTH;
    let level: u32 = 5;
    let days = 15.0;
    let output_every = 6.0 * 3600.0;

    let mesh = Mesh::scvt(level, 60, &surface);
    let case = MountainFlow::default();
    let (mut state, bottom) = case.state(&mesh, &surface);

    let coords: Vec<(f64, f64)> = (0..mesh.n_cells())
        .map(|c| lonlat(&surface, mesh.cell_pos[c]))
        .collect();
    let cells = mesh.n_cells();
    let duals = mesh.n_duals();

    let mut model = Model::new(mesh, &surface, case.g, case.omega);
    model.bottom = bottom.clone();
    let mut work = Work::for_state(&model.mesh, &state);

    let mean_depth = state.h.iter().sum::<f64>() / cells as f64;
    let per_slice = (output_every / model.stable_step(mean_depth, 0.4)).ceil() as usize;
    let dt = output_every / per_slice as f64;
    let slices = (days * 86_400.0 / output_every) as usize + 1;

    let store = Store::create(
        "out/williamson5.zarr",
        &[
            Attribute::text("title", "Williamson case 5, flow over an isolated mountain"),
            Attribute::text("surface", "sphere"),
            Attribute::number("level", level as f64),
            Attribute::text("integrator", "rk4"),
        ],
    )?;
    let lons: Vec<f64> = coords.iter().map(|c| c.0).collect();
    let lats: Vec<f64> = coords.iter().map(|c| c.1).collect();
    let times: Vec<f64> = (0..slices).map(|i| i as f64 * output_every).collect();
    store.coordinate("lon", "cell", "radians_east", &lons)?;
    store.coordinate("lat", "cell", "radians_north", &lats)?;
    store.coordinate("bottom", "cell", "m", &bottom)?;
    store.coordinate("time", "time", "seconds", &times)?;
    let mut h_out = store.array("h", cells, slices, "cell", "m", "fluid thickness")?;
    let mut speed_out = store.array("speed", cells, slices, "cell", "m s-1", "wind speed")?;
    let mut pv_out = store.array(
        "potential_vorticity",
        duals,
        slices,
        "dual",
        "m-1 s-1",
        "potential vorticity",
    )?;

    let start = model.invariants(&state);
    let mut speed = vec![0.0; cells];

    println!("Williamson case 5, {days} days, {cells} cells, dt {dt:.0} s");
    println!(
        "{:>6} {:>13} {:>13} {:>13}",
        "day", "mass", "energy", "pot. enstrophy"
    );

    for slice in 0..slices {
        model.cell_speed(&state, &mut speed);
        let inv = model.invariants(&state);
        h_out.push(&state.h)?;
        speed_out.push(&speed)?;
        pv_out.push(model.potential_vorticity())?;

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
        }
    }

    println!("\nwrote out/williamson5.zarr ({slices} slices)");
    Ok(())
}
