//! Command-line driver: run one standard case and write a Zarr v3 store.
//!
//! ```text
//! charney <case> [--level N] [--days D] [--every HOURS] [--surface sphere|wgs84]
//!                [--damping DAYS] [--out DIR]
//!
//! cases: case1  cosine bell advected once around
//!        case2  steady zonal geostrophic flow
//!        case5  zonal flow over an isolated mountain
//!        case6  Rossby-Haurwitz wave
//! ```

use std::process::ExitCode;

use charney::mesh::Mesh;
use charney::swe::{GRAVITY, Model, OMEGA, State, Work};
use charney::williamson::{CosineBell, MountainFlow, RossbyHaurwitz, SteadyZonalFlow, lonlat};
use charney::zarr::{Attribute, Store};
use charney::{Sphere, Surface, Wgs84};

const USAGE: &str = "\
charney <case> [options]

cases
  case1   cosine bell advected once around the sphere
  case2   steady zonal geostrophic flow
  case5   zonal flow over an isolated mountain
  case6   Rossby-Haurwitz wave, zonal wavenumber four

options
  --level N        bisection level of the icosahedral grid  [5]
  --days D         length of the run                        [case default]
  --every HOURS    output interval                          [6]
  --surface NAME   sphere or wgs84                          [sphere]
  --damping F      biharmonic damping, as a fraction of what the step allows [0]
  --out DIR        Zarr store to write                      [out/<case>.zarr]
";

fn main() -> ExitCode {
    match Options::parse(std::env::args().skip(1)) {
        Ok(Some(options)) => match run(options) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("charney: {e}");
                ExitCode::FAILURE
            }
        },
        Ok(None) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("charney: {message}\n\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

struct Options {
    case: String,
    level: u32,
    days: f64,
    every_hours: f64,
    spheroid: bool,
    damping: f64,
    out: String,
}

impl Options {
    fn parse(args: impl Iterator<Item = String>) -> Result<Option<Options>, String> {
        let args: Vec<String> = args.collect();
        if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
            return Ok(None);
        }

        let case = args[0].clone();
        let days = match case.as_str() {
            "case1" => 12.0,
            "case2" => 5.0,
            "case5" => 15.0,
            "case6" => 14.0,
            other => return Err(format!("unknown case {other}")),
        };
        let mut options = Options {
            out: format!("out/{case}.zarr"),
            case,
            level: 5,
            days,
            every_hours: 6.0,
            spheroid: false,
            damping: 0.0,
        };

        let mut rest = args[1..].iter();
        while let Some(flag) = rest.next() {
            let mut value = || {
                rest.next()
                    .cloned()
                    .ok_or_else(|| format!("{flag} wants a value"))
            };
            match flag.as_str() {
                "--level" => options.level = parse(&value()?, flag)?,
                "--days" => options.days = parse(&value()?, flag)?,
                "--every" => options.every_hours = parse(&value()?, flag)?,
                "--damping" => options.damping = parse(&value()?, flag)?,
                "--out" => options.out = value()?,
                "--surface" => {
                    options.spheroid = match value()?.as_str() {
                        "sphere" => false,
                        "wgs84" => true,
                        other => return Err(format!("unknown surface {other}")),
                    }
                }
                other => return Err(format!("unknown option {other}")),
            }
        }
        Ok(Some(options))
    }
}

fn parse<T: std::str::FromStr>(text: &str, flag: &str) -> Result<T, String> {
    text.parse()
        .map_err(|_| format!("{flag} did not take {text}"))
}

fn run(options: Options) -> std::io::Result<()> {
    let spheroid = Wgs84::WGS84;
    let sphere = Sphere::EARTH;
    let surface: &dyn Surface = if options.spheroid { &spheroid } else { &sphere };
    let radius = surface.authalic_radius();

    let mesh = Mesh::scvt(options.level, 60, surface);
    let cells = mesh.n_cells();
    let (mut state, bottom, frozen) = initial_state(&options.case, &mesh, surface, radius);

    let coords: Vec<(f64, f64)> = (0..cells)
        .map(|c| lonlat(surface, mesh.cell_pos[c]))
        .collect();

    let mut model = Model::new(mesh, surface, GRAVITY, OMEGA);
    model.bottom = bottom.clone();
    model.frozen_flow = frozen;
    let mut work = Work::for_state(&model.mesh, &state);

    let mean_depth = state.h.iter().sum::<f64>() / cells as f64;
    let interval = options.every_hours * 3600.0;
    let per_slice = (interval / model.stable_step(mean_depth, 0.4)).ceil() as usize;
    let dt = interval / per_slice as f64;
    if options.damping > 0.0 {
        model.nu4 = model.nu4_stable(dt, options.damping.min(1.0));
    }
    let slices = (options.days * 86_400.0 / interval).round() as usize + 1;

    let store = Store::create(
        &options.out,
        &[
            Attribute::text("title", &format!("Williamson {}", options.case)),
            Attribute::text("surface", if options.spheroid { "wgs84" } else { "sphere" }),
            Attribute::number("level", options.level as f64),
            Attribute::number("nu4", model.nu4),
            Attribute::text("integrator", "rk4"),
        ],
    )?;
    let times: Vec<f64> = (0..slices).map(|i| i as f64 * interval).collect();
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
    store.coordinate("bottom", "cell", "m", &bottom)?;
    store.coordinate("time", "time", "seconds", &times)?;

    let mut h_out = store.array("h", cells, slices, "cell", "m", "fluid thickness")?;
    let mut speed_out = store.array("speed", cells, slices, "cell", "m s-1", "wind speed")?;
    let mut tracer_out = match state.n_tracers() {
        0 => None,
        _ => Some(store.array("tracer", cells, slices, "cell", "1", "mixing ratio")?),
    };

    println!(
        "{} on {} at level {} ({} cells, dt {:.0} s), {} days",
        options.case,
        if options.spheroid { "wgs84" } else { "sphere" },
        options.level,
        cells,
        dt,
        options.days
    );
    let tracked = if state.n_tracers() > 0 {
        "tracer mass"
    } else {
        "pot. enstrophy"
    };
    println!(
        "{:>7} {:>13} {:>13} {:>13}",
        "day", "mass", "energy", tracked
    );

    let start = model.invariants(&state);
    let mut speed = vec![0.0; cells];
    let mut mixing = vec![0.0; cells];

    for (slice, &now_time) in times.iter().enumerate() {
        model.cell_speed(&state, &mut speed);
        let now = model.invariants(&state);
        h_out.push(&state.h)?;
        speed_out.push(&speed)?;
        if let Some(out) = tracer_out.as_mut() {
            model.mixing_ratio(&state, 0, &mut mixing);
            out.push(&mixing)?;
        }
        let last = if start.tracer_mass.is_empty() {
            (now.potential_enstrophy - start.potential_enstrophy) / start.potential_enstrophy
        } else {
            (now.tracer_mass[0] - start.tracer_mass[0]) / start.tracer_mass[0]
        };
        println!(
            "{:>7.2} {:>13.2e} {:>13.2e} {:>13.2e}",
            now_time / 86_400.0,
            (now.mass - start.mass) / start.mass,
            (now.energy - start.energy) / start.energy,
            last,
        );
        if slice + 1 < slices {
            model.integrate(&mut state, dt, per_slice, &mut work);
        }
    }

    println!("wrote {} ({slices} slices)", options.out);
    Ok(())
}

fn initial_state(
    case: &str,
    mesh: &Mesh,
    surface: &dyn Surface,
    radius: f64,
) -> (State, Vec<f64>, bool) {
    let flat = vec![0.0; mesh.n_cells()];
    match case {
        "case1" => (
            CosineBell::standard(radius).state(mesh, surface),
            flat,
            true,
        ),
        "case2" => (
            SteadyZonalFlow::standard(radius).state(mesh, surface),
            flat,
            false,
        ),
        "case5" => {
            let (state, bottom) = MountainFlow::default().state(mesh, surface);
            (state, bottom, false)
        }
        "case6" => (RossbyHaurwitz::default().state(mesh, surface), flat, false),
        _ => unreachable!("the case name was checked while parsing"),
    }
}
