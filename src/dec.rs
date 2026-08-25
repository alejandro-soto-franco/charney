//! Exterior-derivative operators on the mesh and their orientation signs.
//!
//! Three operators cover the shallow-water system: the gradient of a cell
//! field along an edge normal, the divergence of an edge field over a cell,
//! and the circulation of an edge field around a dual. Each is the discrete
//! `d` of a cochain followed by the metric factor the mesh supplies, so the
//! sign tables here are the only place orientation is decided.

use crate::mesh::Mesh;

/// Orientation tables, built once per mesh.
pub struct Signs {
    /// `+1` where the edge normal leaves the cell, indexed as `edges_on_cell`.
    pub out_of_cell: Vec<Vec<f64>>,
    /// `+1` where the edge is traversed along its normal by the
    /// counter-clockwise circuit of the dual, indexed as `edges_on_dual`.
    pub around_dual: Vec<[f64; 3]>,
}

impl Signs {
    pub fn new(mesh: &Mesh) -> Signs {
        let out_of_cell = (0..mesh.n_cells())
            .map(|c| {
                mesh.edges_on_cell[c]
                    .iter()
                    .map(|&e| mesh.n_sign(e, c))
                    .collect()
            })
            .collect();

        let around_dual = (0..mesh.n_duals())
            .map(|v| {
                let ring = mesh.cells_on_dual[v];
                let mut s = [0.0; 3];
                for k in 0..3 {
                    let e = mesh.edges_on_dual[v][k];
                    // The circuit runs ring[k] -> ring[k+1]; the normal runs
                    // cells_on_edge[0] -> cells_on_edge[1].
                    s[k] = if mesh.cells_on_edge[e][0] == ring[k] {
                        1.0
                    } else {
                        -1.0
                    };
                }
                s
            })
            .collect();

        Signs {
            out_of_cell,
            around_dual,
        }
    }
}

/// Component of `grad f` along each edge normal.
pub fn gradient(mesh: &Mesh, f: &[f64], out: &mut [f64]) {
    for ((o, &[c0, c1]), &dc) in out.iter_mut().zip(&mesh.cells_on_edge).zip(&mesh.dc) {
        *o = (f[c1] - f[c0]) / dc;
    }
}

/// Divergence of an edge-normal field over each cell.
pub fn divergence(mesh: &Mesh, signs: &Signs, u: &[f64], out: &mut [f64]) {
    for (c, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (k, &e) in mesh.edges_on_cell[c].iter().enumerate() {
            acc += signs.out_of_cell[c][k] * mesh.dv[e] * u[e];
        }
        *o = acc / mesh.area_cell[c];
    }
}

/// Circulation of an edge-normal field around each dual, divided by its area.
pub fn curl(mesh: &Mesh, signs: &Signs, u: &[f64], out: &mut [f64]) {
    for (v, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for k in 0..3 {
            let e = mesh.edges_on_dual[v][k];
            acc += signs.around_dual[v][k] * mesh.dc[e] * u[e];
        }
        *o = acc / mesh.area_dual[v];
    }
}

/// Area-weighted average of a cell field onto the duals.
pub fn cell_to_dual(mesh: &Mesh, f: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|x| *x = 0.0);
    for (c, &fc) in f.iter().enumerate() {
        for (k, &v) in mesh.duals_on_cell[c].iter().enumerate() {
            out[v] += mesh.kite_on_cell[c][k] * fc;
        }
    }
    for (o, &area) in out.iter_mut().zip(&mesh.area_dual) {
        *o /= area;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;

    /// `curl grad = 0`: the circulation of a gradient around any closed
    /// circuit vanishes, here to rounding.
    #[test]
    fn curl_of_a_gradient_vanishes() {
        let mesh = Mesh::icosahedral(3, &Sphere::new(1.0));
        let signs = Signs::new(&mesh);
        let f: Vec<f64> = (0..mesh.n_cells())
            .map(|i| ((i as f64) * 0.618_033_988_749).sin())
            .collect();
        let mut g = vec![0.0; mesh.n_edges()];
        let mut z = vec![0.0; mesh.n_duals()];
        gradient(&mesh, &f, &mut g);
        curl(&mesh, &signs, &g, &mut z);
        let worst = z.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(worst < 1e-9, "largest spurious vorticity {worst}");
    }

    /// The divergence theorem on a closed surface: every interior face is
    /// counted once with each sign, so the area-weighted sum is zero.
    #[test]
    fn divergence_sums_to_zero_over_the_closed_surface() {
        let mesh = Mesh::icosahedral(3, &Sphere::new(1.0));
        let signs = Signs::new(&mesh);
        let u: Vec<f64> = (0..mesh.n_edges())
            .map(|i| ((i as f64) * 1.324_717_957_244).cos())
            .collect();
        let mut d = vec![0.0; mesh.n_cells()];
        divergence(&mesh, &signs, &u, &mut d);
        let total: f64 = (0..mesh.n_cells()).map(|c| d[c] * mesh.area_cell[c]).sum();
        let scale: f64 = (0..mesh.n_cells())
            .map(|c| d[c].abs() * mesh.area_cell[c])
            .sum();
        assert!(total.abs() / scale < 1e-13, "net divergence {total}");
    }

    /// The circulation of any edge field over all duals telescopes to zero.
    #[test]
    fn total_circulation_vanishes() {
        let mesh = Mesh::icosahedral(3, &Sphere::new(1.0));
        let signs = Signs::new(&mesh);
        let u: Vec<f64> = (0..mesh.n_edges())
            .map(|i| ((i as f64) * 0.754_877_666_25).sin())
            .collect();
        let mut z = vec![0.0; mesh.n_duals()];
        curl(&mesh, &signs, &u, &mut z);
        let total: f64 = (0..mesh.n_duals()).map(|v| z[v] * mesh.area_dual[v]).sum();
        let scale: f64 = (0..mesh.n_duals())
            .map(|v| z[v].abs() * mesh.area_dual[v])
            .sum();
        assert!(total.abs() / scale < 1e-13, "net circulation {total}");
    }

    #[test]
    fn cell_to_dual_preserves_a_constant() {
        let mesh = Mesh::icosahedral(2, &Sphere::new(1.0));
        let f = vec![3.5; mesh.n_cells()];
        let mut d = vec![0.0; mesh.n_duals()];
        cell_to_dual(&mesh, &f, &mut d);
        assert!(d.iter().all(|&x| (x - 3.5).abs() < 1e-13));
    }
}
