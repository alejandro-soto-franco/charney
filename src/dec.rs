//! Exterior-derivative operators on the mesh and their orientation signs.
//!
//! Three operators cover the shallow-water system: the gradient of a cell
//! field along an edge normal, the divergence of an edge field over a cell,
//! and the circulation of an edge field around a dual. Each is the discrete
//! `d` of a cochain followed by the metric factor the mesh supplies. The
//! orientation signs are folded into those metrics when the mesh is built, so
//! each operator reads one contiguous stream where it would otherwise gather a
//! length and a sign separately.

use crate::mesh::Mesh;
use crate::parallel;

/// Component of `grad f` along each edge normal.
pub fn gradient(mesh: &Mesh, f: &[f64], out: &mut [f64]) {
    parallel::chunks(out, |start, part| {
        for (i, o) in part.iter_mut().enumerate() {
            let e = start + i;
            let [c0, c1] = mesh.cells_on_edge[e];
            *o = (f[c1] - f[c0]) / mesh.dc[e];
        }
    });
}

/// Divergence of an edge-normal field over each cell.
pub fn divergence(mesh: &Mesh, u: &[f64], out: &mut [f64]) {
    parallel::chunks(out, |start, part| {
        for (i, o) in part.iter_mut().enumerate() {
            let c = start + i;
            let mut acc = 0.0;
            for (&e, &dv) in mesh
                .edges_on_cell
                .row(c)
                .iter()
                .zip(mesh.signed_dv_on_cell.row(c))
            {
                acc += dv * u[e];
            }
            *o = acc / mesh.area_cell[c];
        }
    });
}

/// Circulation of an edge-normal field around each dual, divided by its area.
pub fn curl(mesh: &Mesh, u: &[f64], out: &mut [f64]) {
    parallel::chunks(out, |start, part| {
        for (i, o) in part.iter_mut().enumerate() {
            let v = start + i;
            let mut acc = 0.0;
            let dc = mesh.signed_dc_on_dual[v];
            for k in 0..3 {
                acc += dc[k] * u[mesh.edges_on_dual[v][k]];
            }
            *o = acc / mesh.area_dual[v];
        }
    });
}

/// Area-weighted average of a cell field onto the duals.
pub fn cell_to_dual(mesh: &Mesh, f: &[f64], out: &mut [f64]) {
    parallel::chunks(out, |start, part| {
        for (i, o) in part.iter_mut().enumerate() {
            let v = start + i;
            let cells = mesh.cells_on_dual[v];
            let kites = mesh.kite_on_dual[v];
            let mut acc = 0.0;
            for k in 0..3 {
                acc += kites[k] * f[cells[k]];
            }
            *o = acc / mesh.area_dual[v];
        }
    });
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
        let f: Vec<f64> = (0..mesh.n_cells())
            .map(|i| ((i as f64) * 0.618_033_988_749).sin())
            .collect();
        let mut g = vec![0.0; mesh.n_edges()];
        let mut z = vec![0.0; mesh.n_duals()];
        gradient(&mesh, &f, &mut g);
        curl(&mesh, &g, &mut z);
        let worst = z.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(worst < 1e-9, "largest spurious vorticity {worst}");
    }

    /// The divergence theorem on a closed surface: every interior face is
    /// counted once with each sign, so the area-weighted sum is zero.
    #[test]
    fn divergence_sums_to_zero_over_the_closed_surface() {
        let mesh = Mesh::icosahedral(3, &Sphere::new(1.0));
        let u: Vec<f64> = (0..mesh.n_edges())
            .map(|i| ((i as f64) * 1.324_717_957_244).cos())
            .collect();
        let mut d = vec![0.0; mesh.n_cells()];
        divergence(&mesh, &u, &mut d);
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
        let u: Vec<f64> = (0..mesh.n_edges())
            .map(|i| ((i as f64) * 0.754_877_666_25).sin())
            .collect();
        let mut z = vec![0.0; mesh.n_duals()];
        curl(&mesh, &u, &mut z);
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
