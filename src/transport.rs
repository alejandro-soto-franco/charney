//! Tracer flux reconstruction.
//!
//! The centred average of the two cell values either side of an edge is
//! second order and carries no dissipation at all, so a sharp tracer picks up
//! dispersive ripples and goes negative. The alternative here takes the value
//! from the upwind cell and extrapolates it to the edge with a least-squares
//! gradient, optionally limited so the reconstruction introduces no extremum
//! the cell's neighbours do not already have.
//!
//! The gradient is fitted in the tangent plane at each cell centre against
//! the cells across its edges. Since the mesh never moves, the normal
//! equations are inverted once and what survives is a row of weights per
//! neighbour.

use crate::mesh::Mesh;
use crate::vec3::Vec3;

/// How the mixing ratio reaches an edge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TracerFlux {
    /// Centred average. Second order, no dissipation, not monotone.
    Centered,
    /// Upwind cell value extrapolated with the least-squares gradient.
    Upwind,
    /// The same, limited after Barth and Jespersen (1989), so no cell hands
    /// an edge a value outside the range of its own neighbourhood.
    #[default]
    LimitedUpwind,
}

/// Precomputed least-squares gradient stencils, one per cell.
pub struct Gradients {
    /// Tangent-plane basis at each cell centre.
    basis: Vec<(Vec3, Vec3)>,
    /// Two weights per neighbour, laid out to match `Mesh::edges_on_cell`.
    /// The gradient is `sum_i w_i (q_neighbour - q_cell)` in that basis.
    weights: Vec<Vec<(f64, f64)>>,
    /// Components of the arm from a cell centre to each of its edge
    /// midpoints, in the same basis and the same order.
    arms: Vec<Vec<(f64, f64)>>,
}

impl Gradients {
    pub fn new(mesh: &Mesh) -> Gradients {
        let mut basis = Vec::with_capacity(mesh.n_cells());
        let mut weights = Vec::with_capacity(mesh.n_cells());
        let mut arms = Vec::with_capacity(mesh.n_cells());

        for c in 0..mesh.n_cells() {
            let up = mesh.cell_up[c];
            let e1 = mesh.edge_normal[mesh.edges_on_cell[c][0]]
                .tangent_to(up)
                .normalized();
            let e2 = up.cross(e1);

            // Normal equations for the plane fit, weighted by inverse
            // distance so the nearer neighbours count for more.
            let (mut m11, mut m12, mut m22) = (0.0, 0.0, 0.0);
            let mut rows: Vec<(f64, f64, f64)> = Vec::with_capacity(mesh.edges_on_cell[c].len());
            for &e in &mesh.edges_on_cell[c] {
                let [a, b] = mesh.cells_on_edge[e];
                let other = if a == c { b } else { a };
                let d = mesh.cell_pos[other] - mesh.cell_pos[c];
                let (x, y) = (d.dot(e1), d.dot(e2));
                let w = 1.0 / d.dot(d);
                m11 += w * x * x;
                m12 += w * x * y;
                m22 += w * y * y;
                rows.push((x, y, w));
            }

            let det = m11 * m22 - m12 * m12;
            let (i11, i12, i22) = (m22 / det, -m12 / det, m11 / det);
            let cell_weights = rows
                .iter()
                .map(|&(x, y, w)| (w * (i11 * x + i12 * y), w * (i12 * x + i22 * y)))
                .collect();

            let cell_arms = mesh.edges_on_cell[c]
                .iter()
                .map(|&e| {
                    let arm = mesh.edge_pos[e] - mesh.cell_pos[c];
                    (arm.dot(e1), arm.dot(e2))
                })
                .collect();

            basis.push((e1, e2));
            weights.push(cell_weights);
            arms.push(cell_arms);
            let _ = &basis;
        }

        Gradients {
            basis,
            weights,
            arms,
        }
    }

    /// Gradient of a cell field, in each cell's own tangent-plane basis.
    pub fn compute(&self, mesh: &Mesh, q: &[f64], out: &mut [(f64, f64)]) {
        for (c, o) in out.iter_mut().enumerate() {
            let (mut gx, mut gy) = (0.0, 0.0);
            for (&e, &(wx, wy)) in mesh.edges_on_cell[c].iter().zip(&self.weights[c]) {
                let [a, b] = mesh.cells_on_edge[e];
                let other = if a == c { b } else { a };
                let dq = q[other] - q[c];
                gx += wx * dq;
                gy += wy * dq;
            }
            *o = (gx, gy);
        }
    }

    /// Scale each gradient down until no edge reconstruction leaves the range
    /// of the cell's own neighbourhood.
    pub fn limit(&self, mesh: &Mesh, q: &[f64], grad: &mut [(f64, f64)]) {
        for c in 0..mesh.n_cells() {
            let mut lo = q[c];
            let mut hi = q[c];
            for &e in &mesh.edges_on_cell[c] {
                let [a, b] = mesh.cells_on_edge[e];
                let other = if a == c { b } else { a };
                lo = lo.min(q[other]);
                hi = hi.max(q[other]);
            }

            let (gx, gy) = grad[c];
            let mut phi: f64 = 1.0;
            for &(ax, ay) in &self.arms[c] {
                let d = gx * ax + gy * ay;
                let allowed = if d > 0.0 {
                    hi - q[c]
                } else if d < 0.0 {
                    lo - q[c]
                } else {
                    continue;
                };
                phi = phi.min((allowed / d).clamp(0.0, 1.0));
            }
            grad[c] = (gx * phi, gy * phi);
        }
    }

    /// Value the cell hands to one of its edges, `slot` indexing
    /// `Mesh::edges_on_cell`.
    pub fn at_edge(&self, cell: usize, slot: usize, q: f64, grad: (f64, f64)) -> f64 {
        let (ax, ay) = self.arms[cell][slot];
        q + grad.0 * ax + grad.1 * ay
    }

    /// Tangent-plane basis at a cell, for turning a stored gradient back into
    /// an ambient vector.
    pub fn basis(&self, cell: usize) -> (Vec3, Vec3) {
        self.basis[cell]
    }

    /// Position of `edge` in `Mesh::edges_on_cell[cell]`.
    pub fn slot_of(mesh: &Mesh, cell: usize, edge: usize) -> usize {
        mesh.edges_on_cell[cell]
            .iter()
            .position(|&e| e == edge)
            .expect("edge lies on its cell")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;

    /// The plane fit converges on the surface gradient of a smooth field.
    /// It cannot be exact: a field linear in the ambient coordinates is not
    /// linear along a curved surface, and the fit only ever sees the
    /// tangential part of the separation between two cells.
    #[test]
    fn the_gradient_converges() {
        let surface = Sphere::EARTH;
        let error = |level: u32| {
            let mesh = Mesh::scvt(level, 60, &surface);
            let grads = Gradients::new(&mesh);
            let q: Vec<f64> = mesh
                .cell_pos
                .iter()
                .map(|&p| p.z / surface.radius)
                .collect();
            let mut g = vec![(0.0, 0.0); mesh.n_cells()];
            grads.compute(&mesh, &q, &mut g);

            let (mut num, mut den) = (0.0, 0.0);
            for (c, &(gx, gy)) in g.iter().enumerate() {
                let (e1, e2) = grads.basis(c);
                let got = e1 * gx + e2 * gy;
                let want = Vec3::new(0.0, 0.0, 1.0 / surface.radius).tangent_to(mesh.cell_up[c]);
                num += (got - want).dot(got - want);
                den += want.dot(want);
            }
            (num / den).sqrt()
        };
        let (coarse, fine) = (error(3), error(4));
        let rate = (coarse / fine).log2();
        assert!(
            rate > 1.5,
            "convergence rate {rate}, from {coarse} to {fine}"
        );
        assert!(fine < 2e-3, "level 4 gradient error {fine}");
    }

    /// The limiter must stay out of the way where a field is smooth. It does
    /// zero the gradient at a smooth extremum, which is the known cost of the
    /// Barth-Jespersen form, so a handful of cells are always cut.
    #[test]
    fn the_limiter_leaves_a_smooth_field_alone() {
        let surface = Sphere::EARTH;
        let mesh = Mesh::scvt(4, 60, &surface);
        let grads = Gradients::new(&mesh);
        let q: Vec<f64> = mesh
            .cell_pos
            .iter()
            .map(|&p| p.z / surface.radius)
            .collect();

        let mut plain = vec![(0.0, 0.0); mesh.n_cells()];
        grads.compute(&mesh, &q, &mut plain);
        let mut limited = plain.clone();
        grads.limit(&mesh, &q, &mut limited);

        let untouched = (0..mesh.n_cells())
            .filter(|&c| {
                let a = plain[c].0.hypot(plain[c].1);
                let b = limited[c].0.hypot(limited[c].1);
                a == 0.0 || b / a > 0.99
            })
            .count();
        let fraction = untouched as f64 / mesh.n_cells() as f64;
        assert!(
            fraction > 0.99,
            "the limiter cut {:.1}% of a smooth field",
            100.0 * (1.0 - fraction)
        );
    }

    /// At a step it must bite, and the reconstructed edge values must stay
    /// inside the neighbourhood range.
    #[test]
    fn the_limiter_bounds_a_step() {
        let surface = Sphere::EARTH;
        let mesh = Mesh::scvt(3, 60, &surface);
        let grads = Gradients::new(&mesh);
        let q: Vec<f64> = mesh
            .cell_pos
            .iter()
            .map(|&p| if p.z > 0.0 { 1.0 } else { 0.0 })
            .collect();

        let mut g = vec![(0.0, 0.0); mesh.n_cells()];
        grads.compute(&mesh, &q, &mut g);
        grads.limit(&mesh, &q, &mut g);

        for c in 0..mesh.n_cells() {
            let mut lo = q[c];
            let mut hi = q[c];
            for &e in &mesh.edges_on_cell[c] {
                let [a, b] = mesh.cells_on_edge[e];
                let other = if a == c { b } else { a };
                lo = lo.min(q[other]);
                hi = hi.max(q[other]);
            }
            for slot in 0..mesh.edges_on_cell[c].len() {
                let v = grads.at_edge(c, slot, q[c], g[c]);
                assert!(
                    v >= lo - 1e-12 && v <= hi + 1e-12,
                    "cell {c} slot {slot} gave {v}"
                );
            }
        }
    }
}
