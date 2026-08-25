//! Perpendicular-flux weights after Thuburn et al. (2009) and Ringler et al.
//! (2010).
//!
//! A C grid stores only the velocity component normal to each edge, and the
//! Coriolis term needs the tangential one. It is reconstructed from the
//! normal components on the edges of the two cells the edge separates, with
//! weights obtained by walking counter-clockwise from the edge and
//! accumulating kite areas. The weights come out antisymmetric, which is the
//! condition for the Coriolis term to do no work.

use crate::mesh::Mesh;

/// One edge's reconstruction stencil.
pub struct Weights {
    /// Flattened stencil edges, indexed by `offset`.
    pub edges: Vec<u32>,
    /// Weight per stencil entry.
    pub w: Vec<f64>,
    /// `offset[e]..offset[e + 1]` is edge `e`'s slice.
    pub offset: Vec<u32>,
}

impl Weights {
    pub fn new(mesh: &Mesh) -> Weights {
        let mut edges = Vec::new();
        let mut w = Vec::new();
        let mut offset = Vec::with_capacity(mesh.n_edges() + 1);
        offset.push(0u32);

        for e in 0..mesh.n_edges() {
            for &c in mesh.cells_on_edge[e].iter() {
                let ring = &mesh.edges_on_cell[c];
                let m = ring.len();
                let i0 = ring
                    .iter()
                    .position(|&x| x == e)
                    .expect("edge lies on its own cell");
                let sign_e = mesh.n_sign(e, c);
                let mut r = 0.0;
                for step in 1..m {
                    let idx = (i0 + step) % m;
                    // Stepping from edge idx-1 to edge idx crosses the dual
                    // that sits between them, whose kite is at slot idx.
                    r += mesh.kite_on_cell[c][idx] / mesh.area_cell[c];
                    let eoe = ring[idx];
                    let sign_o = mesh.n_sign(eoe, c);
                    edges.push(eoe as u32);
                    w.push((0.5 - r) * sign_e * sign_o * mesh.dv[eoe] / mesh.dc[e]);
                }
            }
            offset.push(edges.len() as u32);
        }

        Weights { edges, w, offset }
    }

    pub fn stencil(&self, e: usize) -> (&[u32], &[f64]) {
        let a = self.offset[e] as usize;
        let b = self.offset[e + 1] as usize;
        (&self.edges[a..b], &self.w[a..b])
    }

    /// Tangential component at every edge, from the normal components.
    pub fn perpendicular(&self, u: &[f64], out: &mut [f64]) {
        for (e, o) in out.iter_mut().enumerate() {
            let (es, ws) = self.stencil(e);
            let mut acc = 0.0;
            for (&eoe, &wt) in es.iter().zip(ws) {
                acc += wt * u[eoe as usize];
            }
            *o = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Sphere, Wgs84};
    use crate::vec3::Vec3;

    /// Solid-body rotation about the polar axis, tangent to any surface of
    /// revolution about that axis.
    fn solid_body(rate: f64, x: Vec3) -> Vec3 {
        Vec3::new(0.0, 0.0, rate).cross(x)
    }

    fn perp_error_on(mesh: &Mesh) -> f64 {
        let weights = Weights::new(mesh);
        let rate = 1.0e-5;
        let u: Vec<f64> = (0..mesh.n_edges())
            .map(|e| solid_body(rate, mesh.edge_site[e]).dot(mesh.edge_normal[e]))
            .collect();
        let exact: Vec<f64> = (0..mesh.n_edges())
            .map(|e| solid_body(rate, mesh.edge_site[e]).dot(mesh.edge_tangent[e]))
            .collect();
        let mut got = vec![0.0; mesh.n_edges()];
        weights.perpendicular(&u, &mut got);

        let num: f64 = got.iter().zip(&exact).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f64 = exact.iter().map(|b| b * b).sum();
        (num / den).sqrt()
    }

    /// Second order on a relaxed grid, on either surface.
    #[test]
    fn perpendicular_reconstruction_is_second_order() {
        let sphere = |l| perp_error_on(&Mesh::scvt(l, 60, &Sphere::EARTH));
        let wgs84 = |l| perp_error_on(&Mesh::scvt(l, 60, &Wgs84::WGS84));
        for (name, err) in [
            ("sphere", &sphere as &dyn Fn(u32) -> f64),
            ("wgs84", &wgs84),
        ] {
            let (e3, e4) = (err(3), err(4));
            let rate = (e3 / e4).log2();
            assert!(e4 < 3e-3, "{name}: level 4 relative error {e4}");
            assert!(rate > 1.6, "{name}: convergence rate {rate}");
        }
    }

    /// On the unrelaxed icosahedron the same reconstruction manages half an
    /// order and more than ten times the error at the same resolution.
    #[test]
    fn relaxation_buys_second_order_reconstruction() {
        let raw = perp_error_on(&Mesh::icosahedral(4, &Sphere::EARTH));
        let even = perp_error_on(&Mesh::scvt(4, 60, &Sphere::EARTH));
        assert!(raw > 0.03, "raw grid error {raw}");
        assert!(
            even * 10.0 < raw,
            "relaxation gained too little: {raw} then {even}"
        );
    }

    /// The condition that makes the Coriolis term energetically neutral:
    /// `w(e, e') dc(e) dv(e)` is antisymmetric under swapping the pair.
    #[test]
    fn weights_are_antisymmetric() {
        let mesh = Mesh::icosahedral(2, &Sphere::EARTH);
        let weights = Weights::new(&mesh);
        // Weights near the antipodal edge of a hexagon vanish, so compare
        // against the typical magnitude rather than each pair's own.
        let typical: f64 = weights.w.iter().map(|x| x.abs()).sum::<f64>() / weights.w.len() as f64;
        let mut worst: f64 = 0.0;
        for e in 0..mesh.n_edges() {
            let (es, ws) = weights.stencil(e);
            for (&eoe, &wt) in es.iter().zip(ws) {
                let eoe = eoe as usize;
                let (es2, ws2) = weights.stencil(eoe);
                let back: f64 = es2
                    .iter()
                    .zip(ws2)
                    .filter(|&(&x, _)| x as usize == e)
                    .map(|(_, &v)| v)
                    .sum();
                let lhs = wt * mesh.dc[e] * mesh.dv[e];
                let rhs = back * mesh.dc[eoe] * mesh.dv[eoe];
                let scale = typical * mesh.dc[e] * mesh.dv[e];
                worst = worst.max((lhs + rhs).abs() / scale);
            }
        }
        assert!(worst < 1e-12, "worst scaled asymmetry {worst}");
    }
}
