//! Icosahedral geodesic mesh: a Delaunay primal and its circumcentric dual.
//!
//! The primal graph is a recursively bisected icosahedron on the unit sphere,
//! pushed onto the surface. The discrete manifold is the inscribed polyhedron
//! it spans: lengths are chords, areas are planar, and the three kites of a
//! triangle tile it exactly, so mass sums telescope to machine precision. The
//! polyhedron's area falls short of the surface's by O(h^2).
//!
//! Fields are sampled on the surface itself. Circumcentres and edge midpoints
//! therefore appear twice: `*_pos` for measurement, `*_site` for sampling.
//!
//! Naming follows MPAS: a *cell* is a Voronoi cell (a primal vertex), a *dual*
//! is a Voronoi corner (a primal triangle), and an *edge* separates two cells
//! and joins two duals.

use std::collections::HashMap;

use crate::geometry::Surface;
use crate::ragged::Ragged;
use crate::vec3::Vec3;

/// A geodesic mesh and every geometric quantity the solver reads.
pub struct Mesh {
    /// Bisection level. Level 0 is the bare icosahedron.
    pub level: u32,

    /// Cell centres, on the surface.
    pub cell_pos: Vec<Vec3>,
    /// Voronoi corners: circumcentres of the planar primal triangles.
    pub dual_pos: Vec<Vec3>,
    /// Chord midpoints of the edges.
    pub edge_pos: Vec<Vec3>,
    /// Voronoi corners projected to the surface, where fields are sampled.
    pub dual_site: Vec<Vec3>,
    /// Edge midpoints projected to the surface, where fields are sampled.
    pub edge_site: Vec<Vec3>,

    /// The two cells either side of each edge. The edge normal points from
    /// the first to the second.
    pub cells_on_edge: Vec<[usize; 2]>,
    /// The two duals at the ends of each edge, ordered along the tangent.
    pub duals_on_edge: Vec<[usize; 2]>,
    /// The three cells around each dual, counter-clockwise.
    pub cells_on_dual: Vec<[usize; 3]>,
    /// The three edges around each dual.
    pub edges_on_dual: Vec<[usize; 3]>,

    /// Edges around each cell, counter-clockwise. Entry `k` separates
    /// `duals_on_cell(c)[k]` from `duals_on_cell(c)[k + 1]`.
    pub edges_on_cell: Ragged<usize>,
    /// Duals around each cell, counter-clockwise.
    pub duals_on_cell: Ragged<usize>,
    /// Area of the intersection of a cell with each of its incident duals,
    /// indexed as `duals_on_cell`.
    pub kite_on_cell: Ragged<f64>,
    /// The same areas seen from the dual, indexed as `cells_on_dual`. Lets an
    /// average onto the duals gather rather than scatter.
    pub kite_on_dual: Vec<[f64; 3]>,
    /// Where an edge sits in each of its two cells' rings, in the order
    /// `cells_on_edge` gives. Saves a search in the upwind reconstruction.
    pub slot_on_cell: Vec<[usize; 2]>,
    /// `dv` of each of a cell's edges, signed `+` where the normal leaves the
    /// cell, indexed as `edges_on_cell`. The divergence then reads one
    /// contiguous stream where it would otherwise gather `dv` and a sign.
    pub signed_dv_on_cell: Ragged<f64>,
    /// `dc` of each of a dual's edges, signed `+` where the counter-clockwise
    /// circuit runs along the normal, indexed as `edges_on_dual`.
    pub signed_dc_on_dual: Vec<[f64; 3]>,

    /// Distance between the two cell centres an edge separates.
    pub dc: Vec<f64>,
    /// Length of the Voronoi edge, the distance between its two duals.
    pub dv: Vec<f64>,
    /// Unit normal at each edge, from the first cell to the second, tangent
    /// to the surface.
    pub edge_normal: Vec<Vec3>,
    /// Unit tangent at each edge, `outward x normal`, pointing from the first
    /// dual to the second.
    pub edge_tangent: Vec<Vec3>,

    /// Voronoi cell areas.
    pub area_cell: Vec<f64>,
    /// Primal triangle areas.
    pub area_dual: Vec<f64>,

    /// Outward surface normal at each cell centre, dual and edge.
    pub cell_up: Vec<Vec3>,
    pub dual_up: Vec<Vec3>,
    pub edge_up: Vec<Vec3>,
}

impl Mesh {
    pub fn n_cells(&self) -> usize {
        self.cell_pos.len()
    }
    pub fn n_duals(&self) -> usize {
        self.dual_pos.len()
    }
    pub fn n_edges(&self) -> usize {
        self.dc.len()
    }

    /// Sign of the edge normal relative to a cell: `+1` where the normal
    /// points out of the cell.
    pub fn n_sign(&self, edge: usize, cell: usize) -> f64 {
        if self.cells_on_edge[edge][0] == cell {
            1.0
        } else {
            -1.0
        }
    }

    /// Shortest edge length, the resolution the time step is set from.
    pub fn min_dc(&self) -> f64 {
        self.dc.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Bisected icosahedron at the given level, embedded in the surface.
    pub fn icosahedral<S: Surface + ?Sized>(level: u32, surface: &S) -> Mesh {
        let (unit, tris) = bisected_icosahedron(level);
        let topology = Topology::of(&tris, &unit);
        Mesh::from_topology(&topology, &unit, surface, level)
    }

    /// The same grid after Lloyd relaxation on the unit sphere, which drives
    /// each cell centre to the centroid of its own Voronoi cell.
    ///
    /// The bisected icosahedron has cell areas spread over a factor of about
    /// 1.35, and the C-grid reconstruction of the tangential velocity is only
    /// exact on a uniform mesh, so the raw grid caps the accuracy the solver
    /// reaches. Relaxation runs on the unit sphere and the result is embedded
    /// afterwards, so a sphere and a spheroid share one grid and differ only
    /// in geometry. Sixty sweeps are enough; `trisk` has a test for what
    /// they buy.
    ///
    /// Moving a cell centre changes no incidence, so the topology is built
    /// once and each sweep touches positions alone.
    pub fn scvt<S: Surface + ?Sized>(level: u32, iterations: u32, surface: &S) -> Mesh {
        let (mut unit, tris) = bisected_icosahedron(level);
        let topology = Topology::of(&tris, &unit);
        let mut sweep = LloydSweep::new(&topology);
        for _ in 0..iterations {
            sweep.run(&topology, &mut unit);
        }
        Mesh::from_topology(&topology, &unit, surface, level)
    }

    fn from_topology<S: Surface + ?Sized>(
        topology: &Topology,
        unit: &[Vec3],
        surface: &S,
        level: u32,
    ) -> Mesh {
        let n_cells = unit.len();
        let n_duals = topology.tris.len();
        let n_edges = topology.cells_on_edge.len();

        let cell_pos: Vec<Vec3> = unit.iter().map(|&u| surface.embed(u)).collect();
        let dual_pos: Vec<Vec3> = topology
            .tris
            .iter()
            .map(|t| Vec3::circumcenter(cell_pos[t[0]], cell_pos[t[1]], cell_pos[t[2]]))
            .collect();
        let dual_site: Vec<Vec3> = dual_pos.iter().map(|&c| surface.project(c)).collect();

        // Per-edge frame. The normal runs from the first cell to the second,
        // projected into the tangent plane at the edge midpoint.
        let mut edge_pos = Vec::with_capacity(n_edges);
        let mut edge_site = Vec::with_capacity(n_edges);
        let mut edge_up = Vec::with_capacity(n_edges);
        let mut edge_normal = Vec::with_capacity(n_edges);
        let mut edge_tangent = Vec::with_capacity(n_edges);
        let mut dc = Vec::with_capacity(n_edges);
        let mut dv = Vec::with_capacity(n_edges);
        let mut duals_on_edge = Vec::with_capacity(n_edges);

        for e in 0..n_edges {
            let [c0, c1] = topology.cells_on_edge[e];
            let mid = (cell_pos[c0] + cell_pos[c1]) / 2.0;
            let site = surface.project(mid);
            let up = surface.normal(site);
            let n = (cell_pos[c1] - cell_pos[c0]).tangent_to(up).normalized();
            let t = up.cross(n);

            let [v0, v1] = topology.duals_on_edge[e];
            let (v0, v1) = if (dual_pos[v1] - dual_pos[v0]).dot(t) >= 0.0 {
                (v0, v1)
            } else {
                (v1, v0)
            };

            dc.push((cell_pos[c1] - cell_pos[c0]).norm());
            dv.push((dual_pos[v1] - dual_pos[v0]).norm());
            duals_on_edge.push([v0, v1]);
            edge_pos.push(mid);
            edge_site.push(site);
            edge_up.push(up);
            edge_normal.push(n);
            edge_tangent.push(t);
        }

        let cell_up: Vec<Vec3> = cell_pos.iter().map(|&p| surface.normal(p)).collect();
        let dual_up: Vec<Vec3> = dual_site.iter().map(|&p| surface.normal(p)).collect();

        let mut kite_on_cell =
            Ragged::filled((0..n_cells).map(|c| topology.duals_on_cell.row_len(c)), 0.0);
        let mut area_cell = vec![0.0; n_cells];
        let mut area_dual = vec![0.0; n_duals];
        kites(
            topology,
            &cell_pos,
            &dual_pos,
            &edge_pos,
            &mut kite_on_cell,
            &mut area_cell,
        );
        for c in 0..n_cells {
            for (k, &v) in topology.duals_on_cell.row(c).iter().enumerate() {
                area_dual[v] += kite_on_cell.row(c)[k];
            }
        }

        // Dual incidence, counter-clockwise, consistent with the primal
        // triangle winding.
        let mut kite_on_dual = vec![[0.0f64; 3]; n_duals];
        let mut cells_on_dual = Vec::with_capacity(n_duals);
        let mut edges_on_dual = Vec::with_capacity(n_duals);
        for (t_idx, t) in topology.tris.iter().enumerate() {
            let mut ccw = *t;
            let n = (cell_pos[t[1]] - cell_pos[t[0]]).cross(cell_pos[t[2]] - cell_pos[t[0]]);
            if n.dot(dual_up[t_idx]) < 0.0 {
                ccw.swap(1, 2);
            }
            let mut es = [0usize; 3];
            for k in 0..3 {
                es[k] = topology.edge_between(ccw[k], ccw[(k + 1) % 3]);
            }
            for (k, &c) in ccw.iter().enumerate() {
                let slot = topology
                    .duals_on_cell
                    .row(c)
                    .iter()
                    .position(|&v| v == t_idx)
                    .expect("a triangle rings each of its own cells");
                kite_on_dual[t_idx][k] = kite_on_cell.row(c)[slot];
            }
            cells_on_dual.push(ccw);
            edges_on_dual.push(es);
        }

        let mut mesh = Mesh {
            level,
            cell_pos,
            dual_pos,
            edge_pos,
            dual_site,
            edge_site,
            cells_on_edge: topology.cells_on_edge.clone(),
            duals_on_edge,
            cells_on_dual,
            edges_on_dual,
            edges_on_cell: topology.edges_on_cell.clone(),
            duals_on_cell: topology.duals_on_cell.clone(),
            kite_on_cell,
            kite_on_dual,
            slot_on_cell: topology.slot_on_cell.clone(),
            dc,
            dv,
            edge_normal,
            edge_tangent,
            area_cell,
            area_dual,
            cell_up,
            dual_up,
            edge_up,
            signed_dv_on_cell: Ragged::from_rows(&[]),
            signed_dc_on_dual: Vec::new(),
        };
        mesh.index_signed_metrics();
        mesh
    }

    /// Fold each orientation sign into the length it multiplies.
    fn index_signed_metrics(&mut self) {
        let rows: Vec<Vec<f64>> = (0..self.n_cells())
            .map(|c| {
                self.edges_on_cell
                    .row(c)
                    .iter()
                    .map(|&e| self.n_sign(e, c) * self.dv[e])
                    .collect()
            })
            .collect();
        self.signed_dv_on_cell = Ragged::from_rows(&rows);

        self.signed_dc_on_dual = (0..self.n_duals())
            .map(|v| {
                let ring = self.cells_on_dual[v];
                let mut s = [0.0; 3];
                for k in 0..3 {
                    let e = self.edges_on_dual[v][k];
                    let sign = if self.cells_on_edge[e][0] == ring[k] {
                        1.0
                    } else {
                        -1.0
                    };
                    s[k] = sign * self.dc[e];
                }
                s
            })
            .collect();
    }
}

/// Incidence that survives a cell centre moving: which cells an edge
/// separates, which edges and duals ring a cell, and which duals an edge
/// joins. Built once and shared by every Lloyd sweep.
struct Topology {
    tris: Vec<[usize; 3]>,
    cells_on_edge: Vec<[usize; 2]>,
    duals_on_edge: Vec<[usize; 2]>,
    edges_on_cell: Ragged<usize>,
    duals_on_cell: Ragged<usize>,
    slot_on_cell: Vec<[usize; 2]>,
    /// Edge index by sorted cell pair.
    key_to_edge: HashMap<(usize, usize), usize>,
}

impl Topology {
    fn of(tris: &[[usize; 3]], unit: &[Vec3]) -> Topology {
        let n_cells = unit.len();
        let mut key_to_edge: HashMap<(usize, usize), usize> = HashMap::new();
        let mut cells_on_edge: Vec<[usize; 2]> = Vec::new();
        let mut duals_of_edge: Vec<Vec<usize>> = Vec::new();
        for (t_idx, t) in tris.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                let e = *key_to_edge.entry(key).or_insert_with(|| {
                    cells_on_edge.push([key.0, key.1]);
                    duals_of_edge.push(Vec::with_capacity(2));
                    cells_on_edge.len() - 1
                });
                duals_of_edge[e].push(t_idx);
            }
        }
        assert!(
            duals_of_edge.iter().all(|d| d.len() == 2),
            "every edge of a closed surface meets exactly two triangles"
        );
        let duals_on_edge: Vec<[usize; 2]> = duals_of_edge.iter().map(|d| [d[0], d[1]]).collect();

        // Which triangles touch each cell.
        let mut duals_at_cell: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
        for (t_idx, t) in tris.iter().enumerate() {
            for &c in t.iter() {
                duals_at_cell[c].push(t_idx);
            }
        }

        // The ring of duals around a cell, in cycle order. Which way round it
        // runs still depends on geometry, so the sense is fixed later, once
        // per build, by `orient_rings`.
        let mut duals_rows: Vec<Vec<usize>> = Vec::with_capacity(n_cells);
        let mut edges_rows: Vec<Vec<usize>> = Vec::with_capacity(n_cells);
        for (c, incident) in duals_at_cell.iter().enumerate() {
            let ring = cycle(c, incident, tris);
            let m = ring.len();
            let mut edges = Vec::with_capacity(m);
            for k in 0..m {
                let other = shared_other_vertex(tris[ring[k]], tris[ring[(k + 1) % m]], c);
                let key = if c < other { (c, other) } else { (other, c) };
                edges.push(key_to_edge[&key]);
            }
            duals_rows.push(ring);
            edges_rows.push(edges);
        }

        let mut topology = Topology {
            tris: tris.to_vec(),
            cells_on_edge,
            duals_on_edge,
            edges_on_cell: Ragged::from_rows(&edges_rows),
            duals_on_cell: Ragged::from_rows(&duals_rows),
            slot_on_cell: Vec::new(),
            key_to_edge,
        };
        topology.orient_rings(unit);
        topology.index_slots();
        topology
    }

    /// Where each edge sits in the rings of the two cells it separates.
    fn index_slots(&mut self) {
        self.slot_on_cell = (0..self.cells_on_edge.len())
            .map(|e| {
                let [c0, c1] = self.cells_on_edge[e];
                let find = |cell: usize| {
                    self.edges_on_cell
                        .row(cell)
                        .iter()
                        .position(|&x| x == e)
                        .expect("an edge lies in the ring of each cell it separates")
                };
                [find(c0), find(c1)]
            })
            .collect();
    }

    fn edge_between(&self, a: usize, b: usize) -> usize {
        let key = if a < b { (a, b) } else { (b, a) };
        self.key_to_edge[&key]
    }

    /// Turn every cell's ring counter-clockwise as seen from outside, for the
    /// given positions. Cheap, and the sense can only flip if a cell centre
    /// crosses its own ring, which relaxation never does.
    fn orient_rings(&mut self, pos: &[Vec3]) {
        for c in 0..pos.len() {
            let ring = self.duals_on_cell.row(c);
            let a = circumcenter_of(&self.tris[ring[0]], pos) - pos[c];
            let b = circumcenter_of(&self.tris[ring[1]], pos) - pos[c];
            if a.cross(b).dot(pos[c].normalized()) < 0.0 {
                self.duals_on_cell.row_mut(c)[1..].reverse();
                let m = self.edges_on_cell.row_len(c);
                let ring: Vec<usize> = self.duals_on_cell.row(c).to_vec();
                let edges = self.edges_on_cell.row_mut(c);
                for k in 0..m {
                    let other =
                        shared_other_vertex(self.tris[ring[k]], self.tris[ring[(k + 1) % m]], c);
                    let key = if c < other { (c, other) } else { (other, c) };
                    edges[k] = self.key_to_edge[&key];
                }
            }
        }
    }
}

fn circumcenter_of(t: &[usize; 3], pos: &[Vec3]) -> Vec3 {
    Vec3::circumcenter(pos[t[0]], pos[t[1]], pos[t[2]])
}

/// Kite areas and the cell areas they sum to, from a set of positions.
fn kites(
    topology: &Topology,
    cell_pos: &[Vec3],
    dual_pos: &[Vec3],
    edge_pos: &[Vec3],
    kite: &mut Ragged<f64>,
    area_cell: &mut [f64],
) {
    for c in 0..cell_pos.len() {
        let ring = topology.duals_on_cell.row(c);
        let edges = topology.edges_on_cell.row(c);
        let m = ring.len();
        let x = cell_pos[c];
        let mut total = 0.0;
        for k in 0..m {
            let v = dual_pos[ring[k]];
            let prev = edge_pos[edges[(k + m - 1) % m]];
            let next = edge_pos[edges[k]];
            let a = Vec3::triangle_area(x, prev, v) + Vec3::triangle_area(x, v, next);
            kite.row_mut(c)[k] = a;
            total += a;
        }
        area_cell[c] = total;
    }
}

/// One Lloyd sweep: move every cell centre to the centroid of its own
/// Voronoi cell on the unit sphere. Buffers are reused across sweeps.
struct LloydSweep {
    dual_pos: Vec<Vec3>,
    edge_pos: Vec<Vec3>,
    centroid: Vec<Vec3>,
}

impl LloydSweep {
    fn new(topology: &Topology) -> LloydSweep {
        LloydSweep {
            dual_pos: vec![Vec3::ZERO; topology.tris.len()],
            edge_pos: vec![Vec3::ZERO; topology.cells_on_edge.len()],
            centroid: vec![Vec3::ZERO; topology.duals_on_cell.rows()],
        }
    }

    fn run(&mut self, topology: &Topology, unit: &mut [Vec3]) {
        for (v, t) in self.dual_pos.iter_mut().zip(&topology.tris) {
            *v = circumcenter_of(t, unit);
        }
        for (e, &[c0, c1]) in self.edge_pos.iter_mut().zip(&topology.cells_on_edge) {
            *e = (unit[c0] + unit[c1]) / 2.0;
        }

        for (c, out) in self.centroid.iter_mut().enumerate() {
            let ring = topology.duals_on_cell.row(c);
            let edges = topology.edges_on_cell.row(c);
            let m = ring.len();
            let x = unit[c];
            let mut acc = Vec3::ZERO;
            let mut area = 0.0;
            for k in 0..m {
                let v = self.dual_pos[ring[k]];
                let prev = self.edge_pos[edges[(k + m - 1) % m]];
                let next = self.edge_pos[edges[k]];
                for (a, b) in [(prev, v), (v, next)] {
                    let t = Vec3::triangle_area(x, a, b);
                    acc += (x + a + b) * (t / 3.0);
                    area += t;
                }
            }
            *out = (acc / area).normalized();
        }
        unit.copy_from_slice(&self.centroid);
    }
}

/// The vertex two triangles share besides `c`.
fn shared_other_vertex(a: [usize; 3], b: [usize; 3], c: usize) -> usize {
    for &x in a.iter() {
        if x != c && b.contains(&x) {
            return x;
        }
    }
    unreachable!("adjacent triangles around a cell share a second vertex")
}

/// Walk the incident triangles of a cell into a cycle. Which way round it
/// runs is settled afterwards by `Topology::orient_rings`.
fn cycle(c: usize, incident: &[usize], tris: &[[usize; 3]]) -> Vec<usize> {
    let adjacent = |a: usize, b: usize| -> bool {
        let ta = tris[a];
        let tb = tris[b];
        ta.iter().filter(|&&x| x != c && tb.contains(&x)).count() == 1
    };

    let mut ring = vec![incident[0]];
    let mut used = vec![false; incident.len()];
    used[0] = true;
    while ring.len() < incident.len() {
        let last = *ring.last().unwrap();
        let mut advanced = false;
        for (k, &t) in incident.iter().enumerate() {
            if !used[k] && adjacent(last, t) {
                ring.push(t);
                used[k] = true;
                advanced = true;
                break;
            }
        }
        assert!(advanced, "the triangles around a cell form one cycle");
    }
    ring
}

/// Unit-sphere icosahedron, bisected `level` times.
fn bisected_icosahedron(level: u32) -> (Vec<Vec3>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut verts: Vec<Vec3> = Vec::with_capacity(12);
    for &s1 in &[1.0, -1.0] {
        for &s2 in &[1.0, -1.0] {
            verts.push(Vec3::new(0.0, s1, s2 * phi).normalized());
            verts.push(Vec3::new(s1, s2 * phi, 0.0).normalized());
            verts.push(Vec3::new(s2 * phi, 0.0, s1).normalized());
        }
    }

    // The 20 faces are the triples whose three mutual distances are minimal.
    let mut shortest = f64::INFINITY;
    for i in 0..12 {
        for j in (i + 1)..12 {
            shortest = shortest.min((verts[i] - verts[j]).norm());
        }
    }
    let tol = shortest * 1.05;
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(20);
    for i in 0..12 {
        for j in (i + 1)..12 {
            for k in (j + 1)..12 {
                let close = (verts[i] - verts[j]).norm() < tol
                    && (verts[j] - verts[k]).norm() < tol
                    && (verts[i] - verts[k]).norm() < tol;
                if close {
                    // Wind counter-clockwise seen from outside.
                    let n = (verts[j] - verts[i]).cross(verts[k] - verts[i]);
                    if n.dot(verts[i] + verts[j] + verts[k]) >= 0.0 {
                        tris.push([i, j, k]);
                    } else {
                        tris.push([i, k, j]);
                    }
                }
            }
        }
    }
    assert_eq!(tris.len(), 20, "the icosahedron has twenty faces");

    for _ in 0..level {
        let mut mid: HashMap<(usize, usize), usize> = HashMap::new();
        let mut next = Vec::with_capacity(tris.len() * 4);
        for t in &tris {
            let mut m = [0usize; 3];
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                m[k] = *mid.entry(key).or_insert_with(|| {
                    verts.push(((verts[a] + verts[b]) / 2.0).normalized());
                    verts.len() - 1
                });
            }
            next.push([t[0], m[0], m[2]]);
            next.push([m[0], t[1], m[1]]);
            next.push([m[2], m[1], t[2]]);
            next.push([m[0], m[1], m[2]]);
        }
        tris = next;
    }

    (verts, tris)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Sphere, Wgs84};

    #[test]
    fn relaxation_reaches_a_fixed_point() {
        let s = Sphere::EARTH;
        let a = Mesh::scvt(3, 60, &s);
        let b = Mesh::scvt(3, 61, &s);
        let moved = (0..a.n_cells())
            .map(|c| (a.cell_pos[c] - b.cell_pos[c]).norm() / a.min_dc())
            .fold(0.0f64, f64::max);
        assert!(
            moved < 1e-3,
            "a further sweep still moves cells by {moved} of an edge"
        );
    }

    #[test]
    fn relaxation_keeps_the_topology() {
        let raw = Mesh::icosahedral(3, &Sphere::EARTH);
        let even = Mesh::scvt(3, 20, &Sphere::EARTH);
        assert_eq!(raw.n_cells(), even.n_cells());
        assert_eq!(raw.n_edges(), even.n_edges());
        assert_eq!(raw.n_duals(), even.n_duals());
        for v in 0..even.n_duals() {
            let t = even.cells_on_dual[v];
            let direct = Vec3::triangle_area(
                even.cell_pos[t[0]],
                even.cell_pos[t[1]],
                even.cell_pos[t[2]],
            );
            assert!((even.area_dual[v] - direct).abs() / direct < 1e-13);
        }
    }

    #[test]
    fn counts_follow_the_bisection_law() {
        for level in 0..4u32 {
            let m = Mesh::icosahedral(level, &Sphere::new(1.0));
            let f = 20 * 4usize.pow(level);
            assert_eq!(m.n_duals(), f);
            assert_eq!(m.n_cells(), 10 * 4usize.pow(level) + 2);
            assert_eq!(m.n_edges(), 30 * 4usize.pow(level));
            let chi = m.n_cells() as i64 - m.n_edges() as i64 + m.n_duals() as i64;
            assert_eq!(chi, 2, "Euler characteristic of a sphere");
        }
    }

    #[test]
    fn polyhedral_area_converges_second_order() {
        let s = Sphere::new(1.0);
        let err: Vec<f64> = (1..5u32)
            .map(|level| {
                let m = Mesh::icosahedral(level, &s);
                let total: f64 = m.area_cell.iter().sum();
                (total - s.total_area()).abs() / s.total_area()
            })
            .collect();
        for w in err.windows(2) {
            let rate = (w[0] / w[1]).log2();
            assert!(
                (rate - 2.0).abs() < 0.1,
                "convergence rate {rate}, errors {err:?}"
            );
        }
    }

    #[test]
    fn cell_and_dual_areas_agree_on_the_total() {
        let m = Mesh::icosahedral(3, &Sphere::new(1.0));
        let a: f64 = m.area_cell.iter().sum();
        let b: f64 = m.area_dual.iter().sum();
        assert!((a - b).abs() / a < 1e-12, "kites must tile both ways");
    }

    #[test]
    fn kite_sums_match_the_primal_triangle_areas() {
        let m = Mesh::icosahedral(2, &Sphere::new(1.0));
        for v in 0..m.n_duals() {
            let t = m.cells_on_dual[v];
            let direct = Vec3::triangle_area(m.cell_pos[t[0]], m.cell_pos[t[1]], m.cell_pos[t[2]]);
            let rel = (m.area_dual[v] - direct).abs() / direct;
            assert!(
                rel < 1e-13,
                "dual {v}: kites {} against {direct}",
                m.area_dual[v]
            );
        }
    }

    #[test]
    fn edge_frames_are_orthonormal_and_tangent() {
        let m = Mesh::icosahedral(2, &Wgs84::WGS84);
        for e in 0..m.n_edges() {
            let (n, t, up) = (m.edge_normal[e], m.edge_tangent[e], m.edge_up[e]);
            assert!(n.dot(up).abs() < 1e-12);
            assert!(t.dot(up).abs() < 1e-12);
            assert!(n.dot(t).abs() < 1e-12);
            assert!((n.norm() - 1.0).abs() < 1e-12);
            assert!((t.norm() - 1.0).abs() < 1e-12);
            // The tangent runs from the first dual to the second.
            let d = m.dual_pos[m.duals_on_edge[e][1]] - m.dual_pos[m.duals_on_edge[e][0]];
            assert!(d.dot(t) > 0.0);
        }
    }

    #[test]
    fn every_cell_ring_closes_on_its_edges() {
        let m = Mesh::icosahedral(2, &Sphere::new(1.0));
        for c in 0..m.n_cells() {
            let ring = &m.duals_on_cell.row(c);
            let edges = &m.edges_on_cell.row(c);
            assert_eq!(ring.len(), edges.len());
            for k in 0..ring.len() {
                let e = edges[k];
                let pair = m.duals_on_edge[e];
                let want = (ring[k], ring[(k + 1) % ring.len()]);
                assert!(
                    (pair[0], pair[1]) == want || (pair[1], pair[0]) == want,
                    "cell {c} slot {k}: edge {e} joins {pair:?}, ring gives {want:?}"
                );
                assert!(m.cells_on_edge[e].contains(&c));
            }
        }
    }

    #[test]
    fn wgs84_mesh_areas_converge_to_the_spheroid_area() {
        let s = Wgs84::WGS84;
        let m = Mesh::icosahedral(5, &s);
        let total: f64 = m.area_cell.iter().sum();
        let err = (total - s.total_area()).abs() / s.total_area();
        assert!(err < 5e-4, "relative area error {err}");
    }
}
