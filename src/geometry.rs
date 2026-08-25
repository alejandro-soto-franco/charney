//! Closed surfaces of revolution about the polar axis.
//!
//! Geometry reaches the solver through this trait alone. [`Mesh`] builds an
//! icosahedral graph on the unit sphere, pushes every point through
//! [`Surface::embed`], and measures lengths and areas in the ambient
//! embedding. Nothing downstream of the mesh reads the surface again, so
//! [`Sphere`] and [`Wgs84`] run the same solver.
//!
//! [`Mesh`]: crate::mesh::Mesh

use crate::vec3::Vec3;

/// A closed surface of revolution about the polar axis, diffeomorphic to the
/// unit sphere.
pub trait Surface: Send + Sync {
    /// Image of a unit-sphere point on the surface.
    fn embed(&self, unit: Vec3) -> Vec3;

    /// Surface point under an ambient point, along the ray the
    /// parametrisation defines. Circumcentres are built in the ambient space
    /// and returned to the surface through this.
    fn project(&self, ambient: Vec3) -> Vec3;

    /// Outward unit normal at a point on the surface.
    fn normal(&self, surface: Vec3) -> Vec3;

    /// Exact total area, for the mesh convergence test.
    fn total_area(&self) -> f64;

    /// Radius of the sphere of equal area. Sets the gravity-wave time scale.
    fn authalic_radius(&self) -> f64 {
        (self.total_area() / (4.0 * std::f64::consts::PI)).sqrt()
    }

    /// Sine of the geodetic latitude, the component of the outward normal
    /// along the rotation axis. The Coriolis parameter is `2 omega` times this.
    fn sin_geodetic_latitude(&self, surface: Vec3) -> f64 {
        self.normal(surface).z
    }
}

/// A sphere of the given radius.
#[derive(Clone, Copy, Debug)]
pub struct Sphere {
    pub radius: f64,
}

impl Sphere {
    /// Earth's authalic radius, 6 371 007.181 m.
    pub const EARTH: Sphere = Sphere {
        radius: 6_371_007.181,
    };

    pub fn new(radius: f64) -> Self {
        Self { radius }
    }
}

impl Surface for Sphere {
    fn embed(&self, unit: Vec3) -> Vec3 {
        unit.normalized() * self.radius
    }

    fn project(&self, ambient: Vec3) -> Vec3 {
        ambient.normalized() * self.radius
    }

    fn normal(&self, surface: Vec3) -> Vec3 {
        surface.normalized()
    }

    fn total_area(&self) -> f64 {
        4.0 * std::f64::consts::PI * self.radius * self.radius
    }
}

/// An oblate spheroid, semi-major axis `a` in the equatorial plane and
/// semi-minor axis `b` along the rotation axis.
///
/// The parametrisation is the linear map `(x, y, z) -> (a x, a y, b z)` on the
/// unit sphere, so meridians and parallels of the icosahedral graph map to
/// meridians and parallels of the spheroid.
#[derive(Clone, Copy, Debug)]
pub struct Spheroid {
    pub a: f64,
    pub b: f64,
}

impl Spheroid {
    /// WGS 84: `a = 6 378 137 m`, inverse flattening `298.257223563`.
    pub const WGS84: Spheroid = Spheroid {
        a: 6_378_137.0,
        b: 6_378_137.0 * (1.0 - 1.0 / 298.257_223_563),
    };

    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    /// First eccentricity.
    pub fn eccentricity(&self) -> f64 {
        (1.0 - (self.b * self.b) / (self.a * self.a)).sqrt()
    }

    /// Flattening `(a - b) / a`.
    pub fn flattening(&self) -> f64 {
        (self.a - self.b) / self.a
    }
}

/// WGS 84, the reference ellipsoid of the Global Positioning System.
pub type Wgs84 = Spheroid;

impl Surface for Spheroid {
    fn embed(&self, unit: Vec3) -> Vec3 {
        let u = unit.normalized();
        Vec3::new(self.a * u.x, self.a * u.y, self.b * u.z)
    }

    fn project(&self, ambient: Vec3) -> Vec3 {
        // Pull back through the linear map, normalise on the sphere, push forward.
        let u = Vec3::new(ambient.x / self.a, ambient.y / self.a, ambient.z / self.b);
        self.embed(u)
    }

    fn normal(&self, surface: Vec3) -> Vec3 {
        let a2 = self.a * self.a;
        let b2 = self.b * self.b;
        Vec3::new(surface.x / a2, surface.y / a2, surface.z / b2).normalized()
    }

    fn total_area(&self) -> f64 {
        let e = self.eccentricity();
        if e < 1e-12 {
            return 4.0 * std::f64::consts::PI * self.a * self.a;
        }
        2.0 * std::f64::consts::PI * self.a * self.a * (1.0 + ((1.0 - e * e) / e) * e.atanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_normal_is_radial() {
        let s = Sphere::new(2.0);
        let p = s.embed(Vec3::new(1.0, 1.0, 1.0));
        assert!((p.norm() - 2.0).abs() < 1e-12);
        assert!((s.normal(p).dot(p.normalized()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn spheroid_reduces_to_sphere_when_unflattened() {
        let s = Spheroid::new(3.0, 3.0);
        assert!((s.total_area() - 4.0 * std::f64::consts::PI * 9.0).abs() < 1e-6);
        assert!(s.flattening().abs() < 1e-15);
    }

    #[test]
    fn wgs84_area_matches_published_value() {
        // 5.100 656 217 e14 m^2, the standard WGS 84 surface area.
        let area = Wgs84::WGS84.total_area();
        assert!(
            (area - 5.100_656_217e14).abs() / 5.100_656_217e14 < 1e-9,
            "WGS84 area was {area:e}"
        );
    }

    #[test]
    fn geodetic_latitude_exceeds_geocentric_off_the_axes() {
        let s = Wgs84::WGS84;
        let p = s.embed(Vec3::new(1.0, 0.0, 1.0));
        let geodetic = s.sin_geodetic_latitude(p).asin().to_degrees();
        let geocentric = (p.z / p.norm()).asin().to_degrees();
        assert!(geodetic > geocentric);
        // The gap peaks near 45 degrees at about 0.19 degrees.
        assert!(
            (geodetic - geocentric - 0.19).abs() < 0.01,
            "gap {}",
            geodetic - geocentric
        );
    }

    #[test]
    fn projection_lands_on_the_surface() {
        let s = Wgs84::WGS84;
        let p = s.project(Vec3::new(1.0e6, -2.0e6, 3.0e6));
        let f = (p.x * p.x + p.y * p.y) / (s.a * s.a) + (p.z * p.z) / (s.b * s.b);
        assert!((f - 1.0).abs() < 1e-12, "off-surface by {}", f - 1.0);
    }
}
