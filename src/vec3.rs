//! A three-component vector, so the crate needs no linear-algebra dependency.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Vec3 {
        let n = self.norm();
        if n == 0.0 { self } else { self / n }
    }

    /// Component of `self` orthogonal to `n`, with `n` a unit vector.
    pub fn tangent_to(self, n: Vec3) -> Vec3 {
        self - n * self.dot(n)
    }

    /// Area of the triangle spanned by the three points.
    pub fn triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f64 {
        0.5 * (b - a).cross(c - a).norm()
    }

    /// Circumcentre of the triangle, in the plane the three points span.
    pub fn circumcenter(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
        let ab = b - a;
        let ac = c - a;
        let n = ab.cross(ac);
        let denom = 2.0 * n.dot(n);
        if denom == 0.0 {
            return (a + b + c) / 3.0;
        }
        let num = n.cross(ab) * ac.dot(ac) + ac.cross(n) * ab.dot(ab);
        a + num / denom
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f64) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circumcenter_is_equidistant() {
        let a = Vec3::new(0.0, 0.0, 1.0);
        let b = Vec3::new(3.0, 0.0, 1.0);
        let c = Vec3::new(0.0, 4.0, 1.0);
        let o = Vec3::circumcenter(a, b, c);
        let ra = (o - a).norm();
        assert!(((o - b).norm() - ra).abs() < 1e-12);
        assert!(((o - c).norm() - ra).abs() < 1e-12);
        assert!((ra - 2.5).abs() < 1e-12, "circumradius {ra}");
    }

    #[test]
    fn triangle_area_matches_the_half_base_times_height() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(4.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 3.0, 0.0);
        assert!((Vec3::triangle_area(a, b, c) - 6.0).abs() < 1e-12);
    }
}
