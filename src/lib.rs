//! Rotating shallow water on a sphere and on an Earth-fitted oblate spheroid.

pub mod dec;
pub mod geometry;
pub mod mesh;
pub mod swe;
pub mod trisk;
pub mod vec3;
pub mod williamson;
pub mod zarr;

pub use geometry::{Sphere, Spheroid, Surface, Wgs84};
pub use mesh::Mesh;
pub use swe::{Model, State};
pub use vec3::Vec3;
