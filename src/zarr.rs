//! Zarr v3 output, written directly.
//!
//! A Zarr v3 store is a directory of JSON metadata and raw chunk files, so a
//! writer for the one case this crate needs, little-endian `float64` with one
//! chunk per output time, is short enough to keep in-tree and leaves the
//! crate with no dependencies. What comes out opens in `zarr-python` and, with
//! the dimension names and the `coordinates` attribute below, in `xarray`
//! with longitude and latitude already attached.
//!
//! Chunks are stored uncompressed. Adding a codec would mean adding a
//! compression dependency, and the trade is not worth it for output a
//! post-processing step reads once.

use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// One entry in a Zarr attributes object.
pub struct Attribute {
    key: String,
    value: String,
}

impl Attribute {
    pub fn text(key: &str, value: &str) -> Attribute {
        Attribute {
            key: key.into(),
            value: format!("\"{}\"", escape(value)),
        }
    }

    pub fn number(key: &str, value: f64) -> Attribute {
        Attribute {
            key: key.into(),
            value: format!("{value}"),
        }
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn object(attributes: &[Attribute]) -> String {
    let mut out = String::from("{");
    for (i, a) in attributes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "\"{}\":{}", escape(&a.key), a.value);
    }
    out.push('}');
    out
}

/// An open store. Arrays are declared once, then appended to a time slice at
/// a time.
pub struct Store {
    root: PathBuf,
}

/// One array in the store, with its time cursor.
pub struct Array {
    path: PathBuf,
    len: usize,
    steps: usize,
    written: usize,
}

impl Store {
    /// Create or truncate a store at `root`.
    pub fn create(root: impl AsRef<Path>, attributes: &[Attribute]) -> io::Result<Store> {
        let root = root.as_ref().to_path_buf();
        if root.exists() {
            fs::remove_dir_all(&root)?;
        }
        fs::create_dir_all(&root)?;
        let meta = format!(
            r#"{{"zarr_format":3,"node_type":"group","attributes":{}}}"#,
            object(attributes)
        );
        fs::write(root.join("zarr.json"), meta)?;
        Ok(Store { root })
    }

    /// Declare a `(time, len)` array of `float64`, chunked one time step deep.
    pub fn array(
        &self,
        name: &str,
        len: usize,
        steps: usize,
        dim: &str,
        units: &str,
        long_name: &str,
    ) -> io::Result<Array> {
        let attributes = [
            Attribute::text("units", units),
            Attribute::text("long_name", long_name),
            Attribute::text("coordinates", "lon lat"),
        ];
        let path = self.root.join(name);
        fs::create_dir_all(&path)?;
        fs::write(
            path.join("zarr.json"),
            array_metadata(&[steps, len], &[1, len], &["time", dim], &attributes),
        )?;
        Ok(Array {
            path,
            len,
            steps,
            written: 0,
        })
    }

    /// Declare a one-dimensional coordinate array and write it whole.
    pub fn coordinate(&self, name: &str, dim: &str, units: &str, values: &[f64]) -> io::Result<()> {
        let len = values.len();
        let attributes = [Attribute::text("units", units)];
        let path = self.root.join(name);
        fs::create_dir_all(&path)?;
        fs::write(
            path.join("zarr.json"),
            array_metadata(&[len], &[len], &[dim], &attributes),
        )?;
        let dir = path.join("c");
        fs::create_dir_all(&dir)?;
        write_f64(&dir.join("0"), values)
    }
}

fn array_metadata(
    shape: &[usize],
    chunk: &[usize],
    dims: &[&str],
    attributes: &[Attribute],
) -> String {
    let list = |v: &[usize]| {
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let names = dims
        .iter()
        .map(|d| format!("\"{d}\""))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            r#"{{"zarr_format":3,"node_type":"array","shape":[{}],"data_type":"float64","#,
            r#""chunk_grid":{{"name":"regular","configuration":{{"chunk_shape":[{}]}}}},"#,
            r#""chunk_key_encoding":{{"name":"default","configuration":{{"separator":"/"}}}},"#,
            r#""fill_value":0.0,"#,
            r#""codecs":[{{"name":"bytes","configuration":{{"endian":"little"}}}}],"#,
            r#""attributes":{},"dimension_names":[{}]}}"#
        ),
        list(shape),
        list(chunk),
        object(attributes),
        names
    )
}

impl Array {
    /// Append one time slice.
    pub fn push(&mut self, values: &[f64]) -> io::Result<()> {
        assert_eq!(values.len(), self.len, "slice length must match the array");
        assert!(
            self.written < self.steps,
            "more slices pushed than the array was declared"
        );
        let dir = self.path.join("c").join(self.written.to_string());
        fs::create_dir_all(&dir)?;
        write_f64(&dir.join("0"), values)?;
        self.written += 1;
        Ok(())
    }

    pub fn steps_written(&self) -> usize {
        self.written
    }
}

fn write_f64(path: &Path, values: &[f64]) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::File::create(path)?.write_all(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_store_lays_out_the_files_zarr_v3_expects() {
        let root = std::env::temp_dir().join("charney-zarr-test");
        let store = Store::create(
            &root,
            &[
                Attribute::text("title", "test"),
                Attribute::number("level", 3.0),
            ],
        )
        .unwrap();
        store
            .coordinate("lat", "cell", "radians_north", &[0.1, 0.2, 0.3])
            .unwrap();
        let mut h = store
            .array("h", 3, 2, "cell", "m", "fluid thickness")
            .unwrap();
        h.push(&[1.0, 2.0, 3.0]).unwrap();
        h.push(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(h.steps_written(), 2);

        assert!(root.join("zarr.json").exists());
        assert!(root.join("lat/c/0").exists());
        assert!(root.join("h/c/0/0").exists());
        assert!(root.join("h/c/1/0").exists());

        let raw = fs::read(root.join("h/c/1/0")).unwrap();
        assert_eq!(raw.len(), 24);
        assert_eq!(f64::from_le_bytes(raw[0..8].try_into().unwrap()), 4.0);
        assert_eq!(f64::from_le_bytes(raw[16..24].try_into().unwrap()), 6.0);

        let meta = fs::read_to_string(root.join("h/zarr.json")).unwrap();
        assert!(meta.contains(r#""zarr_format":3"#));
        assert!(meta.contains(r#""shape":[2,3]"#));
        assert!(meta.contains(r#""dimension_names":["time","cell"]"#));
        assert!(meta.contains(r#""units":"m""#));
        assert!(meta.contains(r#""coordinates":"lon lat""#));

        let group = fs::read_to_string(root.join("zarr.json")).unwrap();
        assert!(group.contains(r#""title":"test""#));
        assert!(group.contains(r#""level":3"#));

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn quotes_in_an_attribute_do_not_break_the_json() {
        let a = Attribute::text("note", r#"a "quoted" word"#);
        let json = object(&[a]);
        assert_eq!(json, r#"{"note":"a \"quoted\" word"}"#);
    }
}
