//! Rows of varying length in one contiguous block.
//!
//! A cell has five or six edges, so the incidence arrays are ragged. Holding
//! them as `Vec<Vec<_>>` costs one heap allocation per cell and scatters the
//! rows, which the tendency loop then walks once per stage. Packing the
//! values end to end with an offset table keeps a sweep over every cell inside
//! one contiguous read.

/// Rows of `T`, packed end to end.
#[derive(Clone, Debug)]
pub struct Ragged<T> {
    values: Vec<T>,
    offsets: Vec<u32>,
}

impl<T> Ragged<T> {
    /// Pack a list of rows.
    pub fn from_rows(rows: &[Vec<T>]) -> Ragged<T>
    where
        T: Clone,
    {
        let mut values = Vec::with_capacity(rows.iter().map(|r| r.len()).sum());
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0);
        for row in rows {
            values.extend_from_slice(row);
            offsets.push(values.len() as u32);
        }
        Ragged { values, offsets }
    }

    /// Rows of the given lengths, every entry `fill`.
    pub fn filled(lengths: impl Iterator<Item = usize>, fill: T) -> Ragged<T>
    where
        T: Clone,
    {
        let mut offsets = vec![0u32];
        for n in lengths {
            offsets.push(offsets.last().unwrap() + n as u32);
        }
        let total = *offsets.last().unwrap() as usize;
        Ragged {
            values: vec![fill; total],
            offsets,
        }
    }

    pub fn rows(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn row(&self, i: usize) -> &[T] {
        &self.values[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.values[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    pub fn row_len(&self, i: usize) -> usize {
        (self.offsets[i + 1] - self.offsets[i]) as usize
    }

    /// Every value, in row order.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Index into [`Ragged::values`] where row `i` starts.
    pub fn start(&self, i: usize) -> usize {
        self.offsets[i] as usize
    }

    pub fn iter_rows(&self) -> impl Iterator<Item = &[T]> {
        (0..self.rows()).map(move |i| self.row(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_come_back_as_they_went_in() {
        let rows = vec![vec![1u32, 2, 3], vec![4], vec![], vec![5, 6]];
        let r = Ragged::from_rows(&rows);
        assert_eq!(r.rows(), 4);
        for (i, want) in rows.iter().enumerate() {
            assert_eq!(r.row(i), want.as_slice());
            assert_eq!(r.row_len(i), want.len());
        }
        assert_eq!(r.values(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(r.start(3), 4);
    }

    #[test]
    fn filled_rows_take_the_lengths_given() {
        let mut r = Ragged::filled([2usize, 0, 3].into_iter(), 0.0);
        assert_eq!(r.rows(), 3);
        r.row_mut(2)[1] = 7.0;
        assert_eq!(r.row(2), &[0.0, 7.0, 0.0]);
        assert_eq!(r.values().len(), 5);
    }
}
