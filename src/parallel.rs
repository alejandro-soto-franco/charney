//! Chunked parallelism, behind the `parallel` feature.
//!
//! Without the feature the crate has no dependencies and every loop here runs
//! on the calling thread. With it, `rayon`'s pool splits the same loops.
//!
//! A pool is the only workable shape at this granularity. One `RK4` stage at
//! level 6 takes a couple of milliseconds and divides into a handful of passes
//! over the edges, the cells and the duals, and `std::thread::scope` charges
//! about sixteen microseconds a thread to enter a region, which on sixteen
//! threads costs more than the pass it would parallelise.

/// Shortest slice worth splitting. Below this the pool costs more than the
/// pass, which is why a coarse mesh runs serially even with the feature on.
#[cfg(feature = "parallel")]
const MIN_PARALLEL: usize = 1 << 16;

/// Smallest chunk worth handing to a thread. A pass over the edges does about
/// ten indexed reads an entry, so a few thousand entries is where the work
/// covers a task's overhead.
#[cfg(feature = "parallel")]
const MIN_CHUNK: usize = 2048;

#[cfg(feature = "parallel")]
fn chunk_size(len: usize) -> usize {
    len.div_ceil(rayon::current_num_threads()).max(MIN_CHUNK)
}

/// Run `f(start, chunk)` over contiguous chunks of `out`, where `start` is the
/// index in `out` the chunk begins at.
#[cfg(feature = "parallel")]
pub fn chunks<T, F>(out: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    use rayon::prelude::*;
    if out.len() < MIN_PARALLEL {
        f(0, out);
        return;
    }
    let size = chunk_size(out.len());
    out.par_chunks_mut(size)
        .enumerate()
        .for_each(|(i, part)| f(i * size, part));
}

#[cfg(not(feature = "parallel"))]
pub fn chunks<T, F>(out: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    f(0, out);
}

/// The same over two output slices of equal length at once.
#[cfg(feature = "parallel")]
pub fn zip_chunks<A, B, F>(a: &mut [A], b: &mut [B], f: F)
where
    A: Send,
    B: Send,
    F: Fn(usize, &mut [A], &mut [B]) + Sync + Send,
{
    use rayon::prelude::*;
    assert_eq!(a.len(), b.len());
    if a.len() < MIN_PARALLEL {
        f(0, a, b);
        return;
    }
    let size = chunk_size(a.len());
    a.par_chunks_mut(size)
        .zip(b.par_chunks_mut(size))
        .enumerate()
        .for_each(|(i, (x, y))| f(i * size, x, y));
}

#[cfg(not(feature = "parallel"))]
pub fn zip_chunks<A, B, F>(a: &mut [A], b: &mut [B], f: F)
where
    A: Send,
    B: Send,
    F: Fn(usize, &mut [A], &mut [B]) + Sync + Send,
{
    assert_eq!(a.len(), b.len());
    f(0, a, b);
}

/// Threads the pool will use, or one when the feature is off.
pub fn threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_every_index_once() {
        for len in [0usize, 1, 7, 1000, 5000] {
            let mut v = vec![0usize; len];
            chunks(&mut v, |start, part| {
                for (i, x) in part.iter_mut().enumerate() {
                    *x = start + i;
                }
            });
            assert!(v.iter().enumerate().all(|(i, &x)| i == x), "len {len}");
        }
    }

    #[test]
    fn zipped_chunks_stay_aligned() {
        let mut a = vec![0usize; 3000];
        let mut b = vec![0usize; 3000];
        zip_chunks(&mut a, &mut b, |start, x, y| {
            for i in 0..x.len() {
                x[i] = start + i;
                y[i] = 2 * (start + i);
            }
        });
        assert!(
            a.iter()
                .zip(&b)
                .enumerate()
                .all(|(i, (&x, &y))| x == i && y == 2 * i)
        );
    }
}
