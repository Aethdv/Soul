//! Safe abstraction macros for debug-checked performance invariants.

/// Unchecked indexing in release, bounds-checked in debug.
///
/// Usage: `debug_index!(collection, index)`
#[macro_export]
macro_rules! debug_index {
    ($collection:expr, $index:expr) => {{
        #[cfg(debug_assertions)]
        {
            &$collection[$index]
        }

        #[cfg(not(debug_assertions))]
        {
            #[allow(unused_unsafe)]
            // SAFETY: Caller guarantees that $index is within the bounds of $collection
            unsafe {
                $collection.get_unchecked($index)
            }
        }
    }};
}

/// Unchecked mutated indexing in release, bounds-checked in debug.
///
/// Why not `get_unchecked_mut`?
/// Both produce a `&mut T`, but this macro derives the reference through `as_mut_ptr()`
/// (an immutable borrow of the field to obtain a raw pointer) followed by pointer arithmetic
/// and dereference. LLVM sees the `&mut T` as originating from a local pointer,
/// not directly from a field borrow, which can avoid forcing reloads of sibling fields on
/// every call, though whether this optimization fires depends on LLVM's aliasing analysis.
///
/// Usage: `debug_index_mut!(collection, index)`
#[macro_export]
macro_rules! debug_index_mut {
    ($collection:expr, $index:expr) => {{
        #[cfg(debug_assertions)]
        {
            &mut $collection[$index]
        }

        #[cfg(not(debug_assertions))]
        {
            #[allow(unused_unsafe)]
            // SAFETY: Caller guarantees that $index is within the bounds of $collection.
            unsafe {
                &mut *$collection.as_mut_ptr().add($index)
            }
        }
    }};
}

/// Unchecked swap in release, bounds-checked swap in debug.
///
/// Usage: `debug_swap!(collection, i, j)`
#[macro_export]
macro_rules! debug_swap {
    ($collection:expr, $i:expr, $j:expr) => {{
        #[cfg(debug_assertions)]
        {
            $collection.swap($i, $j)
        }

        #[cfg(not(debug_assertions))]
        {
            #[allow(unused_unsafe)]
            // SAFETY: Caller guarantees that $i and $j are within the bounds of $collection
            unsafe {
                $collection.swap_unchecked($i, $j)
            }
        }
    }};
}
