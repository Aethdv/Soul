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
            // Some call sites expand inside an unsafe block, where this one is redundant.
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
/// The `&mut T` comes from `as_mut_ptr()` and pointer arithmetic rather than
/// `get_unchecked_mut`, so LLVM sees it originating from a local pointer instead of a field
/// borrow. That can spare reloads of sibling fields, where the aliasing analysis cooperates.
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
            // Some call sites expand inside an unsafe block, where this one is redundant.
            #[allow(unused_unsafe)]
            // SAFETY: Caller guarantees that $index is within the bounds of $collection.
            unsafe {
                &mut *$collection.as_mut_ptr().add($index)
            }
        }
    }};
}
