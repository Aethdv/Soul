/// Bounds-checked in debug; `$index` must be in bounds in release.
#[macro_export]
macro_rules! debug_index {
    ($collection:expr, $index:expr) => {{
        #[cfg(debug_assertions)]
        {
            &$collection[$index]
        }

        #[cfg(not(debug_assertions))]
        {
            // SAFETY: $index is required to be in bounds.
            #[allow(unused_unsafe)]
            unsafe {
                $collection.get_unchecked($index)
            }
        }
    }};
}

#[macro_export]
macro_rules! soul_index_checked {
    ($type:ty, $size:expr) => {
        impl<T> std::ops::Index<$type> for [T; $size] {
            type Output = T;

            #[inline(always)]
            fn index(&self, idx: $type) -> &Self::Output {
                let i = usize::from(idx);
                assert!(i < $size, "Domain-typed index out of bounds");
                &self[i]
            }
        }

        impl<T> std::ops::IndexMut<$type> for [T; $size] {
            #[inline(always)]
            fn index_mut(&mut self, idx: $type) -> &mut Self::Output {
                let i = usize::from(idx);
                assert!(i < $size, "Domain-typed index out of bounds");
                &mut self[i]
            }
        }
    };
}

/// Bounds-checked in debug; `$index` must be in bounds in release.
#[macro_export]
macro_rules! debug_index_mut {
    ($collection:expr, $index:expr) => {{
        #[cfg(debug_assertions)]
        {
            &mut $collection[$index]
        }

        #[cfg(not(debug_assertions))]
        {
            // Pointer arithmetic can avoid sibling-field reloads here.
            // SAFETY: $index is required to be in bounds.
            #[allow(unused_unsafe)]
            unsafe {
                &mut *$collection.as_mut_ptr().add($index)
            }
        }
    }};
}
