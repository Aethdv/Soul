//! Single-producer, multi-consumer broadcast channel.

use std::{
    ptr,
    sync::{
        Arc,
        atomic::{
            AtomicBool, AtomicPtr, AtomicU32,
            Ordering::{Acquire, Relaxed, Release},
        },
    },
};

use atomic_wait::{wait, wake_all};

pub(super) struct Sender<M> {
    shared: Arc<Shared<M>>,
}

pub(super) struct Receiver<M> {
    shared: Arc<Shared<M>>,
    generation: bool,
}

struct Shared<M> {
    msg: AtomicPtr<M>,
    ready: AtomicBool,
    futex: AtomicU32,
    exit: AtomicBool,
    num_receivers: u32,
}

pub(crate) fn channel<M>(num_receivers: u32) -> (Sender<M>, Vec<Receiver<M>>) {
    let shared = Arc::new(Shared {
        msg: AtomicPtr::new(ptr::null_mut()),
        ready: AtomicBool::new(false),
        futex: AtomicU32::new(pack(0, false)),
        exit: AtomicBool::new(false),
        num_receivers,
    });

    let tx = Sender { shared: Arc::clone(&shared) };
    let rxs = (0..num_receivers)
        .map(|_| Receiver { shared: Arc::clone(&shared), generation: true })
        .collect();

    (tx, rxs)
}

impl<M> Sender<M> {
    pub(crate) fn send(&self, msg: M) {
        let shared = &*self.shared;
        let boxed = Box::new(msg);
        shared.msg.store(Box::into_raw(boxed), Relaxed);
        shared.ready.store(true, Release);

        let (_, generation) = unpack(shared.futex.load(Relaxed));
        shared.futex.store(pack(shared.num_receivers, !generation), Release);
        wake_all(&shared.futex);
    }

    /// Waits for every handler to return, then frees the message.
    pub(super) fn wait(&self) {
        let shared = &*self.shared;
        let mut val = shared.futex.load(Acquire);

        while unpack(val).0 != 0 {
            wait(&shared.futex, val);
            val = shared.futex.load(Acquire);
        }

        shared.ready.store(false, Release);

        let ptr = shared.msg.swap(ptr::null_mut(), Relaxed);
        if !ptr.is_null() {
            // SAFETY: send created this with Box::into_raw, and zero outstanding
            // receivers means no handler still borrows it.
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }

    /// Shuts the channel down and wakes every receiver.
    pub(super) fn wake(&self) {
        let shared = &*self.shared;
        shared.exit.store(true, Release);
        wake_all(&shared.futex);
    }
}

impl<M> Receiver<M> {
    /// Waits for the next unseen message, or returns `None` after shutdown.
    pub(super) fn recv<R>(&mut self, handler: impl FnOnce(&M) -> R) -> Option<R> {
        let shared = &*self.shared;

        loop {
            let val = shared.futex.load(Acquire);

            if shared.exit.load(Acquire) {
                return None;
            }

            if shared.ready.load(Acquire) && unpack(val).1 == self.generation {
                break;
            }
            wait(&shared.futex, val);
        }

        self.generation = !self.generation;

        // SAFETY: send publishes the pointer with the generation's Release store,
        // acquired above; wait cannot free it until this handler returns.
        let msg_ref = unsafe { &*shared.msg.load(Relaxed) };
        let result = handler(msg_ref);
        let (outstanding, _) = unpack(shared.futex.fetch_sub(1, Release));
        debug_assert!(outstanding > 0, "a decrement past zero borrows into the generation bit");

        if outstanding == 1 {
            wake_all(&shared.futex);
        }

        Some(result)
    }
}

// SAFETY: Single-producer design. Each receiver is used from one thread.
// The msg pointer is guarded by the futex + generation protocol.
unsafe impl<M: Send> Send for Sender<M> {}
unsafe impl<M: Sync> Sync for Sender<M> {}
unsafe impl<M: Send> Send for Receiver<M> {}
unsafe impl<M: Sync> Sync for Receiver<M> {}

fn pack(remaining: u32, generation: bool) -> u32 {
    debug_assert!(remaining < u32::MAX >> 1);
    remaining | (generation as u32) << 31
}

fn unpack(v: u32) -> (u32, bool) { (v & (u32::MAX >> 1), (v >> 31) != 0) }
