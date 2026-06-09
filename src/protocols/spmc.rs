//! Non-blocking single-producer multi-consumer broadcast channel.
//!
//! `send(msg)` heap-allocates one message, wakes every receiver, and returns.
//! `recv(handler)` blocks until a message arrives, runs the handler with a
//! shared reference to it, then automatically signals completion (decrements
//! the outstanding-receiver count). When every receiver has returned from
//! `recv`, the sender's `wait()` unblocks and the message is deallocated.
//!
//! A generation bit toggles each `send` so receivers don't pick up stale
//! messages between `wait` and the next `send`. An `exit` flag shuts down
//! receivers cleanly — `wake()` sets the flag and unparks every receiver,
//! at which point `recv` returns `None`.

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

struct Shared<M> {
    msg: AtomicPtr<M>,
    ready: AtomicBool,
    futex: AtomicU32,
    exit: AtomicBool,
    num_receivers: u32,
}

pub struct Sender<M> {
    shared: Arc<Shared<M>>,
}

pub struct Receiver<M> {
    shared: Arc<Shared<M>>,
    generation: bool,
}

fn pack(remaining: u32, generation: bool) -> u32 {
    debug_assert!(remaining < u32::MAX >> 1);
    remaining | (generation as u32) << 31
}

fn unpack(v: u32) -> (u32, bool) {
    (v & (u32::MAX >> 1), (v >> 31) != 0)
}

pub fn channel<M>(num_receivers: u32) -> (Sender<M>, Vec<Receiver<M>>) {
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
    /// Stores a heap-allocated `msg`, toggles the generation, wakes every
    /// receiver, and returns. The message is freed in `wait()`.
    pub fn send(&self, msg: M) {
        let shared = &*self.shared;

        let boxed = Box::new(msg);
        shared.msg.store(Box::into_raw(boxed), Relaxed);
        shared.ready.store(true, Release);

        let (_, generation) = unpack(shared.futex.load(Relaxed));
        shared.futex.store(pack(shared.num_receivers, !generation), Release);
        wake_all(&shared.futex);
    }

    /// Blocks until every receiver has returned from `recv`, then frees the
    /// message so the next `send()` can allocate fresh.
    pub fn wait(&self) {
        let shared = &*self.shared;

        let mut val = shared.futex.load(Acquire);

        while unpack(val).0 != 0 {
            wait(&shared.futex, val);
            val = shared.futex.load(Acquire);
        }

        shared.ready.store(false, Release);

        let ptr = shared.msg.swap(ptr::null_mut(), Relaxed);

        if !ptr.is_null() {
            // SAFETY: Created by Box::into_raw in send(); all receivers
            // have returned from recv (outstanding count is zero).
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }

    /// Signals all receivers to exit and unparks them. After this call,
    /// every parked `recv()` returns `None` and receivers should terminate.
    pub fn wake(&self) {
        let shared = &*self.shared;
        shared.exit.store(true, Release);
        wake_all(&shared.futex);
    }
}

impl<M> Receiver<M> {
    /// Blocks until `send()` delivers a message with a fresh generation,
    /// runs `handler` on a shared reference to it, automatically decrements
    /// the outstanding-receiver count, and returns `Some(handler_result)`.
    ///
    /// Returns `None` if the channel has been shut down via `Sender::wake()`.
    pub fn recv<R>(&mut self, handler: impl FnOnce(&M) -> R) -> Option<R> {
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

        // SAFETY: The Acquire loads above establish that send()'s
        // msg store (happens-before the Release store on ready/futex)
        // is visible here.
        let msg_ref = unsafe { &*shared.msg.load(Relaxed) };
        let result = handler(msg_ref);

        // Decrement outstanding. If this was the last receiver, wake the
        // sender (parked in wait()).
        if unpack(shared.futex.fetch_sub(1, Release)).0 == 1 {
            wake_all(&shared.futex);
        }

        Some(result)
    }
}

// SAFETY: Single-producer design. Each receiver is used from one thread.
// The msg pointer is guarded by the futex + generation protocol.
unsafe impl<M: Send> Send for Sender<M> {}
unsafe impl<M: Send> Send for Receiver<M> {}
unsafe impl<M: Sync> Sync for Sender<M> {}
unsafe impl<M: Sync> Sync for Receiver<M> {}
