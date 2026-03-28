use std::{
    fs::{File, OpenOptions},
    io::{self, BufWriter, Write},
    path::Path,
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::Serialize;

pub struct JsonLogger {
    writer: Mutex<BufWriter<File>>,
}

impl JsonLogger {
    /// # Errors
    /// if log file cannot be created/opened.
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;

        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }

    /// # Panics
    /// if system time is before UNIX EPOCH (should be impossible).
    pub fn log<T: Serialize>(&self, event: &str, data: &T) {
        #[derive(Serialize)]
        struct LogEntry<'a, T> {
            timestamp: u64,
            event:     &'a str,
            #[serde(flatten)]
            data:      &'a T,
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entry = LogEntry {
            timestamp,
            event,
            data,
        };

        if let Ok(json) = serde_json::to_string(&entry)
            && let Ok(mut writer) = self.writer.lock()
        {
            writeln!(writer, "{json}").ok();
            writer.flush().ok();
        }
    }
}
