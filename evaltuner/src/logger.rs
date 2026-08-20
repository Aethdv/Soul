use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    path::Path,
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::Serialize;

pub struct JsonLogger {
    file: Mutex<File>,
}

impl JsonLogger {
    /// # Errors
    /// Returns an error if the file cannot be created or opened.
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file: Mutex::new(file) })
    }

    pub fn log<T: Serialize>(&self, event: &str, data: &T) {
        #[derive(Serialize)]
        struct LogEntry<'a, T> {
            timestamp: u64,
            event: &'a str,
            #[serde(flatten)]
            data: &'a T,
        }

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let entry = LogEntry { timestamp, event, data };

        if let Ok(mut json) = serde_json::to_string(&entry)
            && let Ok(mut file) = self.file.lock()
        {
            json.push('\n');
            file.write_all(json.as_bytes()).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_event_is_one_readable_line() {
        let path = std::env::temp_dir().join(format!("soul_log_test_{}.jsonl", std::process::id()));
        let logger = JsonLogger::new(&path).unwrap();
        logger.log("epoch", &serde_json::json!({ "n": 1 }));
        logger.log("final", &serde_json::json!({ "n": 2 }));
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let events: Vec<serde_json::Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["event"], "epoch");
        assert_eq!(events[1]["event"], "final");
    }
}
