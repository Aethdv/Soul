use std::io::Error;

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("I/O error: {0}")]
    Io(#[from] Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
