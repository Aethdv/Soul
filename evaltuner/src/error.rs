use std::io::Error;

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("I/O error: {0}")]
    Io(#[from] Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Layout renamed parameters: {0}. Start fresh, or check out the layout that wrote it")]
    LayoutRenamed(String),
}
