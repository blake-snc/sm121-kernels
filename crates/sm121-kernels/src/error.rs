use std::fmt;

/// Error type for all fallible operations in this crate.
#[derive(Debug)]
pub enum SparkError {
    /// A requested kernel entry point was not found in the embedded cubins.
    KernelNotFound(String),
    /// The GPU is not the supported SM121 target; carries the detected
    /// compute-capability `major`.`minor`.
    UnsupportedArch {
        /// Detected compute-capability major version.
        major: u32,
        /// Detected compute-capability minor version.
        minor: u32,
    },
    /// An error returned by the underlying CUDA driver via `cudarc`.
    Driver(cudarc::driver::DriverError),
    /// A caller-supplied argument failed validation (bad dims, undersized buffer, etc.).
    InvalidArgument(String),
    /// A kernel launch returned a non-success `CUresult`.
    LaunchFailed(String),
    /// An I/O error, typically while loading or saving adapter files.
    Io(String),
    /// Any other error described by the contained message.
    Other(String),
}

impl fmt::Display for SparkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelNotFound(name) => write!(f, "kernel not found: {name}"),
            Self::UnsupportedArch { major, minor } => write!(
                f,
                "unsupported GPU architecture: compute capability {major}.{minor}, expected 12.1 (SM121)"
            ),
            Self::Driver(e) => write!(f, "CUDA driver error: {e:?}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::LaunchFailed(msg) => write!(f, "kernel launch failed: {msg}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for SparkError {}

impl From<cudarc::driver::DriverError> for SparkError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Driver(e)
    }
}

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, SparkError>;
