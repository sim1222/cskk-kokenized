//! libcskk用の最小ロガー。
//!
//! IMEプロセス (fcitx5) にはRustのロガーが存在しないため独自にログを出す:
//! - 標準エラー出力: info レベル (環境変数 CSKK_LOG で変更可)
//! - **ログファイル (debug レベル)**: `~/.local/share/libcskk/cskk.log`
//!   (開発中の既定動作。2MB超過で切り詰め)
//!
//! 環境変数 CSKK_LOG_FILE でログファイルのパスを、
//! CSKK_LOG_FILE=off でファイル出力を無効化できる。

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// ログファイルの最大サイズ (超過で切り詰め)。
const MAX_LOG_SIZE: u64 = 2 * 1024 * 1024;

struct MiniLogger {
    stderr_level: log::LevelFilter,
    file: Mutex<Option<File>>,
}

impl log::Log for MiniLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        // ファイルはdebugまで、stderrはstderr_levelまで
        metadata.level() <= log::Level::Debug
            && (metadata.level() <= self.stderr_level || !self.file_is_disabled())
    }

    fn log(&self, record: &log::Record) {
        let level = record.level();
        if level <= self.stderr_level {
            eprintln!("[cskk {}] {}", level, record.args());
        }
        // ファイルにはdebugレベルまで常に書く (開発用)
        if level <= log::LevelFilter::Debug {
            if let Ok(mut guard) = self.file.lock() {
                if let Some(file) = guard.as_mut() {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default();
                    let _ = writeln!(
                        file,
                        "[{}.{:03} {}] {}",
                        now.as_secs(),
                        now.subsec_millis(),
                        level,
                        record.args()
                    );
                    let _ = file.flush();
                }
            }
        }
    }

    fn flush(&self) {
        if let Ok(mut guard) = self.file.lock() {
            if let Some(file) = guard.as_mut() {
                let _ = file.flush();
            }
        }
    }
}

impl MiniLogger {
    fn file_is_disabled(&self) -> bool {
        self.file.lock().map(|g| g.is_none()).unwrap_or(true)
    }
}

static INSTALL: OnceLock<()> = OnceLock::new();

fn default_log_file_path() -> Option<PathBuf> {
    let data_home = std::env::var("XDG_DATA_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|home| PathBuf::from(home).join(".local/share"))
        })?;
    Some(data_home.join("libcskk").join("cskk.log"))
}

fn open_log_file() -> Option<File> {
    // 環境変数で明示的にオフにできる
    if let Ok(value) = std::env::var("CSKK_LOG_FILE") {
        if value == "off" {
            return None;
        }
    }
    let path = std::env::var("CSKK_LOG_FILE")
        .ok()
        .map(PathBuf::from)
        .or_else(default_log_file_path)?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // 容量超過時は切り詰めて頭出しする
    let truncate = std::fs::metadata(&path)
        .map(|m| m.len() > MAX_LOG_SIZE)
        .unwrap_or(false);
    OpenOptions::new()
        .append(!truncate)
        .truncate(truncate)
        .create(true)
        .open(&path)
        .ok()
}

/// ロガーをインストールする (複数回呼んでも安全)。
pub(crate) fn install_logger() {
    INSTALL.get_or_init(|| {
        // stderrは既定info、ファイルはdebug
        let stderr_level = std::env::var("CSKK_LOG")
            .ok()
            .and_then(|v| v.parse::<log::LevelFilter>().ok())
            .unwrap_or(log::LevelFilter::Info);
        let file = open_log_file();
        if file.is_some() {
            let path = std::env::var("CSKK_LOG_FILE")
                .unwrap_or_else(|_| {
                    default_log_file_path()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_default()
                });
            eprintln!("[cskk] debug log file: {}", path);
        }
        let _ = log::set_boxed_logger(Box::new(MiniLogger {
            stderr_level,
            file: Mutex::new(file),
        }));
        log::set_max_level(log::LevelFilter::Debug);
        eprintln!("[cskk] logger installed (stderr = {})", stderr_level);
    });
}