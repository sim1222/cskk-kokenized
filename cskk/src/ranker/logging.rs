//! libcskk用の最小ロガー。
//!
//! IMEプロセス (fcitx5) にはRustのロガーが存在しないため、
//! 環境変数 CSKK_LOG (debug/info/warn/error) が設定されている場合に
//! 標準エラー出力へログを出す。未設定の場合は warn 以上のみ出力する。
//!
//! 使い方:
//!   CSKK_LOG=info fcitx5 -r   # 端末でfcitx5を起動するとログが見える

use std::sync::OnceLock;

struct MiniLogger {
    max_level: log::LevelFilter,
}

impl log::Log for MiniLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= self.max_level
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("[cskk {}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: OnceLock<MiniLogger> = OnceLock::new();

/// ロガーをインストールする (複数回呼んでも安全)。
pub(crate) fn install_logger() {
    LOGGER.get_or_init(|| {
        let level = std::env::var("CSKK_LOG")
            .ok()
            .and_then(|v| v.parse::<log::LevelFilter>().ok())
            .unwrap_or(log::LevelFilter::Warn);
        let logger = MiniLogger { max_level: level };
        // 先に他のロガーが入っている場合は諦める (テストのenv_logger等)
        let _ = log::set_boxed_logger(Box::new(MiniLogger {
            max_level: logger.max_level,
        }));
        log::set_max_level(level);
        eprintln!("[cskk] logger installed (level = {})", level);
        logger
    });
}
