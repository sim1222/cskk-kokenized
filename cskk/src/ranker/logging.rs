//! libcskk用の最小ロガー。
//!
//! IMEプロセス (fcitx5) にはRustのロガーが存在しないため、標準エラー出力へ
//! ログを出す。既定では info レベル (ロード状況などの状態ログが出る)。
//! 環境変数 CSKK_LOG でレベルを変更できる:
//!   CSKK_LOG=debug  詳細 (prefetch、候補スコア値)
//!   CSKK_LOG=warn   異常のみ
//!   CSKK_LOG=off    ログ無効
//!
//! fcitx5を端末から起動するとログがそのまま見える:
//!   fcitx5 -r

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
        // 既定は info: 環境変数なしでロード状況が見える
        let level = std::env::var("CSKK_LOG")
            .ok()
            .and_then(|v| v.parse::<log::LevelFilter>().ok())
            .unwrap_or(log::LevelFilter::Info);
        let _ = log::set_boxed_logger(Box::new(MiniLogger { max_level: level }));
        log::set_max_level(level);
        eprintln!("[cskk] logger installed (level = {})", level);
        MiniLogger { max_level: level }
    });
}
