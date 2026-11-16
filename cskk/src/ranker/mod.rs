//! 文脈適応型変換 (Context-aware candidate re-ranking)
//!
//! Space 変換時に、カーソル周辺の確定済みテキスト (文脈) に基づいて
//! 変換候補を再ランキングする機構。
//!
//! 設計: docs/context-ranking-design.md, docs/zenz-integration-plan.md
//!
//! - IME (fcitx5-cskk) は変換前に `skk_context_set_surrounding_text` で
//!   カーソル前テキストを設定する。
//! - 読み入力中に `prefetch` でスコアリングを開始し (遅延隠蔽)、
//!   変換時に `rerank` がキャッシュを参照して候補順序を決定する。
//! - スコアラーは zenz系モデル (llama.cpp) + 個人変換履歴。
//! - ランカーがエラーを返した場合やスコア未到達の場合は呼び出し側が
//!   元の辞書順を維持する (従来SKK動作へのフォールバック要件)。

use crate::dictionary::Candidate;
use crate::error::CskkError;

pub mod history;
pub mod llama_ffi;
pub mod zenz_ranker;

pub use zenz_ranker::ZenzRanker;

/// 候補再ランキング器のインタフェース。
///
/// 実装は入力操作をブロックしてはならない:
/// - `rerank` は短いタイムアウト内でキャッシュを参照する
/// - 重い推論は `prefetch` で読み入力中に非同期に行う
pub trait ContextRanker: Send {
    /// `before_text` (カーソル前の確定済みテキスト) と `reading`
    /// (ひらがな読み) を元に `candidates` をその場で並べ替える。
    /// 失敗時やスコア未到達の場合は元の順序を維持する。
    fn rerank(
        &self,
        before_text: &str,
        reading: &str,
        candidates: &mut [Candidate],
    ) -> Result<(), CskkError>;

    /// 読み入力中に呼ばれ、変換に先立ってスコアリングを開始する。
    /// デフォルト実装はなにもしない。
    fn prefetch(&self, _before_text: &str, _reading: &str, _candidate_outputs: &[String]) {}

    /// 候補が確定した時に呼ばれる。個人学習を行うランカーはこれを記録する。
    /// デフォルト実装はなにもしない。
    fn record(&self, _before_text: &str, _candidate_output: &str) {}
}

/// なにもしないランカー。文脈ランキングの基盤確認用。
///
/// 順序は常に元の候補順 (辞書順) を維持する。
pub struct IdentityRanker;

impl ContextRanker for IdentityRanker {
    fn rerank(
        &self,
        _before_text: &str,
        _reading: &str,
        _candidates: &mut [Candidate],
    ) -> Result<(), CskkError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_ranker_keeps_order() {
        let ranker = IdentityRanker;
        let mut candidates = vec![
            Candidate::new(
                "あい".to_string(),
                None,
                false,
                "愛".to_string(),
                None,
                "愛".to_string(),
            ),
            Candidate::new(
                "あい".to_string(),
                None,
                false,
                "相".to_string(),
                None,
                "相".to_string(),
            ),
        ];
        let result = ranker.rerank("今日は", "あい", &mut candidates);
        assert!(result.is_ok());
        assert_eq!(candidates[0].kouho_text, "愛");
        assert_eq!(candidates[1].kouho_text, "相");
    }
}
