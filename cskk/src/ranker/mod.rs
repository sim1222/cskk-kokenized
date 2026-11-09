//! 文脈適応型変換 (Context-aware candidate re-ranking)
//!
//! Space 変換時に、カーソル周辺の確定済みテキスト (文脈) に基づいて
//! 変換候補を再ランキングする機構。
//!
//! 設計: docs/context-ranking-design.md
//!
//! - IME (fcitx5-cskk) は変換前に `skk_context_set_surrounding_text` で
//!   カーソル前テキストを設定する。
//! - `update_candidate_list()` が `ContextRanker` を呼び出し、候補順序を決定する。
//! - ランカーがエラーを返した場合は呼び出し側が元の辞書順を維持する
//!   (従来SKK動作へのフォールバック要件)。

use crate::dictionary::Candidate;
use crate::error::CskkError;

pub mod ngram_ranker;

pub use ngram_ranker::NgramRanker;

/// 候補再ランキング器のインタフェース。
///
/// `rerank` は `before_text` (カーソル前の確定済みテキスト) を元に
/// `candidates` をその場で並べ替える。失敗時は `Err` を返し、
/// 呼び出し側は元の順序を維持する。
///
/// 実装は入力操作をブロックしないよう、同期実行でもマイクロ秒オーダーで
/// 完了するか、タイムアウト付きの非同期実装であることを想定している。
pub trait ContextRanker: Send {
    fn rerank(&self, before_text: &str, candidates: &mut [Candidate]) -> Result<(), CskkError>;

    /// 候補が確定した時に呼ばれる。個人学習を行うランカーはこれを記録する。
    /// デフォルト実装はなにもしない。
    fn record(&self, _before_text: &str, _candidate_output: &str) {}
}

/// なにもしないランカー。文脈ランキングの基盤確認用。
///
/// 順序は常に元の候補順 (辞書順) を維持する。
pub struct IdentityRanker;

impl ContextRanker for IdentityRanker {
    fn rerank(&self, _before_text: &str, _candidates: &mut [Candidate]) -> Result<(), CskkError> {
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
            Candidate::new("あい".to_string(), None, false, "愛".to_string(), None, "愛".to_string()),
            Candidate::new(
                "あい".to_string(),
                None,
                false,
                "相".to_string(),
                None,
                "相".to_string(),
            ),
        ];
        let result = ranker.rerank("今日は", &mut candidates);
        assert!(result.is_ok());
        assert_eq!(candidates[0].kouho_text, "愛");
        assert_eq!(candidates[1].kouho_text, "相");
    }
}
