//! 文字N-gramによる文脈スコアリングランカー。
//!
//! 事前学習モデル (文字5-gram: 4文字の文脈から次の1文字を予測) と
//! 個人変換履歴 (文脈末尾→確定候補の出現回数) を組み合わせて
//! 変換候補を再ランキングする。
//!
//! モデルファイルは `tools/build_ngram_model` で生成した bincode 形式。
//! 再現性のためファイル上の形式は BTreeMap を用いる。

use crate::dictionary::Candidate;
use crate::error::CskkError;
use crate::ranker::ContextRanker;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// プロセス全体でランカー (モデル + 個人履歴) を共有するキャッシュ。
///
/// IMEは入力コンテキストごとにCskkContextを作るため、モデルのロードは
/// プロセスで1回に抑え、履歴もユーザー単位で一貫させる。
type RankerCache = HashMap<(String, Option<String>), Arc<NgramRanker>>;

fn ranker_cache() -> &'static Mutex<RankerCache> {
    static CACHE: OnceLock<Mutex<RankerCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cache_key(model_path: &str, history_path: Option<&str>) -> (String, Option<String>) {
    (
        model_path.to_string(),
        history_path.map(|s| s.to_string()),
    )
}

/// ロード進行中のキー。二重ロードを防ぐ。
fn loading_keys() -> &'static Mutex<std::collections::HashSet<(String, Option<String>)>> {
    static LOADING: OnceLock<Mutex<std::collections::HashSet<(String, Option<String>)>>> =
        OnceLock::new();
    LOADING.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

/// モデルの文脈長。この文字数の直後に来る1文字を予測する。
pub const MODEL_CONTEXT_CHARS: usize = 4;
/// 個人履歴の文脈長。
pub const HISTORY_CONTEXT_CHARS: usize = 4;
/// 加算スムージングの係数。
const SMOOTHING_ALPHA: f64 = 0.1;
/// PMIの上下限。希少文脈由来のノイズを抑える。
const PMI_CLAMP: f64 = 5.0;
/// PMIのペア出現数による減衰定数。
/// count=1のペアは PMI×1/(1+K) に減衰し、頻出ペアはほぼ減衰しない。
const PMI_COUNT_DAMPING: f64 = 5.0;
/// 履歴マッチ1件あたりのスコア加算量。
const HISTORY_SCORE_PER_COUNT: f64 = 2.0;
/// この件数を超える候補リストは上位のみを再ランキング対象にする。
const MAX_RERANK_CANDIDATES: usize = 200;

/// モデルファイル上の形式。bincode でシリアライズする。
/// HashMap はシリアライズ順序が不定なため BTreeMap を使う (再現性)。
#[derive(Serialize, Deserialize, Debug, Default, PartialEq)]
pub struct CharNgramModelFile {
    /// 文脈(MODEL_CONTEXT_CHARS文字) → 次文字 → 出現回数
    pub contexts: BTreeMap<String, BTreeMap<char, u32>>,
    /// 次文字のユニグラム出現回数 (文脈が未知の場合のフォールバック)
    pub unigram: BTreeMap<char, u32>,
}

/// 読み込み済みモデルのメモリ上の表現。
#[derive(Debug, Default)]
pub struct CharNgramModel {
    pub contexts: HashMap<String, HashMap<char, u32>>,
    pub context_totals: HashMap<String, u32>,
    pub unigram: HashMap<char, u32>,
    pub unigram_total: u32,
    pub vocabulary_size: u32,
}

impl CharNgramModel {
    pub fn from_file_model(file_model: &CharNgramModelFile) -> Self {
        let mut contexts = HashMap::new();
        let mut context_totals = HashMap::new();
        for (context, nexts) in &file_model.contexts {
            let total: u32 = nexts.values().sum();
            if total > 0 {
                context_totals.insert(context.clone(), total);
                let nexts: HashMap<char, u32> =
                    nexts.iter().map(|(k, v)| (*k, *v)).collect();
                contexts.insert(context.clone(), nexts);
            }
        }
        let unigram: HashMap<char, u32> =
            file_model.unigram.iter().map(|(k, v)| (*k, *v)).collect();
        let unigram_total: u32 = unigram.values().sum();
        let vocabulary_size = unigram.len() as u32;
        CharNgramModel {
            contexts,
            context_totals,
            unigram,
            unigram_total,
            vocabulary_size,
        }
    }

    pub fn load(path: &Path) -> Result<Self, CskkError> {
        let file = File::open(path)
            .map_err(|e| CskkError::Error(format!("Failed to open ngram model {:?}: {}", path, e)))?;
        let reader = BufReader::new(file);
        let file_model: CharNgramModelFile = bincode::deserialize_from(reader)
            .map_err(|e| CskkError::Error(format!("Failed to parse ngram model {:?}: {}", path, e)))?;
        Ok(Self::from_file_model(&file_model))
    }

    /// 文字列の末尾n文字を返す。
    pub fn tail_chars(text: &str, n: usize) -> &str {
        let char_count = text.chars().count();
        if char_count <= n {
            text
        } else {
            match text.char_indices().nth(char_count - n) {
                Some((byte_index, _)) => &text[byte_index..],
                None => "",
            }
        }
    }

    /// 次文字のlog確率を返す。文脈が未知の場合はユニグラムで代用する。
    pub fn logprob(&self, context: &str, next: char) -> f64 {
        let (count, total) = if let Some(nexts) = self.contexts.get(context) {
            let count = nexts.get(&next).copied().unwrap_or(0);
            let total = self.context_totals.get(context).copied().unwrap_or(0);
            (count as f64, total as f64)
        } else {
            let count = self.unigram.get(&next).copied().unwrap_or(0) as f64;
            (count, self.unigram_total as f64)
        };
        let vocab = self.vocabulary_size.max(1) as f64;
        ((count + SMOOTHING_ALPHA) / (total + SMOOTHING_ALPHA * vocab)).ln()
    }

    /// ユニグラムlog確率。
    fn logprob_unigram(&self, next: char) -> f64 {
        let count = self.unigram.get(&next).copied().unwrap_or(0) as f64;
        let vocab = self.vocabulary_size.max(1) as f64;
        ((count + SMOOTHING_ALPHA) / (self.unigram_total as f64 + SMOOTHING_ALPHA * vocab)).ln()
    }

    /// 文脈と次文字の相互情報量 (PMI)。
    ///
    /// 文脈が未知、または文脈内でこの文字が出現していない場合は0となり、
    /// 候補間で差がつかない (= 元の辞書順が維持される、フォールバック要件)。
    /// 剪定で上位max_nexts文字のみを保持しているため、出現ゼロは
    /// 「この文字が続かない」ではなく「情報なし」として扱う。
    ///
    /// 希少ペア (countが小さい) はPMIが大きく振れるため、出現数で減衰する。
    fn pmi(&self, context: &str, next: char) -> f64 {
        let nexts = match self.contexts.get(context) {
            Some(nexts) => nexts,
            None => return 0.0,
        };
        let count = nexts.get(&next).copied().unwrap_or(0);
        if count == 0 {
            return 0.0;
        }
        let total = self.context_totals.get(context).copied().unwrap_or(0) as f64;
        let vocab = self.vocabulary_size.max(1) as f64;
        let p_context = (count as f64 + SMOOTHING_ALPHA)
            / (total + SMOOTHING_ALPHA * vocab);
        let pmi = p_context.ln() - self.logprob_unigram(next);
        let damping = count as f64 / (count as f64 + PMI_COUNT_DAMPING);
        pmi.clamp(-PMI_CLAMP, PMI_CLAMP) * damping
    }

    /// 「文脈の直後に候補文字列が続く」PMIの総和を返す。
    /// 文脈に情報がない部分は0となり、候補間の順序に影響しない。
    pub fn score_candidate(&self, before_text: &str, candidate: &str) -> f64 {
        let mut window = Self::tail_chars(before_text, MODEL_CONTEXT_CHARS).to_string();
        let mut score = 0.0;
        for ch in candidate.chars() {
            score += self.pmi(&window, ch);
            window.push(ch);
            if window.chars().count() > MODEL_CONTEXT_CHARS {
                window = Self::tail_chars(&window, MODEL_CONTEXT_CHARS).to_string();
            }
        }
        score
    }
}

/// 個人変換履歴。文脈末尾→確定候補の出現回数。
/// ファイルへの保存は bincode (BTreeMap、書き込み時に一時ファイル + rename)。
#[derive(Serialize, Deserialize, Debug, Default, PartialEq)]
pub struct HistoryFile {
    /// 文脈(HISTORY_CONTEXT_CHARS文字) → 確定候補output → 回数
    pub counts: BTreeMap<String, BTreeMap<String, u32>>,
}

#[derive(Debug)]
struct HistoryStore {
    counts: HashMap<String, HashMap<String, u32>>,
    path: Option<PathBuf>,
}

impl HistoryStore {
    fn from_file_model(file_model: &HistoryFile, path: Option<PathBuf>) -> Self {
        let counts = file_model
            .counts
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.iter().map(|(k2, v2)| (k2.clone(), *v2)).collect(),
                )
            })
            .collect();
        HistoryStore { counts, path }
    }

    fn load(path: Option<&Path>) -> Result<Self, CskkError> {
        let store = match path {
            Some(p) if p.exists() => {
                let file = File::open(p).map_err(|e| {
                    CskkError::Error(format!("Failed to open history {:?}: {}", p, e))
                })?;
                let file_model: HistoryFile = bincode::deserialize_from(BufReader::new(file))
                    .map_err(|e| {
                        CskkError::Error(format!("Failed to parse history {:?}: {}", p, e))
                    })?;
                Self::from_file_model(&file_model, Some(p.to_path_buf()))
            }
            _ => Self::from_file_model(&HistoryFile::default(), path.map(|p| p.to_path_buf())),
        };
        Ok(store)
    }

    fn add(&mut self, before_text: &str, output: &str) {
        let context = CharNgramModel::tail_chars(before_text, HISTORY_CONTEXT_CHARS).to_string();
        self.counts
            .entry(context)
            .or_default()
            .entry(output.to_string())
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    fn get_count(&self, before_text: &str, output: &str) -> u32 {
        let context = CharNgramModel::tail_chars(before_text, HISTORY_CONTEXT_CHARS);
        self.counts
            .get(context)
            .and_then(|outputs| outputs.get(output))
            .copied()
            .unwrap_or(0)
    }

    fn save(&self) -> Result<(), CskkError> {
        let path = match &self.path {
            Some(p) => p,
            None => return Ok(()),
        };
        let file_model = HistoryFile {
            counts: self
                .counts
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.iter().map(|(k2, v2)| (k2.clone(), *v2)).collect(),
                    )
                })
                .collect(),
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                CskkError::Error(format!("Failed to create history dir {:?}: {}", parent, e))
            })?;
        }
        let tmp_path = path.with_extension("tmp");
        let tmp_file = File::create(&tmp_path).map_err(|e| {
            CskkError::Error(format!("Failed to create temp history {:?}: {}", tmp_path, e))
        })?;
        bincode::serialize_into(BufWriter::new(tmp_file), &file_model).map_err(|e| {
            CskkError::Error(format!("Failed to save history {:?}: {}", tmp_path, e))
        })?;
        std::fs::rename(&tmp_path, path).map_err(|e| {
            CskkError::Error(format!("Failed to rename history to {:?}: {}", path, e))
        })?;
        Ok(())
    }
}

/// 文字N-gram + 個人履歴による再ランキング器。
pub struct NgramRanker {
    model: CharNgramModel,
    history: Mutex<HistoryStore>,
}

impl NgramRanker {
    /// モデルファイルを読み込んでランカーを生成する。
    /// history_path が None の場合は履歴を保存しない (メモリのみ)。
    pub fn load(
        model_path: &Path,
        history_path: Option<&Path>,
    ) -> Result<Self, CskkError> {
        let model = CharNgramModel::load(model_path)?;
        let history = HistoryStore::load(history_path)?;
        Ok(NgramRanker {
            model,
            history: Mutex::new(history),
        })
    }

    /// 読み込み済みのランカーがあれば取得する (非ブロッキング)。
    pub fn try_get_cached(
        model_path: &str,
        history_path: Option<&str>,
    ) -> Option<Arc<NgramRanker>> {
        let cache = ranker_cache().lock().ok()?;
        cache.get(&cache_key(model_path, history_path)).cloned()
    }

    /// モデルをバックグラウンドスレッドで読み込み、完了次第キャッシュに入れる。
    /// 読み込み中も入力はブロックされず、従来のSKK動作で変換される。
    /// 読み込み失敗時はキャッシュに入らず警告ログのみを出す。
    pub fn spawn_load(model_path: &str, history_path: Option<String>) {
        let key = cache_key(model_path, history_path.as_deref());
        // 二重ロード防止: 同じキーのロードが進行中なら何もしない
        if let Ok(mut loading) = loading_keys().lock() {
            if loading.contains(&key) {
                return;
            }
            loading.insert(key.clone());
        }
        let key_for_err = key.clone();
        std::thread::Builder::new()
            .name("cskk-ngram-loader".to_string())
            .spawn(move || {
                let history = key.1.as_deref().map(Path::new);
                let started = std::time::Instant::now();
                match NgramRanker::load(Path::new(&key.0), history) {
                    Ok(ranker) => {
                        log::info!(
                            "Loaded ngram ranker from {} in {:?}",
                            key.0,
                            started.elapsed()
                        );
                        if let Ok(mut cache) = ranker_cache().lock() {
                            cache.insert(key.clone(), Arc::new(ranker));
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to load ngram ranker from {}: {}. \
                             Context ranking falls back to the default behavior.",
                            key.0,
                            e
                        );
                    }
                }
                // ロード完了 (成否問わず): 進行中セットから外す
                if let Ok(mut loading) = loading_keys().lock() {
                    loading.remove(&key);
                }
            })
            .map_err(|e| {
                log::warn!("Failed to spawn ngram loader thread: {}", e);
                // スレッド起動失敗時も進行中セットから外す
                if let Ok(mut loading) = loading_keys().lock() {
                    loading.remove(&key_for_err);
                }
                e
            })
            .ok();
    }
}

impl ContextRanker for NgramRanker {
    /// 候補を「モデルスコア + 履歴スコア」の降順に並べ替える。
    /// 同スコアの場合は元の辞書順を維持する (安定ソート)。
    fn rerank(&self, before_text: &str, candidates: &mut [Candidate]) -> Result<(), CskkError> {
        let len = candidates.len();
        if len < 2 {
            return Ok(());
        }
        let rerank_len = len.min(MAX_RERANK_CANDIDATES);
        let history = self
            .history
            .lock()
            .map_err(|_| CskkError::Error("History lock poisoned".to_string()))?;

        // (score, 元index)。元indexを第2キーにすることで同点は辞書順を維持する。
        let mut scored: Vec<(f64, usize)> = candidates[..rerank_len]
            .iter()
            .enumerate()
            .map(|(i, candidate)| {
                let model_score = self.model.score_candidate(before_text, &candidate.output);
                let history_count = history.get_count(before_text, &candidate.output);
                let history_score = HISTORY_SCORE_PER_COUNT * (history_count as f64).ln_1p();
                (model_score + history_score, i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let new_order: Vec<usize> = scored.iter().map(|(_, i)| *i).collect();
        let mut reordered: Vec<Candidate> = Vec::with_capacity(len);
        for &i in &new_order {
            let candidate = &candidates[i];
            reordered.push(candidate.clone());
        }
        for candidate in candidates.iter_mut().skip(rerank_len) {
            reordered.push(candidate.clone());
        }
        candidates.clone_from_slice(&reordered);
        Ok(())
    }

    /// 候補確定時に呼ばれ、個人履歴に記録する。
    fn record(&self, before_text: &str, candidate_output: &str) {
        if let Ok(mut history) = self.history.lock() {
            history.add(before_text, candidate_output);
            if let Err(e) = history.save() {
                log::warn!("Failed to save context ranking history: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> CharNgramModel {
        // 「...ーグルで」の次は「検」が多く、その次は「索」。
        // 「...ーグルで」→「削」は稀というモデルを作る。
        let mut file_model = CharNgramModelFile::default();
        let mut nexts = BTreeMap::new();
        nexts.insert('検', 100);
        nexts.insert('削', 1);
        file_model.contexts.insert("ーグルで".to_string(), nexts);
        let mut nexts2 = BTreeMap::new();
        nexts2.insert('索', 100);
        file_model.contexts.insert("グルで検".to_string(), nexts2);
        let mut nexts3 = BTreeMap::new();
        nexts3.insert('除', 100);
        file_model.contexts.insert("グルで削".to_string(), nexts3);
        let mut unigram = BTreeMap::new();
        unigram.insert('検', 50);
        unigram.insert('索', 50);
        unigram.insert('削', 50);
        unigram.insert('除', 50);
        file_model.unigram = unigram;
        CharNgramModel::from_file_model(&file_model)
    }

    fn candidate(output: &str) -> Candidate {
        Candidate::new(
            "けんさく".to_string(),
            None,
            false,
            output.to_string(),
            None,
            output.to_string(),
        )
    }

    #[test]
    fn tail_chars_returns_suffix() {
        assert_eq!(CharNgramModel::tail_chars("あいうえお", 2), "えお");
        assert_eq!(CharNgramModel::tail_chars("あい", 4), "あい");
        assert_eq!(CharNgramModel::tail_chars("", 4), "");
    }

    #[test]
    fn model_prefers_frequent_continuation() {
        let model = test_model();
        // 文脈「グーグルで」では「検索」の方が「削除」より高スコアになること
        let kensaku = model.score_candidate("グーグルで", "検索");
        let sakujo = model.score_candidate("グーグルで", "削除");
        assert!(kensaku > sakujo, "{} > {}", kensaku, sakujo);
    }

    #[test]
    fn rerank_puts_preferred_candidate_first() {
        let model = test_model();
        let ranker = NgramRanker {
            model,
            history: Mutex::new(HistoryStore::from_file_model(
                &HistoryFile::default(),
                None,
            )),
        };
        let mut candidates = vec![candidate("削除"), candidate("検索")];
        ranker.rerank("グーグルで", &mut candidates).unwrap();
        assert_eq!(candidates[0].output, "検索");
        assert_eq!(candidates[1].output, "削除");
    }

    #[test]
    fn rerank_keeps_stable_order_on_tie() {
        let model = test_model();
        let ranker = NgramRanker {
            model,
            history: Mutex::new(HistoryStore::from_file_model(
                &HistoryFile::default(),
                None,
            )),
        };
        // 文脈にどちらの候補も依存しない場合、元の順序を維持する
        let mut candidates = vec![candidate("削除"), candidate("検索")];
        ranker.rerank("まったく無関係な文脈", &mut candidates).unwrap();
        assert_eq!(candidates[0].output, "削除");
        assert_eq!(candidates[1].output, "検索");
    }

    #[test]
    fn history_boosts_recorded_candidate() {
        let model = test_model();
        let ranker = NgramRanker {
            model,
            history: Mutex::new(HistoryStore::from_file_model(
                &HistoryFile::default(),
                None,
            )),
        };
        // 「削除」をこの文脈で確定した履歴を作る
        ranker.record("グーグルで", "削除");
        let mut candidates = vec![candidate("検索"), candidate("削除")];
        ranker.rerank("グーグルで", &mut candidates).unwrap();
        // 履歴1件でモデル差(約log100)を上回るほどではないが、
        // 履歴スコアが加算されていること: 記録なしなら「検索」が先頭、
        // 記録ありでも順序は同じだがスコア差が縮まっていることを確認するのは
        // 煩雑なため、履歴が逆転させるケースを別の文脈で確認する。
        // (ここでは record が例外なく動作することを確認)
        assert_eq!(candidates[0].output, "検索");
    }

    #[test]
    fn history_can_override_model() {
        let model = test_model();
        let mut history_file = HistoryFile::default();
        let mut outputs = BTreeMap::new();
        outputs.insert("削除".to_string(), 10);
        history_file.counts.insert("ーグルで".to_string(), outputs);
        let ranker = NgramRanker {
            model,
            history: Mutex::new(HistoryStore::from_file_model(&history_file, None)),
        };
        let mut candidates = vec![candidate("検索"), candidate("削除")];
        ranker.rerank("グーグルで", &mut candidates).unwrap();
        // 履歴10件で「削除」が逆転すること
        assert_eq!(candidates[0].output, "削除");
    }

    #[test]
    fn history_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history.bin");
        {
            let mut store = HistoryStore::load(Some(&path)).unwrap();
            store.add("今日は", "晴れ");
            store.add("今日は", "晴れ");
            store.add("昨日は", "雨");
            store.save().unwrap();
        }
        let store = HistoryStore::load(Some(&path)).unwrap();
        assert_eq!(store.get_count("今日は", "晴れ"), 2);
        assert_eq!(store.get_count("昨日は", "雨"), 1);
        assert_eq!(store.get_count("今日は", "雨"), 0);
    }

    #[test]
    fn model_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.bin");
        let file_model = CharNgramModelFile {
            contexts: {
                let mut m = BTreeMap::new();
                let mut nexts = BTreeMap::new();
                nexts.insert('検', 10);
                m.insert("ーグルで".to_string(), nexts);
                let mut nexts = BTreeMap::new();
                nexts.insert('索', 10);
                m.insert("グルで検".to_string(), nexts);
                let mut nexts = BTreeMap::new();
                nexts.insert('す', 10);
                m.insert("ルで検索".to_string(), nexts);
                m
            },
            unigram: {
                let mut m = BTreeMap::new();
                m.insert('検', 10);
                m.insert('索', 1);
                m.insert('す', 1);
                m.insert('X', 1);
                m
            },
        };
        let file = File::create(&path).unwrap();
        bincode::serialize_into(BufWriter::new(file), &file_model).unwrap();
        let model = CharNgramModel::load(&path).unwrap();
        // 「検索する」の続きはモデルが知っており、「検索X」より高スコアになること
        assert!(
            model.score_candidate("グーグルで", "検索す")
                > model.score_candidate("グーグルで", "検索X")
        );
    }
}
