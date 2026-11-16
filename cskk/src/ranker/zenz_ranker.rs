//! zenz系モデル (zenz-v2/v3、jinen) による文脈スコアリングランカー。
//!
//! llama.cpp (dlopen) でモデルをVulkan/CPUにロードし、
//! 「左文脈 + 読み + 候補」の条件付きlog確率で候補を再ランキングする。
//!
//! 入力形式 (AzooKeyKanaKanjiConverter ZenzPromptBuilder 準拠):
//! - v3 (zenz-v3.x / jinen): `U+EE02 + 左文脈 + U+EE00 + 読み(カタカナ) + U+EE01`
//! - v2 (zenz-v2/v2.5):     `U+EE00 + 読み(カタカナ) + U+EE02 + 左文脈 + U+EE01`
//! 読みはカタカナ、左文脈は末尾40文字。
//!
//! 遅延隠蔽: SKKの読み入力中に prefetch でスコアリングを開始し、
//! 変換時点 (Space) ではキャッシュを最大 wait_ms だけ待つ。
//! 間に合わなければ元の辞書順を維持する (従来SKK動作へのフォールバック)。

use crate::dictionary::Candidate;
use crate::error::CskkError;
use crate::ranker::history::{self, HistoryStore};
use crate::ranker::llama_ffi::LlamaRuntime;
use crate::ranker::ContextRanker;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use log::{debug, log_enabled};

/// プロンプト形式。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PromptVersion {
    /// zenz-v3.x / jinen 形式
    V3,
    /// zenz-v2 / v2.5 形式
    V2,
}

const INPUT_TAG: char = '\u{EE00}';
const OUTPUT_TAG: char = '\u{EE01}';
const CONTEXT_TAG: char = '\u{EE02}';
/// 左文脈の最大文字数 (zenz準拠)。
const MAX_CONTEXT_CHARS: usize = 40;
/// prefetchする候補の上限。
const MAX_PREFETCH_CANDIDATES: usize = 10;
/// キャッシュ保持エントリ上限。
const CACHE_MAX_ENTRIES: usize = 64;
/// 変換時点でスコアを待つ上限。
/// 送り仮名あり変換は大文字入力で即変換されるためprefetchの暇がなく、
/// この場でスコアリングを開始して待つ (実測10候補で約20ms)。
const COLD_SCORING_TIMEOUT: Duration = Duration::from_millis(150);

fn input_tag_str() -> String {
    INPUT_TAG.to_string()
}
fn output_tag_str() -> String {
    OUTPUT_TAG.to_string()
}
fn context_tag_str() -> String {
    CONTEXT_TAG.to_string()
}

/// プロンプトを構築する。
pub fn build_prompt(version: PromptVersion, left_context: &str, reading_katakana: &str) -> String {
    let context: String = crate::ranker::history::tail_chars(left_context, MAX_CONTEXT_CHARS)
        .to_string();
    match version {
        PromptVersion::V3 => format!(
            "{}{}{}{}{}",
            context_tag_str(),
            context,
            input_tag_str(),
            reading_katakana,
            output_tag_str()
        ),
        PromptVersion::V2 => format!(
            "{}{}{}{}{}",
            input_tag_str(),
            reading_katakana,
            context_tag_str(),
            context,
            output_tag_str()
        ),
    }
}

/// ひらがな読みをカタカナに変換する (zenzの入力仕様)。
pub fn hiragana_to_katakana(text: &str) -> String {
    text.chars()
        .map(|c| {
            let code = c as u32;
            if ('ぁ'..='ゖ').contains(&c) {
                char::from_u32(code + 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// モデル名 (GGUF general.name) からプロンプト形式を推定する。
fn detect_prompt_version(model_name: &str) -> PromptVersion {
    let name = model_name.to_lowercase();
    if name.contains("v2") && !name.contains("v3") && !name.contains("jinen") {
        // "zenz-v2" / "zenz-v2.5" 等
        PromptVersion::V2
    } else {
        // v3系 / jinen / 不明 → v3形式
        PromptVersion::V3
    }
}

/// スコアリングの実体の抽象 (テストでスタブに差し替え可能)。
pub trait ZenzScorer: Send + Sync {
    /// 候補ごとのスコア (条件付きlog確率の総和) を返す。
    /// before/left_context と reading からプロンプトを作り、各候補を評価する。
    fn score(
        &self,
        left_context: &str,
        reading: &str,
        candidates: &[String],
    ) -> Result<Vec<f64>, CskkError>;
}

/// llama.cppによるスコアラー。
pub struct LlamaScorer {
    runtime: Mutex<LlamaRuntime>,
    version: PromptVersion,
}

impl LlamaScorer {
    pub fn load(model_path: &str) -> Result<Self, CskkError> {
        let runtime = LlamaRuntime::new(model_path, 512)?;
        let version = runtime
            .metadata("general.name")
            .map(|name| {
                let v = detect_prompt_version(&name);
                log::info!("libcskk: zenz model \"{}\" → prompt {:?}", name, v);
                v
            })
            .unwrap_or(PromptVersion::V3);
        let scorer = LlamaScorer {
            runtime: Mutex::new(runtime),
            version,
        };
        // ウォームアップ: 初回推論はGPUバックエンドの初期化
        // (Vulkanのシェーダコンパイル等) で数秒かかるため、
        // モデルロード時に1回済ませておく (変換時のタイムアウトを防ぐ)。
        let started = Instant::now();
        let warmup = scorer.score("", "ア", &["ア".to_string()]);
        match warmup {
            Ok(_) => log::info!(
                "libcskk: zenz warmup done in {:?}",
                started.elapsed()
            ),
            Err(e) => log::warn!("libcskk: zenz warmup failed: {}", e),
        }
        Ok(scorer)
    }
}

impl ZenzScorer for LlamaScorer {
    fn score(
        &self,
        left_context: &str,
        reading: &str,
        candidates: &[String],
    ) -> Result<Vec<f64>, CskkError> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| CskkError::Error("llama runtime lock poisoned".to_string()))?;
        let prompt = build_prompt(self.version, left_context, reading);
        let t0 = std::time::Instant::now();
        let (prompt_len, prompt_row) = runtime.prepare_prompt(&prompt)?;
        let t1 = std::time::Instant::now();
        let mut scores = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            scores.push(runtime.score_continuation(prompt_len, &prompt_row, candidate)?);
        }
        let t2 = std::time::Instant::now();
        debug!(
            "zenz score timing: prompt={:?} candidates({})={:?}",
            t1 - t0,
            candidates.len(),
            t2 - t1
        );
        runtime.clear_cache();
        Ok(scores)
    }
}

/// prefetchタスク。
struct Task {
    left_context: String,
    reading: String,
    candidates: Vec<String>,
}

type CacheKey = (String, String);

struct ScoreEntry {
    scores: HashMap<String, f64>,
}

/// zenz + 個人履歴による再ランキング器。
pub struct ZenzRanker {
    scorer: Mutex<Option<Arc<dyn ZenzScorer>>>,
    history: Mutex<HistoryStore>,
    cache: Mutex<HashMap<CacheKey, Arc<ScoreEntry>>>,
    in_flight: Mutex<HashSet<CacheKey>>,
    condvar: Condvar,
    /// 変換時点でキャッシュを待つ上限。
    wait_timeout: Duration,
    worker: Mutex<mpsc::Sender<Task>>,
}

impl ZenzRanker {
    /// スコアラーを同期的に構築してランカーを作る (テスト用)。
    pub fn new_with_scorer(
        scorer: Arc<dyn ZenzScorer>,
        history: HistoryStore,
    ) -> Arc<Self> {
        Self::spawn(scorer, history, Duration::from_millis(50))
    }

    /// スコアラー非同期ロード用に、後からスコアラーを差し込めるランカーを作る。
    pub fn spawn_placeholder(history: HistoryStore) -> Arc<Self> {
        Self::spawn(
            Arc::new(NoopScorer),
            history,
            Duration::from_millis(50),
        )
    }

    fn spawn(
        scorer: Arc<dyn ZenzScorer>,
        history: HistoryStore,
        wait_timeout: Duration,
    ) -> Arc<Self> {
        let (tx, rx) = mpsc::channel::<Task>();
        let ranker = Arc::new(ZenzRanker {
            scorer: Mutex::new(Some(scorer)),
            history: Mutex::new(history),
            cache: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(HashSet::new()),
            condvar: Condvar::new(),
            wait_timeout,
            worker: Mutex::new(tx),
        });
        let worker_ranker = Arc::clone(&ranker);
        std::thread::Builder::new()
            .name("cskk-zenz-worker".to_string())
            .spawn(move || {
                for task in rx {
                    worker_ranker.process_task(task);
                }
            })
            .ok();
        ranker
    }

    /// スコアラーを差し替える (非同期ロード完了時)。
    pub fn set_scorer(&self, scorer: Arc<dyn ZenzScorer>) {
        if let Ok(mut guard) = self.scorer.lock() {
            *guard = Some(scorer);
        }
    }

    fn process_task(&self, task: Task) {
        let key = (task.left_context.clone(), task.reading.clone());
        let scorer = match self.scorer.lock() {
            Ok(guard) => match guard.clone() {
                Some(s) => s,
                None => {
                    self.finish_task(&key);
                    return;
                }
            },
            Err(_) => {
                self.finish_task(&key);
                return;
            }
        };
        let started = Instant::now();
        let result = scorer.score(&task.left_context, &task.reading, &task.candidates);
        debug!(
            "zenz score took {:?} for {} candidates",
            started.elapsed(),
            task.candidates.len()
        );
        match result {
            Ok(scores) => {
                if log_enabled!(log::Level::Debug) {
                    let pairs: Vec<String> = task
                        .candidates
                        .iter()
                        .zip(scores.iter())
                        .map(|(c, s)| format!("{}={:.2}", c, s))
                        .collect();
                    debug!(
                        "zenz scores: context=\"{}\" reading=\"{}\" {}",
                        task.left_context,
                        task.reading,
                        pairs.join(" ")
                    );
                }
                let mut map = HashMap::new();
                for (candidate, score) in task.candidates.iter().zip(scores.iter()) {
                    map.insert(candidate.clone(), *score);
                }
                if let Ok(mut cache) = self.cache.lock() {
                    if cache.len() >= CACHE_MAX_ENTRIES {
                        // 粗く古いエントリを削除 (順序保持ではないが十分)
                        if let Some(oldest) = cache.keys().next().cloned() {
                            cache.remove(&oldest);
                        }
                    }
                    cache.insert(key.clone(), Arc::new(ScoreEntry { scores: map }));
                }
            }
            Err(e) => {
                debug!(
                    "zenz scoring failed: context=\"{}\" reading=\"{}\": {}",
                    task.left_context, task.reading, e
                );
            }
        }
        self.finish_task(&key);
    }

    fn finish_task(&self, key: &CacheKey) {
        if let Ok(mut in_flight) = self.in_flight.lock() {
            in_flight.remove(key);
        }
        self.condvar.notify_all();
    }

    fn lookup_cache(
        &self,
        key: &CacheKey,
    ) -> Option<Arc<ScoreEntry>> {
        self.cache.lock().ok()?.get(key).cloned()
    }

    fn is_in_flight(&self, key: &CacheKey) -> bool {
        self.in_flight
            .lock()
            .map(|set| set.contains(key))
            .unwrap_or(false)
    }

    /// スコアリングタスクをワーカーへ投入する。
    /// 読みは送り仮名 (ふりがな) 込みのフル読みとしてカタカナに正規化する。
    fn enqueue_task(
        &self,
        key: &CacheKey,
        before_text: &str,
        reading: &str,
        okuri: &str,
        candidate_outputs: &[String],
    ) {
        if let Ok(mut in_flight) = self.in_flight.lock() {
            in_flight.insert(key.clone());
        }
        let candidates: Vec<String> = candidate_outputs
            .iter()
            .take(MAX_PREFETCH_CANDIDATES)
            .cloned()
            .collect();
        let n = candidates.len();
        let task = Task {
            left_context: before_text.to_string(),
            reading: hiragana_to_katakana(&format!("{}{}", reading, okuri)),
            candidates,
        };
        if let Ok(worker) = self.worker.lock() {
            let _ = worker.send(task);
            debug!(
                "zenz prefetch: context=\"{}\" reading=\"{}\" candidates={}",
                before_text,
                reading,
                n
            );
        }
    }
}

/// スコアラー未ロード時のプレースホルダ (常に失敗=フォールバック)。
struct NoopScorer;

impl ZenzScorer for NoopScorer {
    fn score(
        &self,
        _left: &str,
        _reading: &str,
        _candidates: &[String],
    ) -> Result<Vec<f64>, CskkError> {
        Err(CskkError::Error("zenz scorer not loaded yet".to_string()))
    }
}

impl ContextRanker for ZenzRanker {
    /// 読み入力中に呼ばれ、バックグラウンドでスコアリングを開始する。
    fn prefetch(&self, before_text: &str, reading: &str, okuri: &str, candidate_outputs: &[String]) {
        if reading.is_empty() || candidate_outputs.is_empty() {
            return;
        }
        // キーは読み+送り仮名をカタカナに正規化して揃える
        let key = cache_key_of(before_text, reading, okuri);
        if self.lookup_cache(&key).is_some() || self.is_in_flight(&key) {
            return;
        }
        self.enqueue_task(&key, before_text, reading, okuri, candidate_outputs);
    }

    /// 変換時に呼ばれる。キャッシュがあればスコア適用。
    /// なければこの場でスコアリングを開始し COLD_SCORING_TIMEOUT まで待つ
    /// (送り仮名あり変換のようにprefetchの暇がないケース向け)。
    /// 間に合わなければ元の順序を維持する。
    fn rerank(
        &self,
        before_text: &str,
        reading: &str,
        okuri: &str,
        candidates: &mut [Candidate],
    ) -> Result<(), CskkError> {
        let len = candidates.len();
        debug!(
            "zenz rerank: context=\"{}\" reading=\"{}\" okuri=\"{}\" candidates={}",
            before_text,
            reading,
            okuri,
            len
        );
        if len < 2 || reading.is_empty() {
            return Ok(());
        }
        let key = cache_key_of(before_text, reading, okuri);
        let mut entry = self.lookup_cache(&key);
        if entry.is_none() {
            // キャッシュmiss: この場でスコアリングを開始する
            // 送り仮名 (ふりがな) 込みの評価面でスコアする
            self.enqueue_task(&key, before_text, reading, okuri, &{
                candidates
                    .iter()
                    .map(|c| format!("{}{}", c.output, okuri))
                    .collect::<Vec<String>>()
            });
            // 完成を待つ (ワーカーは通常アイドルで、実測10候補≈20ms)
            if let Ok(in_flight) = self.in_flight.lock() {
                let deadline = Instant::now() + COLD_SCORING_TIMEOUT;
                let mut guard = in_flight;
                while guard.contains(&key) && Instant::now() < deadline {
                    let (result, _timeout) = self
                        .condvar
                        .wait_timeout(guard, Duration::from_millis(5))
                        .unwrap_or_else(|e| e.into_inner());
                    guard = result;
                }
            }
            entry = self.lookup_cache(&key);
        }
        let entry = match entry {
            Some(entry) => entry,
            None => {
                debug!(
                    "zenz rerank: no scores yet (context=\"{}\" reading=\"{}\"), keeping dictionary order",
                    before_text,
                    reading
                );
                return Ok(()); // フォールバック: 元の辞書順
            }
        };
        let history = self
            .history
            .lock()
            .map_err(|_| CskkError::Error("history lock poisoned".to_string()))?;
        // (スコア+履歴ブースト, 元index)。同点は元の辞書順を維持する。
        // スコアのキーは送り仮名 (ふりがな) 込みの評価面。
        let mut scored: Vec<(f64, usize)> = candidates
            .iter()
            .enumerate()
            .map(|(i, candidate)| {
                let surface = format!("{}{}", candidate.output, okuri);
                let score = entry.scores.get(&surface).copied();
                match score {
                    Some(score) => {
                        let boost = history::history_score(&history, before_text, &candidate.output);
                        (score + boost, i)
                    }
                    // スコア未取得の候補は最下位グループとして元順を維持
                    None => (f64::NEG_INFINITY, i),
                }
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let new_order: Vec<usize> = scored.iter().map(|(_, i)| *i).collect();
        let reordered: Vec<Candidate> = new_order
            .iter()
            .map(|&i| candidates[i].clone())
            .collect();
        candidates.clone_from_slice(&reordered);
        Ok(())
    }

    /// 候補確定時に呼ばれ、個人履歴に記録する。
    fn record(&self, before_text: &str, candidate_output: &str) {
        history::record_and_save(&self.history, before_text, candidate_output);
    }
}

/// zenzモデルの非同期ロード + プロセス共有キャッシュ。
type RankerCache = HashMap<(String, Option<String>), Arc<ZenzRanker>>;

fn ranker_cache() -> &'static Mutex<RankerCache> {
    static CACHE: std::sync::OnceLock<Mutex<RankerCache>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn loading_keys() -> &'static Mutex<HashSet<(String, Option<String>)>> {
    static LOADING: std::sync::OnceLock<Mutex<HashSet<(String, Option<String>)>>> =
        std::sync::OnceLock::new();
    LOADING.get_or_init(|| Mutex::new(HashSet::new()))
}

/// ランカー内キャッシュのキー: (文脈, カタカナ(読み+送り仮名))。
fn cache_key_of(before_text: &str, reading: &str, okuri: &str) -> CacheKey {
    (
        before_text.to_string(),
        hiragana_to_katakana(&format!("{}{}", reading, okuri)),
    )
}

fn cache_key(
    model_path: &str,
    history_path: Option<&str>,
) -> (String, Option<String>) {
    (
        model_path.to_string(),
        history_path.map(|s| s.to_string()),
    )
}

impl ZenzRanker {
    /// 読み込み済みのランカーがあれば取得する (非ブロッキング)。
    pub fn try_get_cached(
        model_path: &str,
        history_path: Option<&str>,
    ) -> Option<Arc<ZenzRanker>> {
        let cache = ranker_cache().lock().ok()?;
        cache.get(&cache_key(model_path, history_path)).cloned()
    }

    /// モデルをバックグラウンドスレッドで読み込み、完了次第ランカーに反映する。
    /// 読み込み中も入力はブロックされず、従来のSKK動作で変換される。
    pub fn spawn_load(model_path: &str, history_path: Option<String>) {
        let key = cache_key(model_path, history_path.as_deref());
        if let Ok(mut loading) = loading_keys().lock() {
            if loading.contains(&key) {
                return;
            }
            loading.insert(key.clone());
        }
        let key_for_err = key.clone();
        std::thread::Builder::new()
            .name("cskk-zenz-loader".to_string())
            .spawn(move || {
                let history = history_path.as_deref().map(Path::new);
                let history_store = match HistoryStore::load(history) {
                    Ok(store) => store,
                    Err(e) => {
                        log::warn!("Failed to load history: {}. Using empty history.", e);
                        HistoryStore::from_file_model(&Default::default(), history.map(PathBuf::from))
                    }
                };
                let started = Instant::now();
                match LlamaScorer::load(&key.0) {
                    Ok(scorer) => {
                        log::info!(
                            "Loaded zenz model {} in {:?}",
                            key.0,
                            started.elapsed()
                        );
                        let ranker = ZenzRanker::new_with_scorer(Arc::new(scorer), history_store);
                        if let Ok(mut cache) = ranker_cache().lock() {
                            cache.insert(key.clone(), ranker);
                        }
                        let _ = &key;
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to load zenz model from {}: {}. \
                             Context ranking falls back to the default behavior.",
                            key.0,
                            e
                        );
                    }
                }
                if let Ok(mut loading) = loading_keys().lock() {
                    loading.remove(&key);
                }
            })
            .map_err(|e| {
                log::warn!("Failed to spawn zenz loader thread: {}", e);
                if let Ok(mut loading) = loading_keys().lock() {
                    loading.remove(&key_for_err);
                }
                e
            })
            .ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 候補の並びを固定した決定的スタブスコアラー。
    struct StubScorer {
        preference: HashMap<String, f64>,
    }

    impl ZenzScorer for StubScorer {
        fn score(
            &self,
            _left: &str,
            _reading: &str,
            candidates: &[String],
        ) -> Result<Vec<f64>, CskkError> {
            Ok(candidates
                .iter()
                .map(|c| self.preference.get(c).copied().unwrap_or(-10.0))
                .collect())
        }
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
    fn katakana_conversion() {
        assert_eq!(hiragana_to_katakana("あいうえお"), "アイウエオ");
        assert_eq!(hiragana_to_katakana("がぎぐ"), "ガギグ");
        assert_eq!(hiragana_to_katakana("かい"), "カイ");
        assert_eq!(hiragana_to_katakana("A1-"), "A1-");
    }

    #[test]
    fn prompt_v3_and_v2() {
        let p3 = build_prompt(PromptVersion::V3, "文脈", "カイ");
        assert!(p3.starts_with('\u{EE02}'));
        assert!(p3.contains("文脈"));
        assert!(p3.contains('\u{EE00}'));
        assert!(p3.ends_with('\u{EE01}'));
        let p2 = build_prompt(PromptVersion::V2, "文脈", "カイ");
        assert!(p2.starts_with('\u{EE00}'));
    }

    #[test]
    fn detect_version_from_name() {
        assert_eq!(detect_prompt_version("Zenz v3.2 Xsmall"), PromptVersion::V3);
        assert_eq!(detect_prompt_version("zenz-v2"), PromptVersion::V2);
        assert_eq!(detect_prompt_version("zenz-v2.5-xsmall"), PromptVersion::V2);
        assert_eq!(detect_prompt_version("jinen-v1-xsmall"), PromptVersion::V3);
    }

    #[test]
    fn prefetch_and_rerank_reorders() {
        let mut preference = HashMap::new();
        preference.insert("検索".to_string(), -1.0);
        preference.insert("削除".to_string(), -5.0);
        let scorer = Arc::new(StubScorer { preference });
        let ranker = ZenzRanker::new_with_scorer(scorer, HistoryStore::from_file_model(&Default::default(), None));
        let outputs = vec!["削除".to_string(), "検索".to_string()];
        // prefetch → ワーカーが処理するまで待つ
        ranker.prefetch("グーグルで", "けんさく", "", &outputs);
        let key = ("グーグルで".to_string(), "ケンサク".to_string());
        let deadline = Instant::now() + Duration::from_secs(5);
        while ranker.lookup_cache(&key).is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        let mut candidates = vec![candidate("削除"), candidate("検索")];
        ranker
            .rerank("グーグルで", "けんさく", "", &mut candidates)
            .unwrap();
        assert_eq!(candidates[0].output, "検索");
    }

    #[test]
    fn rerank_without_cache_keeps_order() {
        let scorer = Arc::new(StubScorer {
            preference: HashMap::new(),
        });
        let ranker = ZenzRanker::new_with_scorer(scorer, HistoryStore::from_file_model(&Default::default(), None));
        let mut candidates = vec![candidate("削除"), candidate("検索")];
        // prefetchなし → キャッシュなし → 元の順序
        ranker
            .rerank("グーグルで", "けんさく", "", &mut candidates)
            .unwrap();
        assert_eq!(candidates[0].output, "削除");
    }

    #[test]
    fn history_can_override_model() {
        let mut preference = HashMap::new();
        preference.insert("検索".to_string(), -1.0);
        preference.insert("削除".to_string(), -2.0);
        let scorer = Arc::new(StubScorer { preference });
        let mut history_file = crate::ranker::history::HistoryFile::default();
        let mut outputs = std::collections::BTreeMap::new();
        outputs.insert("削除".to_string(), 10);
        history_file
            .counts
            .insert("ーグルで".to_string(), outputs);
        let ranker = ZenzRanker::new_with_scorer(
            scorer,
            HistoryStore::from_file_model(&history_file, None),
        );
        let outputs = vec!["検索".to_string(), "削除".to_string()];
        ranker.prefetch("グーグルで", "けんさく", "", &outputs);
        let key = ("グーグルで".to_string(), "ケンサク".to_string());
        let deadline = Instant::now() + Duration::from_secs(5);
        while ranker.lookup_cache(&key).is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        let mut candidates = vec![candidate("検索"), candidate("削除")];
        ranker
            .rerank("グーグルで", "けんさく", "", &mut candidates)
            .unwrap();
        // 履歴10件で「削除」が逆転すること
        assert_eq!(candidates[0].output, "削除");
    }

    /// 実モデルでスコアラーを直接検証する (通常のcargo testではスキップ)。
    /// 実行: CSKK_ZENZ_MODEL=model.gguf cargo test llama_scorer_smoke -- --ignored --nocapture
    #[test]
    #[ignore]
    fn llama_scorer_smoke() {
        let model = std::env::var("CSKK_ZENZ_MODEL").expect("CSKK_ZENZ_MODEL");
        let scorer = LlamaScorer::load(&model).unwrap();
        let scores = scorer
            .score("私は今日は三", "カイ", &["会".to_string(), "回".to_string(), "開".to_string(), "回".to_string()])
            .unwrap();
        println!("scores = {:?}", scores);
        assert!(scores[1] > scores[0], "回 should beat 会");
        // KV巻き戻しが正しければ同一候補のスコアは同一になる
        assert!(
            (scores[1] - scores[3]).abs() < 1e-4,
            "rewind broken: {} vs {}",
            scores[1],
            scores[3]
        );
    }
}
