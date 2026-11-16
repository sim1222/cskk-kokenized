//! 個人変換履歴。文脈末尾→確定候補の出現回数。
//! ファイルへの保存は bincode (BTreeMap、一時ファイル + rename)。
//! N-gram時代と同じ形式なので、既存の履歴ファイルはそのまま使える。

use crate::error::CskkError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// 履歴の文脈長。
pub const HISTORY_CONTEXT_CHARS: usize = 4;
/// 履歴マッチ1件あたりのスコア加算量。
pub const HISTORY_SCORE_PER_COUNT: f64 = 2.0;

#[derive(Serialize, Deserialize, Debug, Default, PartialEq)]
pub struct HistoryFile {
    /// 文脈(HISTORY_CONTEXT_CHARS文字) → 確定候補output → 回数
    pub counts: BTreeMap<String, BTreeMap<String, u32>>,
}

#[derive(Debug)]
pub(crate) struct HistoryStore {
    counts: HashMap<String, HashMap<String, u32>>,
    path: Option<PathBuf>,
}

impl HistoryStore {
    pub(crate) fn from_file_model(file_model: &HistoryFile, path: Option<PathBuf>) -> Self {
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

    pub(crate) fn load(path: Option<&Path>) -> Result<Self, CskkError> {
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

    pub(crate) fn add(&mut self, before_text: &str, output: &str) {
        let context = tail_chars(before_text, HISTORY_CONTEXT_CHARS).to_string();
        self.counts
            .entry(context)
            .or_default()
            .entry(output.to_string())
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    pub(crate) fn get_count(&self, before_text: &str, output: &str) -> u32 {
        let context = tail_chars(before_text, HISTORY_CONTEXT_CHARS);
        self.counts
            .get(context)
            .and_then(|outputs| outputs.get(output))
            .copied()
            .unwrap_or(0)
    }

    pub(crate) fn save(&self) -> Result<(), CskkError> {
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

/// 履歴スコア (文脈末尾と候補の一致回数に基づく)。
pub(crate) fn history_score(history: &HistoryStore, before_text: &str, output: &str) -> f64 {
    let count = history.get_count(before_text, output);
    HISTORY_SCORE_PER_COUNT * (count as f64).ln_1p()
}

/// 確定を履歴に記録して保存する。
pub(crate) fn record_and_save(history: &Mutex<HistoryStore>, before_text: &str, output: &str) {
    if let Ok(mut store) = history.lock() {
        store.add(before_text, output);
        if let Err(e) = store.save() {
            log::warn!("Failed to save context ranking history: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_chars_returns_suffix() {
        assert_eq!(tail_chars("あいうえお", 2), "えお");
        assert_eq!(tail_chars("あい", 4), "あい");
        assert_eq!(tail_chars("", 4), "");
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
    fn history_score_grows_with_count() {
        let store = HistoryStore::from_file_model(&HistoryFile::default(), None);
        assert_eq!(history_score(&store, "文脈", "候補"), 0.0);
    }
}
