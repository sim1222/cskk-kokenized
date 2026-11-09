//! 文脈適応型変換用の文字N-gramモデルをコーパスから構築するツール。
//!
//! 使い方:
//!   cargo run --release -- -o <出力モデル.bin> <コーパステキスト>...
//!
//! コーパスの取得 (青空文庫、パブリックドメイン):
//!   ../fetch_corpus.sh
//!
//! モデル形式は cskk::ranker::ngram_ranker::CharNgramModelFile (bincode)。
//! BTreeMap を使うため、同じ入力からは常にバイト同一のモデルが生成される (再現性)。

use cskk::ranker::ngram_ranker::{CharNgramModelFile, MODEL_CONTEXT_CHARS};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read};
use std::path::Path;

/// この回数未満の文脈は剪定する。
const DEFAULT_MIN_CONTEXT_TOTAL: u32 = 10;
/// 文脈ごとに保持する次文字の種類の上限。
const DEFAULT_MAX_NEXTS: usize = 8;

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut output_path = String::from("context_ngram_model.bin");
    let mut min_context_total = DEFAULT_MIN_CONTEXT_TOTAL;
    let mut max_nexts = DEFAULT_MAX_NEXTS;
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                output_path = args.get(i).cloned().ok_or("-o requires a path")?;
            }
            "--min-count" => {
                i += 1;
                min_context_total = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or("--min-count requires a number")?;
            }
            "--max-nexts" => {
                i += 1;
                max_nexts = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or("--max-nexts requires a number")?;
            }
            other => rest.push(other.to_string()),
        }
        i += 1;
    }
    let args = rest;
    if args.is_empty() {
        return Err("usage: build_ngram_model -o <output.bin> <corpus text or dir>...".to_string());
    }

    // 引数のディレクトリは再帰的に .txt ファイルに展開する
    let mut files: Vec<String> = Vec::new();
    for arg in &args {
        let path = Path::new(arg);
        if path.is_dir() {
            collect_txt_files(path, &mut files);
        } else {
            files.push(arg.clone());
        }
    }
    eprintln!("{} files to process", files.len());

    let mut contexts: BTreeMap<String, BTreeMap<char, u32>> = BTreeMap::new();
    let mut unigram: BTreeMap<char, u32> = BTreeMap::new();
    let mut total_chars: u64 = 0;

    for path in &files {
        let text = read_text(Path::new(path))?;
        for line in text.split('\n') {
            // 前処理: ルビ《...》、注記[#...]、CRを除去する
            let line = preprocess(line);
            let chars: Vec<char> = line.chars().collect();
            total_chars += chars.len() as u64;
            for ch in &chars {
                *unigram.entry(*ch).or_insert(0) += 1;
            }
            for i in MODEL_CONTEXT_CHARS..chars.len() {
                let context: String =
                    chars[i - MODEL_CONTEXT_CHARS..i].iter().collect();
                let next = chars[i];
                *contexts
                    .entry(context)
                    .or_default()
                    .entry(next)
                    .or_insert(0) += 1;
            }
        }
        eprintln!("processed: {}", path);
    }

    // 剪定: 合計出現数が少ない文脈を除き、各文脈は上位max_nexts文字のみ保持する
    contexts.retain(|_, nexts| nexts.values().sum::<u32>() >= min_context_total);
    for nexts in contexts.values_mut() {
        let mut sorted: Vec<(char, u32)> =
            nexts.iter().map(|(k, v)| (*k, *v)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        sorted.truncate(max_nexts);
        *nexts = sorted.into_iter().collect();
    }

    let file_model = CharNgramModelFile { contexts, unigram };
    let file = File::create(&output_path)
        .map_err(|e| format!("failed to create {}: {}", output_path, e))?;
    bincode::serialize_into(BufWriter::new(file), &file_model)
        .map_err(|e| format!("failed to serialize: {}", e))?;
    eprintln!(
        "model written to {} ({} contexts, {} chars seen)",
        output_path,
        file_model.contexts.len(),
        total_chars
    );
    Ok(())
}

/// ディレクトリを再帰的に走査して .txt ファイルのパスを集める。
fn collect_txt_files(dir: &Path, files: &mut Vec<String>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("warn: cannot read dir {:?}: {}", dir, e);
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_txt_files(&path, files);
        } else if path.extension().map(|e| e == "txt").unwrap_or(false) {
            files.push(path.to_string_lossy().to_string());
        }
    }
}

fn read_text(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| format!("failed to open {:?}: {}", path, e))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("failed to read {:?}: {}", path, e))?;
    // UTF-8を試し、だめならShift_JIS (CP932) としてデコードする。
    // 青空文庫のテキストはShift_JISが多いため。
    if let Ok(text) = std::str::from_utf8(&bytes) {
        return Ok(text.to_string());
    }
    let (text, _, had_errors) = encoding_rs::SHIFT_JIS.decode(&bytes);
    if had_errors {
        eprintln!(
            "warn: {:?} is neither UTF-8 nor clean Shift_JIS. Using best-effort decoding.",
            path
        );
    }
    Ok(text.into_owned())
}

/// 青空文庫テキストの簡易前処理。
/// ルビ《...》と注記[#...]を除去する。
fn preprocess(line: &str) -> String {
    let mut result = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '《' => {
                for c in chars.by_ref() {
                    if c == '》' {
                        break;
                    }
                }
            }
            '[' => {
                if chars.peek() == Some(&'#') {
                    chars.next();
                    for c in chars.by_ref() {
                        if c == ']' {
                            break;
                        }
                    }
                } else {
                    result.push(ch);
                }
            }
            '\r' => {}
            _ => result.push(ch),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_removes_ruby() {
        assert_eq!(preprocess("私《わたし》は"), "私は");
        assert_eq!(preprocess("注記[#footnote]本文"), "注記本文");
        assert_eq!(preprocess("改行\r\n"), "改行\n");
    }
}
