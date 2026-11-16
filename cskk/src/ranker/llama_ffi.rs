//! llama.cpp 動的ロード (dlopen) バインディング。
//!
//! llama.cpp v1系の安定C ABIを対象とする。ビルド時にllama.cppに依存せず、
//! 実行時に dlopen するため、llama.cppが無い環境では文脈ランキングが
//! 無効になるだけで他の機能 (従来のSKK変換) は影響を受けない。
//!
//! 対応する2つのAPI世代:
//! - 新 (0.9.x〜v1): llama_get_memory + llama_memory_seq_rm。
//!   model_params size 80 (n_gpu_layers@16), context_params size 168 (no_perf@138)
//! - 旧 (azooKeyフォーク): llama_kv_cache_seq_rm。zenz公式GGUFの
//!   gpt2-small-japanese-char トークナイザ対応。
//!   model_params size 64 (n_gpu_layers@8), context_params size 120 (no_perf@100)
//!   ※ サイズ・オフセットは各ヘッダに対するoffsetof/sizeof検証済み。
//!
//! 構造体は固定最大長バイト配列として扱う。値渡し構造体がどちらの関数でも
//! 末尾引数であるため、大きいバッファから小さい構造体を読む呼び出しは
//! x86_64/aarch64の呼出規約上安全である。llama_batch は両世代で同一
//! (size 56, n_tokens@0, token@8)。

use crate::error::CskkError;
use libloading::Library;
use std::ffi::{c_char, c_int, c_void, CString};
use std::os::raw::c_float;

pub(crate) type LlamaModel = c_void;
pub(crate) type LlamaContext = c_void;
pub(crate) type LlamaVocab = c_void;
pub(crate) type LlamaToken = i32;

/// 探索するライブラリ名 (優先順位)。
/// 1. 環境変数 CSKK_LLAMA_LIB で明示指定
/// 2. azooKeyフォークのプライベート配置 (zenz公式GGUF対応、
///    llama-cpp-azookeyパッケージが /usr/lib/llama-azookey/ に配置)
/// 3. azooKeyフォークのシステム配置
/// 4. システムのllama.cpp (mainline互換モデル向け)
const LIBRARY_CANDIDATES: &[&str] = &[
    "/usr/lib/llama-azookey/libllama.so",
    "libllama-azookey.so.0",
    "libllama-azookey.so",
    "libllama.so.1",
    "libllama.so.0",
    "libllama.so",
];

const MODEL_PARAMS_MAX: usize = 80;
const CONTEXT_PARAMS_MAX: usize = 168;

/// C構造体をバイト列として扱うラッパー。値渡し時にスタックアラインメントが
/// C構造体 (align 8) と一致するよう repr(align(8)) を付ける。
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct ModelParamsBuf([u8; MODEL_PARAMS_MAX]);

#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct ContextParamsBuf([u8; CONTEXT_PARAMS_MAX]);

#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct BatchBuf([u8; 56]);
const CONTEXT_PARAMS_N_CTX: usize = 0;
const CONTEXT_PARAMS_N_SEQ_MAX: usize = 12;

/// API世代。
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum LlamaApiGen {
    /// 0.9.x〜v1
    New,
    /// azooKeyフォーク (旧KV API)
    Old,
}

impl LlamaApiGen {
    fn model_n_gpu_layers(self) -> usize {
        match self {
            LlamaApiGen::New => 16,
            LlamaApiGen::Old => 8,
        }
    }
    fn context_no_perf(self) -> usize {
        match self {
            LlamaApiGen::New => 138,
            LlamaApiGen::Old => 100,
        }
    }
}

/// 読み込み済みllama.cppの関数群。
pub(crate) struct LlamaApi {
    _lib: Library,
    pub(crate) gen: LlamaApiGen,
    backend_init: unsafe extern "C" fn(),
    model_default_params: unsafe extern "C" fn() -> ModelParamsBuf,
    model_load_from_file: unsafe extern "C" fn(
        *const c_char,
        ModelParamsBuf,
    ) -> *mut LlamaModel,
    model_free: unsafe extern "C" fn(*mut LlamaModel),
    model_get_vocab: unsafe extern "C" fn(*const LlamaModel) -> *const LlamaVocab,
    model_meta_val_str: unsafe extern "C" fn(
        *const LlamaModel,
        *const c_char,
        *mut c_char,
        usize,
    ) -> c_int,
    context_default_params: unsafe extern "C" fn() -> ContextParamsBuf,
    init_from_model: unsafe extern "C" fn(
        *const LlamaModel,
        ContextParamsBuf,
    ) -> *mut LlamaContext,
    free: unsafe extern "C" fn(*mut LlamaContext),
    vocab_n_tokens: unsafe extern "C" fn(*const LlamaVocab) -> c_int,
    vocab_eos: unsafe extern "C" fn(*const LlamaVocab) -> LlamaToken,
    tokenize: unsafe extern "C" fn(
        *const LlamaVocab,
        *const c_char,
        c_int,
        *mut LlamaToken,
        c_int,
        bool,
        bool,
    ) -> c_int,
    batch_get_one: unsafe extern "C" fn(
        *mut LlamaToken,
        c_int,
        *mut i32,
        *mut i32,
    ) -> BatchBuf,
    decode: unsafe extern "C" fn(*mut LlamaContext, BatchBuf) -> c_int,
    get_logits_ith: unsafe extern "C" fn(*mut LlamaContext, c_int) -> *mut c_float,
    /// 旧APIのKV巻き戻し (新APIでは未使用)
    kv_cache_seq_rm: unsafe extern "C" fn(*mut LlamaContext, i32, i32, i32) -> bool,
    /// 新APIのKV巻き戻し (旧APIでは未使用)
    get_memory: unsafe extern "C" fn(*mut LlamaContext) -> *mut c_void,
    memory_seq_rm: unsafe extern "C" fn(*mut c_void, i32, i32, i32) -> bool,
}

impl LlamaApi {
    /// llama.cppをdlopenする。見つからない場合は Err。
    pub(crate) fn load() -> Result<Self, CskkError> {
        let mut names: Vec<String> = Vec::new();
        if let Ok(explicit) = std::env::var("CSKK_LLAMA_LIB") {
            names.push(explicit);
        }
        names.extend(LIBRARY_CANDIDATES.iter().map(|s| s.to_string()));

        let mut last_err = String::new();
        for name in names {
            match unsafe { Library::new(&name) } {
                Ok(lib) => {
                    log::info!("libcskk: loaded llama.cpp from {}", name);
                    return Self::from_library(lib);
                }
                Err(e) => {
                    last_err = format!("{}: {}", name, e);
                }
            }
        }
        Err(CskkError::Error(format!(
            "libllama not found (llama.cpp required for zenz ranking): {}",
            last_err
        )))
    }

    fn from_library(lib: Library) -> Result<Self, CskkError> {
        // シンボルは生ポインタで取得し、型付き関数ポインタへtransmuteする
        // (libloadingの型チェックに左右されない定石的手法)。
        unsafe fn rawsym(lib: &Library, name: &[u8]) -> Result<*mut c_void, CskkError> {
            let symbol = lib
                .get::<*mut c_void>(name)
                .map_err(|e| {
                    CskkError::Error(format!(
                        "dlsym {}: {}",
                        String::from_utf8_lossy(name.strip_suffix(b"\0").unwrap_or(name)),
                        e
                    ))
                })?;
            Ok(*symbol)
        }
        macro_rules! raw {
            ($name:literal) => {
                rawsym(&lib, $name)?
            };
        }
        macro_rules! fnsym {
            ($t:ty, $name:literal) => {
                unsafe { std::mem::transmute::<*mut c_void, $t>(raw!($name)) }
            };
        }
        // 世代判定: 新APIの llama_memory_seq_rm があるか
        let gen = if unsafe {
            lib.get::<*mut c_void>(b"llama_memory_seq_rm\0")
        }
        .is_ok()
        {
            LlamaApiGen::New
        } else {
            LlamaApiGen::Old
        };
        log::info!("libcskk: llama.cpp API generation = {:?}", gen);

        unsafe extern "C" fn kv_dummy(_: *mut LlamaContext, _: i32, _: i32, _: i32) -> bool {
            false
        }
        unsafe extern "C" fn mem_dummy(_: *mut LlamaContext) -> *mut c_void {
            std::ptr::null_mut()
        }
        unsafe extern "C" fn seq_dummy(_: *mut c_void, _: i32, _: i32, _: i32) -> bool {
            false
        }

        let (kv_cache_seq_rm, get_memory, memory_seq_rm): (
            unsafe extern "C" fn(*mut LlamaContext, i32, i32, i32) -> bool,
            unsafe extern "C" fn(*mut LlamaContext) -> *mut c_void,
            unsafe extern "C" fn(*mut c_void, i32, i32, i32) -> bool,
        ) = match gen {
            LlamaApiGen::Old => (
                fnsym!(unsafe extern "C" fn(*mut LlamaContext, i32, i32, i32) -> bool, b"llama_kv_cache_seq_rm\0"),
                mem_dummy,
                seq_dummy,
            ),
            LlamaApiGen::New => (
                kv_dummy,
                fnsym!(unsafe extern "C" fn(*mut LlamaContext) -> *mut c_void, b"llama_get_memory\0"),
                fnsym!(unsafe extern "C" fn(*mut c_void, i32, i32, i32) -> bool, b"llama_memory_seq_rm\0"),
            ),
        };

        Ok(LlamaApi {
            gen,
            backend_init: fnsym!(unsafe extern "C" fn(), b"llama_backend_init\0"),
            model_default_params: fnsym!(
                unsafe extern "C" fn() -> ModelParamsBuf,
                b"llama_model_default_params\0"
            ),
            model_load_from_file: fnsym!(
                unsafe extern "C" fn(*const c_char, ModelParamsBuf) -> *mut LlamaModel,
                b"llama_model_load_from_file\0"
            ),
            model_free: fnsym!(unsafe extern "C" fn(*mut LlamaModel), b"llama_model_free\0"),
            model_get_vocab: fnsym!(
                unsafe extern "C" fn(*const LlamaModel) -> *const LlamaVocab,
                b"llama_model_get_vocab\0"
            ),
            model_meta_val_str: fnsym!(
                unsafe extern "C" fn(*const LlamaModel, *const c_char, *mut c_char, usize) -> c_int,
                b"llama_model_meta_val_str\0"
            ),
            context_default_params: fnsym!(
                unsafe extern "C" fn() -> ContextParamsBuf,
                b"llama_context_default_params\0"
            ),
            init_from_model: fnsym!(
                unsafe extern "C" fn(*const LlamaModel, ContextParamsBuf) -> *mut LlamaContext,
                b"llama_init_from_model\0"
            ),
            free: fnsym!(unsafe extern "C" fn(*mut LlamaContext), b"llama_free\0"),
            vocab_n_tokens: fnsym!(
                unsafe extern "C" fn(*const LlamaVocab) -> c_int,
                b"llama_vocab_n_tokens\0"
            ),
            vocab_eos: fnsym!(
                unsafe extern "C" fn(*const LlamaVocab) -> LlamaToken,
                b"llama_vocab_eos\0"
            ),
            tokenize: fnsym!(
                unsafe extern "C" fn(
                    *const LlamaVocab,
                    *const c_char,
                    c_int,
                    *mut LlamaToken,
                    c_int,
                    bool,
                    bool,
                ) -> c_int,
                b"llama_tokenize\0"
            ),
            batch_get_one: fnsym!(
                unsafe extern "C" fn(*mut LlamaToken, c_int, *mut i32, *mut i32) -> BatchBuf,
                b"llama_batch_get_one\0"
            ),
            decode: fnsym!(
                unsafe extern "C" fn(*mut LlamaContext, BatchBuf) -> c_int,
                b"llama_decode\0"
            ),
            get_logits_ith: fnsym!(
                unsafe extern "C" fn(*mut LlamaContext, c_int) -> *mut c_float,
                b"llama_get_logits_ith\0"
            ),
            kv_cache_seq_rm,
            get_memory,
            memory_seq_rm,
            _lib: lib,
        })
    }
}

/// llama.cppを使ったzenzスコアリングの実体。
/// 1つのllama_contextを持ち、ZenzScorerの実装側で直列化して使う。
pub(crate) struct LlamaRuntime {
    api: LlamaApi,
    model: *mut LlamaModel,
    context: *mut LlamaContext,
    vocab: *const LlamaVocab,
    vocab_n: usize,
    eos: LlamaToken,
}

// llama.cppのcontextは単一スレッドからのみ使う。LlamaRuntimeは
// Mutexで保護された単一ワーカースレッドからのみアクセスされる。
unsafe impl Send for LlamaRuntime {}

/// logits行から target のlog確率を返す (log_softmax)。
fn logprob_from_row(row: &[f32], target: LlamaToken) -> f64 {
    let maxv = row
        .iter()
        .map(|v| *v as f64)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let sum: f64 = row.iter().map(|v| (*v as f64 - maxv).exp()).sum();
    let logz = maxv + sum.ln();
    row[target as usize] as f64 - logz
}

/// batchのlogitsフィールド (@48) にflagsへのポインタを書き込む。
/// flagsはdecode呼び出し中のみ参照されるため呼び出しスコープで有効ならよい。
unsafe fn poke_batch_logits(batch: &mut BatchBuf, flags: &[i8]) {
    let ptr = flags.as_ptr() as u64;
    batch.0[48..56].copy_from_slice(&ptr.to_ne_bytes());
}

impl LlamaRuntime {
    /// モデルをロードする。n_gpu_layers = -1 で全レイヤーGPUオフロードを試み、
    /// 無理な場合はllama.cppが自動的にCPUへフォールバックする (Vulkan/CPU)。
    pub(crate) fn new(model_path: &str, n_ctx: u32) -> Result<Self, CskkError> {
        let api = LlamaApi::load()?;
        unsafe {
            (api.backend_init)();
            let mut mparams = (api.model_default_params)();
            let gpu = api.gen.model_n_gpu_layers();
            mparams.0[gpu..gpu + 4].copy_from_slice(&(-1i32).to_ne_bytes());
            let path = CString::new(model_path)
                .map_err(|e| CskkError::Error(format!("model path: {}", e)))?;
            let model = (api.model_load_from_file)(path.as_ptr(), mparams);
            if model.is_null() {
                return Err(CskkError::Error(format!(
                    "failed to load model: {} (mainlineのllama.cppではzenz公式GGUFを読めません。azooKey/llama.cppフォーク (libllama-azookey) が必要です)",
                    model_path
                )));
            }
            let vocab = (api.model_get_vocab)(model);
            let vocab_n = (api.vocab_n_tokens)(vocab) as usize;
            let eos = (api.vocab_eos)(vocab);

            let mut cparams = (api.context_default_params)();
            cparams.0[CONTEXT_PARAMS_N_CTX..CONTEXT_PARAMS_N_CTX + 4]
                .copy_from_slice(&n_ctx.to_ne_bytes());
            cparams.0[CONTEXT_PARAMS_N_SEQ_MAX..CONTEXT_PARAMS_N_SEQ_MAX + 4]
                .copy_from_slice(&1u32.to_ne_bytes());
            let no_perf = api.gen.context_no_perf();
            cparams.0[no_perf] = 1; // no_perf = true
            let context = (api.init_from_model)(model, cparams);
            if context.is_null() {
                (api.model_free)(model);
                return Err(CskkError::Error(
                    "failed to create llama context".to_string(),
                ));
            }
            Ok(LlamaRuntime {
                api,
                model,
                context,
                vocab,
                vocab_n,
                eos,
            })
        }
    }

    /// GGUFメタデータ (general.name 等) を取得する。
    pub(crate) fn metadata(&self, key: &str) -> Option<String> {
        let mut buf = vec![0u8; 256];
        let ckey = CString::new(key).ok()?;
        let n = unsafe {
            (self.api.model_meta_val_str)(
                self.model,
                ckey.as_ptr(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
            )
        };
        if n <= 0 {
            return None;
        }
        buf.truncate(n as usize);
        Some(String::from_utf8_lossy(&buf).to_string())
    }

    fn tokenize(&self, text: &str, buf: &mut Vec<LlamaToken>) -> Result<(), CskkError> {
        let bytes = text.as_bytes();
        for _ in 0..2 {
            let n = unsafe {
                (self.api.tokenize)(
                    self.vocab,
                    bytes.as_ptr() as *const c_char,
                    bytes.len() as c_int,
                    buf.as_mut_ptr(),
                    buf.capacity() as c_int,
                    false,
                    true,
                )
            };
            if n >= 0 {
                unsafe { buf.set_len(n as usize) };
                return Ok(());
            }
            buf.reserve_exact((-n) as usize);
        }
        Err(CskkError::Error("llama_tokenize failed".to_string()))
    }

    fn decode_tokens(&mut self, tokens: &[LlamaToken]) -> Result<(), CskkError> {
        // n_batchはデフォルト (1024以上) を想定し512ずつ安全に分割。
        // logitsは最終トークンのみ要求する (バッチ内の最後のフラグを立てる)。
        const CHUNK: usize = 512;
        for chunk in tokens.chunks(CHUNK) {
            let last = chunk.len() - 1;
            let mut flags = vec![0i8; chunk.len()];
            flags[last] = 1;
            unsafe {
                let mut batch = (self.api.batch_get_one)(
                    chunk.as_ptr() as *mut LlamaToken,
                    chunk.len() as c_int,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
                poke_batch_logits(&mut batch, &flags);
                if (self.api.decode)(self.context, batch) != 0 {
                    return Err(CskkError::Error("llama_decode failed".to_string()));
                }
            }
        }
        Ok(())
    }

    fn decode_one(&mut self, token: LlamaToken) -> Result<(), CskkError> {
        let mut flags = [1i8];
        unsafe {
            let mut batch = (self.api.batch_get_one)(
                &token as *const LlamaToken as *mut LlamaToken,
                1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            poke_batch_logits(&mut batch, &flags);
            if (self.api.decode)(self.context, batch) != 0 {
                return Err(CskkError::Error("llama_decode failed".to_string()));
            }
        }
        Ok(())
    }

    /// 現在の最終位置のlogitsから target のlog確率を返す。
    /// 直前のdecodeでは最後のトークンのlogitsのみ要求しているため、
    /// インデックス -1 (出力の末尾) を使う。
    fn logprob_of(&mut self, target: LlamaToken) -> Result<f64, CskkError> {
        let row = self.logits_row()?;
        Ok(logprob_from_row(&row, target))
    }

    /// 現在の最終位置のlogits行をコピーして返す。
    fn logits_row(&mut self) -> Result<Vec<f32>, CskkError> {
        unsafe {
            let logits = (self.api.get_logits_ith)(self.context, -1);
            if logits.is_null() {
                return Err(CskkError::Error("no logits".to_string()));
            }
            let slice = std::slice::from_raw_parts(logits, self.vocab_n);
            Ok(slice.to_vec())
        }
    }

    /// KVキャッシュを position p0 以降を削除する (p0=-1で全クリア)。
    fn rewind_kv(&mut self, p0: i32) {
        unsafe {
            match self.api.gen {
                LlamaApiGen::New => {
                    let mem = (self.api.get_memory)(self.context);
                    (self.api.memory_seq_rm)(mem, 0, p0, -1);
                }
                LlamaApiGen::Old => {
                    (self.api.kv_cache_seq_rm)(self.context, 0, p0, -1);
                }
            }
        }
    }

    /// 「promptの続きとしてcandidateが現れる」log確率の総和 (EOS込み)。
    /// prompt_row はprompt末尾位置のlogits (候補1トークン目の評価に使う。
    /// 前候補のスコアリング後もlogitsバッファは書き換わっているため)。
    /// 実行後、KVキャッシュはprompt末尾まで巻き戻される。
    pub(crate) fn score_continuation(
        &mut self,
        prompt_len: i32,
        prompt_row: &[f32],
        candidate: &str,
    ) -> Result<f64, CskkError> {
        let mut cand_tokens: Vec<LlamaToken> = Vec::with_capacity(64);
        self.tokenize(candidate, &mut cand_tokens)?;
        if cand_tokens.is_empty() {
            return Ok(f64::NEG_INFINITY);
        }
        let mut score = 0.0;
        for (i, &token) in cand_tokens.iter().enumerate() {
            score += if i == 0 {
                logprob_from_row(prompt_row, token)
            } else {
                self.logprob_of(token)?
            };
            self.decode_one(token)?;
        }
        score += self.logprob_of(self.eos)?;
        self.rewind_kv(prompt_len);
        Ok(score)
    }

    /// promptをデコード (KVに載せる)。返り値はpromptのトークン数と、
    /// prompt末尾位置のlogits行。
    pub(crate) fn prepare_prompt(&mut self, prompt: &str) -> Result<(i32, Vec<f32>), CskkError> {
        let mut tokens: Vec<LlamaToken> = Vec::with_capacity(256);
        self.tokenize(prompt, &mut tokens)?;
        self.decode_tokens(&tokens)?;
        let row = self.logits_row()?;
        Ok((tokens.len() as i32, row))
    }

    /// KVキャッシュを全クリアする。
    pub(crate) fn clear_cache(&mut self) {
        self.rewind_kv(-1);
    }
}

impl Drop for LlamaRuntime {
    fn drop(&mut self) {
        unsafe {
            (self.api.free)(self.context);
            (self.api.model_free)(self.model);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 実モデルでのスモークテスト (通常のcargo testではスキップ)。
    /// 実行: CSKK_ZENZ_MODEL=model.gguf cargo test llama_runtime_smoke -- --ignored --nocapture
    #[test]
    #[ignore]
    fn llama_runtime_smoke() {
        let model = std::env::var("CSKK_ZENZ_MODEL").expect("CSKK_ZENZ_MODEL");
        let mut rt = LlamaRuntime::new(&model, 512).expect("runtime");
        println!("metadata name: {:?}", rt.metadata("general.name"));
        let prompt = format!(
            "{}{}{}{}{}",
            '\u{EE02}', "私は今日は三", '\u{EE00}', "カイ", '\u{EE01}'
        );
        let (len, row) = rt.prepare_prompt(&prompt).expect("prompt");
        for candidate in ["回", "会", "開", "回"] {
            let score = rt.score_continuation(len, &row, candidate).expect("score");
            println!("score({}) = {:.3}", candidate, score);
        }
    }
}
