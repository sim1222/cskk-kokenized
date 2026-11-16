# 文脈適応型変換 (Context-aware candidate re-ranking) — 設計

## 目的
Space 変換時に、カーソル周辺の文章 (文脈) に基づいて変換候補を自動再ランキングし、
最も適切な候補を先頭に表示する。SKK の操作体系・辞書・学習機能との互換性を維持する。

## 最重要要件
1. 推論によって入力操作をブロックしないこと。
2. 推論結果が利用できない場合は従来の SKK 変換動作へフォールバックすること
   (候補順序は現行と完全同一)。
3. ランカー無効化時の動作は現行と完全同一であること。

## アーキテクチャ (決定済み: ハイブリッド)

```
fcitx5-cskk (C++)                      libcskk (Rust, fork)
┌─────────────────────────┐            ┌──────────────────────────────┐
│ FcitxCskkContext        │            │ CskkContext                  │
│  - SurroundingText取得   │  C API     │  - surrounding_text: String │
│  - 自己コミット履歴(代替) │ ──────────▶ │  - context_ranking_enabled   │
│  - 設定 (config)         │            │  update_candidate_list()     │
│  - 候補リスト表示         │            │    └▶ ranker.rerank(...)     │
└─────────────────────────┘            │  ranker: ContextRanker trait │
                                       │   - IdentityRanker (no-op)   │
                                       │   - NgramRanker (phase B)    │
                                       │   - RemoteRanker (phase C)   │
                                       └──────────────────────────────┘
```

- 文脈取得・候補表示は C++ 側、スコアリング本体は Rust 側。
- 再ランキングは libcskk 内部で行う。理由: preedit と候補ウィンドウの候補位置は
  共に libcskk の候補リスト状態から生成されるため、C++ 側で順序だけ差し替えると
  preedit とウィンドウが不一致になる (cskkcandidatelist.cpp は index ベースで
  `skk_context_select_candidate_at` を呼ぶ)。
- 推論バックエンドは段階的ハイブリッド: デフォルトは軽量統計モデル (インプロセス)。
  設定により外部推論サーバー (HTTP 非同期) を有効化可能 (phase C)。

## C API 契約 (libcskk/cskk/src/capi.rs に追加)

```c
// カーソルより前の確定済みテキスト (UTF-8, 最大でも直近N文字程度) を設定する。
// NULL または空文字でクリア。変換 (候補リスト更新) の前に呼ばれることを想定。
void skk_context_set_surrounding_text(CskkContext *context, const char *text_before_cursor);

// 文脈再ランキングの有効/無効。デフォルト: 無効 (従来動作)。
void skk_context_set_context_ranking_enabled(CskkContext *context, bool enabled);
bool skk_context_get_context_ranking_enabled(const CskkContext *context);
```

実装メモ:
- cbindgen は Rust `bool` を C `bool` (stdbool.h) に map する。既存の
  `skk_context_select_candidate_at` の戻り値と同じ扱い。
- 呼び出しタイミングは C++ 側の `FcitxCskkContext::keyEvent` の冒頭で良い
  (変換確定前ならいつでもよい。毎キー更新して librskk 側は保持するだけ)。

## libcskk 側実装

### 統合ポイント (調査済み)
- 候補順序の決定: `cskk/src/dictionary/mod.rs:132` `get_all_candidates()`
  (辞書順: ユーザー辞書 → 静的辞書、エントリ順)。
- 変換開始時の呼び出し: `cskk/src/lib.rs:467` `update_candidate_list()`。
- 学習: `lib.rs:509` `confirm_current_composition_candidate()` →
  `dictionary/mod.rs:94` `confirm_candidate()` → `user_dictionary.rs:118`
  `select_candidate()` (LruOrderedMap: 最近使用が先頭へ)。
- C API: index ベース (`skk_context_get_current_candidates`,
  `skk_context_select_candidate_at`, `skk_context_confirm_candidate_at`)。

### ranker モジュール (新規 `cskk/src/ranker/`)
```rust
pub(crate) trait ContextRanker: Send {
    /// before: カーソル前テキスト。candidates をその場で並べ替える。
    /// 失敗時は Err を返し、呼び出し側は元の順序を保持する。
    fn rerank(&self, before: &str, candidates: &mut [Candidate]) -> Result<(), CskkError>;
}
```
- `IdentityRanker`: なにもしない (phase A のデフォルト)。
- `NgramRanker` (phase B): 共起/N-gram 統計 + 個人変換履歴の重み付け。
  スコアリングはマイクロ秒オーダーのため同期実行でよい。
- `RemoteRanker` (phase C): HTTP 非同期 + タイムアウト (数十ms)。
  タイムアウト時は元の順序。

### update_candidate_list() の変更
```rust
fn update_candidate_list(&mut self) {
    let composite_key = ...;
    let mut candidates = get_all_candidates(&self.dictionaries, &composite_key);
    if self.context_ranking_enabled {
        if !self.surrounding_text.is_empty() {
            if let Some(ranker) = &self.context_ranker {
                // 失敗・パニック時は元の順序を維持 (フォールバック要件)
                let _ = ranker.rerank(&self.surrounding_text, &mut candidates);
            }
        }
    }
    self.current_state().set_new_candidate_list(candidates);
}
```

## fcitx5-cskk 側実装

### 文脈取得 (`src/cskk.cpp`)
- `FcitxCskkContext::keyEvent()` 冒頭で:
  1. `ic_->surroundingText()` が `valid()` なら、`text()` の先頭から `cursor()`
     文字目までを取り出し (直近 ~64 文字に切り詰め)、
     `skk_context_set_surrounding_text()` に渡す。
  2. 無効ならフォールバック: 自エンジンのコミット文字列のリングバッファ
     (`updateUI()` 内 `ic_->commitString(output)` の箇所で蓄積、
     InputContext ごとに保持) を渡す。
- SurroundingText API (fcitx5 5.1.23 ヘッダで確認済み):
  `valid()`, `text()`, `cursor()`, `anchor()`。text は UTF-8、
  cursor は文字数オフセット → バイト位置への変換が必要。

### 設定 (`src/cskkconfig.h`)
- `contextRankingEnabled` (bool, デフォルト: true — 失敗時は自動フォールバックする
  ため安全性は担保される。ただし phase A の IdentityRanker は順序を変えないため
  有効化しても挙動は同一)。
- 後段 (phase C) で `rankingBackend` (ngram / remote) 設定を追加。

## フェーズ計画

| Phase | 内容 | 検証 | 状態 |
|-------|------|------|------|
| A | libcskk: surrounding text API + rerank フック + IdentityRanker + 有効化フラグ。fcitx5-cskk: 文脈取得・自己コミット履歴・config | 両リポジトリの全テスト維持、有効化しても候補順序が変わらないこと | 完了 |
| B | ~~NgramRanker~~ → Phase C でzenzスコアリングに置換 (削除済み) | — | 置換済み |
| C | zenz系モデル (llama.cpp) によるスコアリング。prefetchによる遅延隠蔽、個人履歴との合成 | 実測: 「今日は三」+かい→回、「飛行」+き→機、変換0.02ms/回 | 完了 |
| D | ベンチ (top-1 命中率、打鍵数、レイテンシ) | 計測レポート | 未 |

## Phase C 実装 (現行)

スコアリング本体は **zenz系モデル + llama.cpp**:
- プロンプト: `U+EE02 + 左文脈(40字) + U+EE00 + 読み(カタカナ) + U+EE01`
  (zenz-v3.x / jinen 形式。zenz-v2/2.5 は `EE00+読み+EE02+文脈+EE01` で
  GGUFの general.name から自動判別)
- 候補スコア = プロンプト続きのlog確率の総和 (EOS込み)
- **遅延隠蔽**: 読み入力中 (▽モード) に上位10候補のスコアリングを
  バックグラウンド開始 (`ContextRanker::prefetch`)。変換時はキャッシュを
  最大50ms待ち、間に合わなければ辞書順を維持
- モデルロードもバックグラウンド (プロセスで1回、キャッシュ共有)。
  Vulkan利用時は全レイヤーオフロード (`n_gpu_layers=-1`)、無理ならCPU
- llama.cppは実行時にdlopen (ビルド依存なし)。2つのAPI世代に対応:
  mainline (0.9.x〜v1) と azooKeyフォーク (zenz公式GGUFの
  gpt2-small-japanese-char トークナイザ対応、`/usr/lib/llama-azookey/`)
- モデル配置: XDGデータディレクトリの `libcskk/zenz.gguf`
  (パッケージ: libcskk-koke-zenz-model = jinen-v1-xsmall 26M)

### 個人履歴 (継続)
- 確定時に `(文脈末尾4文字, 候補output)` を履歴に記録
  (`$XDG_DATA_HOME/libcskk/context_ranking_history.bin`、N-gram時代と同一形式)。
- スコア = `zenzスコア + 2.0·log(1+履歴回数)`。使い続けるほど個人適応が効く。

### 安全機構
- 元の順位をタイブレーカーとする安定ソート: 同スコア/スコア未取得の候補は
  辞書順を維持 (下位グループに沈めない)。
- モデル・llama.cpp不在、ロード失敗、スコア未到達の場合はすべて従来SKK動作。



## ビルド (ローカル開発)
- libcskk: `cargo build --release` + `cbindgen --config cbindgen.toml --output libcskk.h`
  → `/tmp/opencode/cskk-install` に .so / .h / cskk.pc を配置。
- fcitx5-cskk: `PKG_CONFIG_PATH=/tmp/opencode/cskk-install/lib/pkgconfig cmake -B build`
- ベースライン: libcskk cargo test 全緑 (77+ tests)。fcitx5-cskk は fcitx5 5.1.23
  dev ヘッダでビルド可能。
