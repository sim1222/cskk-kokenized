# 文脈適応型変換ツール群

libcskk/fcitx5-cskk の文脈適応型変換 (context-aware candidate re-ranking) 用の
モデル生成パイプライン。設計は `../../docs/context-ranking-design.md` を参照。

## モデルの再現可能な構築手順

### 1. コーパスの取得

青空文庫 (パブリックドメイン) を使用する。

```shell
./fetch_corpus.sh            # ./corpus/ にクローン、コミットを記録
# または出力先を指定:
./fetch_corpus.sh /path/to/corpus
```

取得時点のコミットハッシュが `corpus/corpus_commit.txt` に記録される。
モデルの再現には同じコミットを使用すること。

### 2. モデルの構築

```shell
cd build_ngram_model
cargo run --release -- -o context_ngram_model.bin ../corpus/aozorabunko_text
```

ディレクトリを渡すと配下の .txt を再帰的に処理する。

- 形式: 文字5-gram (4文字の文脈から次の1文字を予測)、bincode、BTreeMap による
  決定的シリアライズ (同じコーパスから常にバイト同一のモデルが生成される)
- ルビ《...》と注記[#...]は構築時に除去される
- コーパスはUTF-8/Shift_JIS (CP932) 両対応 (青空文庫のテキストはShift_JISが多い)

### 3. インストール

libcskk は以下の順でモデルを探す (`skk_context_use_ngram_ranker` に空パスを渡した場合):

1. `$XDG_DATA_HOME/libcskk/context_ngram_model.bin` (デフォルト `~/.local/share/...`)
2. `$XDG_DATA_DIRS` 各ディレクトリ配下の `libcskk/context_ngram_model.bin`
   (システムインストール時は `/usr/share/libcskk/context_ngram_model.bin` 等)

```shell
mkdir -p ~/.local/share/libcskk
cp context_ngram_model.bin ~/.local/share/libcskk/
```

個人変換履歴は `$XDG_DATA_HOME/libcskk/context_ranking_history.bin` に自動保存される。

### fcitx5-cskk 側の設定

- `ContextAwareRanking` (bool, デフォルト ON): 文脈適応型変換の有効/無効
- `ContextAwareRankingModelPath` (string, デフォルト空): 空の場合は上記 XDG
  から自動検索

モデルが存在しない場合は従来のSKK変換動作にフォールバックする
(何も設定しなくても壊れない)。
