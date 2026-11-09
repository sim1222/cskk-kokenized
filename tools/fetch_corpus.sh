#!/bin/bash
# 文脈適応型変換用コーパスの取得スクリプト (青空文庫、パブリックドメイン)。
#
# aozorabunko/aozorabunko 公式リポジトリは巨大なため、テキストのみを抽出した
# aozorahack/aozorabunko_text (青空文庫公認の派生、毎日更新) を使用する。
#
# 再現性: クローン時点のコミットを corpus_commit.txt に記録する。
# モデル再構築時は同じコミットを使うこと。
#
# 使い方:
#   ./fetch_corpus.sh [出力ディレクトリ (デフォルト: ./corpus)]

set -euo pipefail

OUTPUT_DIR="${1:-corpus}"
REPO_URL="https://github.com/aozorahack/aozorabunko_text"

mkdir -p "$OUTPUT_DIR"
if [ ! -d "$OUTPUT_DIR/aozorabunko_text" ]; then
  git clone --depth 1 "$REPO_URL" "$OUTPUT_DIR/aozorabunko_text"
fi

# クローン時点のコミットを記録して再現可能にする
git -C "$OUTPUT_DIR/aozorabunko_text" rev-parse HEAD > "$OUTPUT_DIR/corpus_commit.txt"
echo "corpus fetched: $(cat "$OUTPUT_DIR/corpus_commit.txt")"
echo "text files: $(find "$OUTPUT_DIR/aozorabunko_text" -name '*.txt' | wc -l)"
