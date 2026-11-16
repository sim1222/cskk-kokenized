# cskk-kokenized

[naokiri/cskk](https://github.com/naokiri/cskk) のごった混ぜ fork です。

## 追加機能

- **文脈適応型変換候補再ランキング**: 変換時に、カーソル前の文脈から
  変換候補を自動で並べ替えます。かな漢字変換モデル [zenzai (zenz)](https://huggingface.co/collections/Miwa-Keita/zenz-v25-6784cd5d5714761bc4c303e)
  を [llama.cpp](https://github.com/ggml-org/llama.cpp) でローカル実行し、
  各候補の続きやすさ (条件付きlog確率) で順序付けします。
  - 読み入力中の prefetch による遅延隠蔽、変換時は最大150msだけ待機
  - モデル・llama.cpp が無い環境では従来のSKK変換にフォールバック
  - 既存の辞書・学習・操作体系はそのまま
  - 送り仮名込みのスコアリング、個人変換履歴との合成
  - llama.cpp は実行時に dlopen (ビルド依存なし)。Vulkan/CPU
- 設計の詳細は [docs/context-ranking-design.md](docs/context-ranking-design.md) を参照

## 開発について

- 本 fork の開発には Coding Agent を利用しています。

---

![cskk logo](https://raw.githubusercontent.com/naokiri/cskk-icons/master/256x256/apps/cskk.png) 


# LibCSKK

Stable Rust version build status: 
[![Build Status](https://github.com/naokiri/cskk/workflows/Stable/badge.svg)](https://github.com/naokiri/cskk/actions)

Beta and Nightly Rust version build status:
[![Build Status (Beta or Nightly)](https://github.com/naokiri/cskk/workflows/Beta/badge.svg)](https://github.com/naokiri/cskk/actions)

[English version](https://github.com/naokiri/cskk/blob/master/README.en.md)

Cobalt SKK ライブラリ.

CSKK はSKK(Simple Kana Kanji 変換)用ライブラリです。
CSKKはSKKと外延的に同値であるため、こう名付けられました。

Fcitx5用: [fcitx5-cskk](https://github.com/fcitx/fcitx5-cskk)

説明書: [cskk-docs](https://naokiri.github.io/cskk-docs/)

ロゴ・アイコン類: [cskk-icons](https://github.com/naokiri/cskk-icons)

参考
- ddskk: http://openlab.ring.gr.jp/skk/ddskk.html
- libskk: https://github.com/ueno/libskk

## 必要ライブラリ類

- libxkbcommon

Ubuntu等では以下のコマンドでインストール

```shell
    sudo apt install libxkbcommon-dev
```

- cbindgen

```shell
    cargo install --force cbindgen
```

- cargo-c

```shell
    cargo install --force cargo-c
```

## インストール方法

root権限がある場合、以下を実行する。
これは`cargo cbuild --release`でtarget/{arch}/release/以下に生成されたファイルを適切なシステムのディレクトリに、そしてassets/以下のファイルを適切なデータディレクトリのlibcskk/以下にインストールする。

```shell
    cargo cinstall --release
```

標準的なパス以外にインストールする場合は、以下のような引数を与える。
詳細は [cargo-c](https://github.com/lu-zero/cargo-c) を参照のこと。

```shell
    cargo cinstall --release --prefix="/tmp" --datadir="$HOME/.local/share"
```

- prefix: libdir, includedir, pkgconfigdir, datarootdir 共通接頭部分。デフォルトは '/usr/local'
- libdir: ライブラリインストール先。デフォルトは '/lib'
- includedir: ヘッダファイルイストール先。デフォルトは '/include'
- pkgconfigdir: pkg-config用の.pcファイルインストール先。デフォルトは '/lib/pkgconfig'
- datarootdir: データファイル(cskkプロジェクトではassetsディレクトリ下)のインストール先。デフォルトは'share'
- datadir: datarootdirを上書きするデータファイルのインストール先。prefixを無視してdataのインストール先を指定できる。デフォルトは指定なし(つまりdatarootdirを用いる。)

## 開発状況

### 基本機能・DDSKKの機能

- [x] ひらがな入力
- [x] カタカナ入力・カタカナモード
- [x] ｶﾀｶﾅ入力・ｶﾀｶﾅモード
- [x] Basic 漢字変換
- [x] static dictionary
- [x] user dictionary
  - ddskk compatible since v0.11.0
- [ ] 接頭辞・接尾辞変換
- [x] 数値変換
- [x] auto-start-henkan   
- 実装見込が現在ないもの
    - [ ] Kuten 変換
    - [ ] 今日の日付入力
    - [ ] 異字体変換
    - [ ] SKK辞書サーバー対応
    - [ ] 外部辞書

## ueno/libskk の機能

- [x] 句読点設定
- [x] AZIK rule
- [ ] Nicola rule

### C FFI + IME plugin

- [x] C ABI library for fcitx5-skk
  最低限のみ。https://github.com/naokiri/fcitx5-cskk 参照。

### 開発環境・デプロイ環境

- [ ] github projects board や issue にこれらのリストを移す
- [x] changelog

## 著作権表示

Copyright (C) 2018 Naoaki Iwakiri

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see <https://www.gnu.org/licenses/>.
