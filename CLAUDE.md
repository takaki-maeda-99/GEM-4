# CLAUDE.md

## Project: VLA-Gemma4

Gemma 4 E2B backbone VLA for robot manipulation.

## Rules

### README.md の更新

コミットを作成する際、以下のいずれかに該当する変更を含む場合は **README.md も合わせて更新すること**:

- 新しいモジュール・ファイルの追加 → Project Structure セクションを更新
- 新しい機能・コマンドの追加 → Quick Start / 使い方セクションを更新
- アーキテクチャの変更 → Architecture セクションを更新
- 新しい実験結果 → Results セクションを更新
- 設定項目の追加・変更 → Configuration セクションを更新
- Roadmap 項目の完了 → チェックボックスを更新

軽微な修正（バグフィックス、リファクタリング、テスト追加のみ）の場合は更新不要。

### コーディング規約

- 日本語コメントOK（ドキュメント・README は日本語）
- アクションヘッドは `ActionHead` ABC を継承すること
- 設定は YAML ファイルで管理、ハードコードしない
- テストは pytest、モックで Gemma 4 のロードを回避
