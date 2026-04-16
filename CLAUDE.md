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

### トラブルシューティングログの更新

バグ修正や問題解決を行った際は、**`docs/troubleshooting.md` に記録を追加すること**:

- 問題の現象
- 原因
- 対策
- 教訓（同種の問題を防ぐための知見があれば）

学習の発散、推論の異常動作、環境構築の問題など、デバッグに時間がかかった問題は必ず記録する。
