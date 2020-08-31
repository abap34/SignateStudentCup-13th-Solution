# SignateStudentCup-13th-Solution
 
Signate Student Cup 2020予測部門(https://signate.jp/competitions/281) の13位解法です。


# 概要

まずtrain, test共に再翻訳によるデータの水増しを行いました。

経由した言語はスペイン語、ドイツ語、ロシア語の3つで、googletransを使用しました。

学習に使ったモデルは、BERT, XLNet, RoBERTaの3つです。
特にXLNetの追加はPrivate scoreの向上に大きく寄与していました。

ハイパーパラメータについてですが、
`max_seq_length`を大きくしたところPublic scoreが大きく上昇したため、そのままの設定で行きました。(learning rate を下げればバッチサイズを小さめにしても安定して学習できることも考慮しました。)

ですが、Private scoreにはあまり貢献していませんでした.....

NLPをわかっている人から見ると相当頓珍漢なことをやっている自信がありますが、優しくマサカリを投げてくださると嬉しいです。

また、このレポジトリのコードはGoogleColabを使用していた関係で多少書き換えられています。

簡単に動作確認をしましたが、計算環境の関係で全体を確認できていません。
(なのでエラーが出るかもしれません。その場合はissueを建ててくださると泣いて喜びます。)


# 実行
```
pipenv sync
pipenv run python src/translate_augmentation.py
pipenv run python src/main.py
```