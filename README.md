# learnable_encryption_robustness

learnable encryption手法の，元画像の復元を目的とした攻撃への耐性の評価

## 暗号化画像の生成
`learnable_encryption_robustness/notebook/` 内の `generate_hoge_dataset.ipynb` を参照．

## 暗号化手法
4種類の暗号化．
各暗号化手法の略称はmaungさんの論文より．
実装は， `learnable_encryption_robustness/learnable_encryption_robustness/encryption/` 参照．
* LE
* ELE
* PE
* EtC

## 攻撃手法
3種類の攻撃手法．
各攻撃手法の略称はmaungさんの論文より．
* FR-Attack
  * 実装は `learnable_encryption_robustness/decrypt_diffkey_basic.m` 参照．
  * オリジナルのリポジトリは[ここ](https://github.com/ahchang98/image-encryption-scheme-attacks)．ファイル読み込み部分を少し修正．
* GAN-Attack
* ITN-Attack
