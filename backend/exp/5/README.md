# 1. データ増強の検証
各データ増強を3回おこなった上での特徴量ごとの平均

### feature_dif_mean_gain.csv
- 音量[0.9, 1.1]倍
### feature_dif_mean_inverse.csv
- 位相を反転
### feature_dif_mean_noise.csv
- [0, 0.02]の振幅をもつホワイトノイズを付加
### feature_dif_mean_pitch.csv
- ピッチを[-20, 20]セント変化
### feature_dif_mean_time.csv
- 任意の位置に再生開始位置を変更

# 2. 特徴量の分析
各特徴量の分散と相関係数

### feature_cor.png
- 特徴量間の相関係数の絶対値のヒートマップ
### feature_var.csv
- データセット内の各特徴量の分散

# 3. 正規化の検証
正規化による影響

### norm_graph_45-10-1_0.png, norm_graph_45-10-1_1.png, norm_graph_45-10-1_2.png
- 正規化で学習した45-10-1モデルの損失推移
### norm_model_45-10-1_0.ptn, norm_model_45-10-1_1.ptn, norm_model_45-10-1_2.ptn
- 正規化で学習した45-10-1モデル
### norm_result_45-10-1_0.csv, norm_result_45-10-1_1.csv, norm_result_45-10-1_2.csv
- 正規化で学習した45-10-1モデルのテスト結果
### norm_graph_45-20-10-1_0.png, norm_graph_45-20-10-1_1.png, norm_graph_45-20-10-1_2.png
- 正規化で学習した45-20-10-1モデルの損失推移
### norm_model_45-20-10-1_0.ptn, norm_model_45-20-10-1_1.ptn, norm_model_45-20-10-1_2.ptn
- 正規化で学習した45-20-10-1モデル
### norm_result_45-20-10-1_0.csv, norm_result_45-20-10-1_1.csv, norm_result_45-20-10-1_2.csv
- 正規化で学習した45-20-10-1モデルのテスト結果