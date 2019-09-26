# character-level-cnn_keras
character-level-cnn written by keras.  
character-level-cnnをkerasで実装しました．  

# Usage（使い方）
* **Preprocessing（前処理）**  
text -> convert unicode  
categorical feature -> Labelencoding  
numerical feature -> Scaling [0,1]  
テキスト -> unicodeに変換  
カテゴリ変数 -> ラベルエンコーディング  
数値変数 -> [0,1]にスケーリング  

unicode変換は ord関数を用いてください．

* **Input shape（入力形式）**  
Store text, categorical variables, and numeric variables in separate lists.  
For example inputs = [text, numerical_features, categorical_features]  
テキスト，カテゴリ変数，数値変数をそれぞれ別々のリストに格納してください．
例えば，入力は inputs = [text, numerical_features, categorical_features] となります．  

# Environment（実行環境）
* Keras v2.3.0
* tensorflow v1.14.0

# Reference（参考文献）
[1]. Character-level Convolutional Networks for Text Classification [[link]](https://arxiv.org/abs/1509.01626#)
