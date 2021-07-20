- 👋 Hi, I’m @takuone
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
takuone/takuone is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->import numpy as np
import cv2

from keras.preprocessing import image
import keras.applications.vgg16 as vgg16

# 「ImageNet」データセットを利用して学習したモデルを取得
model = vgg16.VGG16(weights='imagenet', input_shape=(224, 224, 3))

model.summary()
# 画像ファイルのパス
path = "48113381998_22db4ed163_q"

# 既定のサイズ（今回は224x224）にリサイズして読み込み
img = image.load_img(path, target_size=(150, 150))

# PIL形式 - > numpy.ndarray
X = image.img_to_array(img)

# (width, height, ch) -> (batch, rwidth, height, ch)
X = np.expand_dims(X, axis=0)

# 前処理（平均値を引くなど）
X = vgg16.preprocess_input(X)

# 推定
Y = model.predict(X)

# 上位推定結果を表示
list= vgg16.decode_predictions(Y, top=4)[0]
for v in list:
    print(v)
    
