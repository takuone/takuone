- ð Hi, Iâm @takuone
- ð Iâm interested in ...
- ð± Iâm currently learning ...
- ðï¸ Iâm looking to collaborate on ...
- ð« How to reach me ...

<!---
takuone/takuone is a â¨ special â¨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->import numpy as np
import cv2

from keras.preprocessing import image
import keras.applications.vgg16 as vgg16

# ãImageNetããã¼ã¿ã»ãããå©ç¨ãã¦å­¦ç¿ããã¢ãã«ãåå¾
model = vgg16.VGG16(weights='imagenet', input_shape=(224, 224, 3))

model.summary()
# ç»åãã¡ã¤ã«ã®ãã¹
path = "48113381998_22db4ed163_q"

# æ¢å®ã®ãµã¤ãºï¼ä»åã¯224x224ï¼ã«ãªãµã¤ãºãã¦èª­ã¿è¾¼ã¿
img = image.load_img(path, target_size=(150, 150))

# PILå½¢å¼ - > numpy.ndarray
X = image.img_to_array(img)

# (width, height, ch) -> (batch, rwidth, height, ch)
X = np.expand_dims(X, axis=0)

# åå¦çï¼å¹³åå¤ãå¼ããªã©ï¼
X = vgg16.preprocess_input(X)

# æ¨å®
Y = model.predict(X)

# ä¸ä½æ¨å®çµæãè¡¨ç¤º
list= vgg16.decode_predictions(Y, top=4)[0]
for v in list:
    print(v)
    
