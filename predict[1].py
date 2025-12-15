import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("model/forgery_model.h5")

img = cv2.imread("test.jpg")
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]

if pred > 0.5:
    print("Forged Image")
else:
    print("Authentic Image")
