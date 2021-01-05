import cv2
import numpy as np
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from collections import deque
import os
"""
__________part-4__________
drawing recognition
把手指塗鴉的圖片讀入
辨識塗鴉的種類
@author: 林子云
"""
class QuickDraw:
    def __init__(self):
        self.model = tf.keras.models.load_model('QuickDraw_v2.h5')
        self.classes = ["apple", "book", "bowtie", "cloud", "cup", "door", "envelope", "eyeglasses", "hat", "ice cream",
                        "lightning", "pants", "scissors", "star", "t-shirt"]

    def keras_process_image(self, img):
        image_x = 28
        image_y = 28
        img = cv2.resize(img, (image_x, image_y))
        img = np.array(img, dtype=np.float32)
        img = np.reshape(img, (-1, image_x, image_y, 1))
        return img

    def keras_predict(self, model, image):
        processed = self.keras_process_image(image)
        print("processed: " + str(processed.shape))
        pred_probab = model.predict(processed)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        return max(pred_probab), pred_class

    def draw_detect(self, image):
        pred_probab, pred_class = self.keras_predict(self.model, image)
        print(self.classes[pred_class], pred_probab)