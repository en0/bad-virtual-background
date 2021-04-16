import time
from functools import reduce
from collections import deque
import pyfakewebcam
import tensorflow as tf
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths
import numpy as np
import cv2

WIDTH = 640
HEIGHT = 480
bg_color = [0, 0, 0]

model = download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16)
model = load_model(model)

background = cv2.imread("background.png")
background = cv2.resize(background, (WIDTH, HEIGHT))

fakecam = pyfakewebcam.FakeWebcam('/dev/video2', WIDTH, HEIGHT)
realcam = cv2.VideoCapture(0)
realcam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
realcam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
realcam.set(cv2.CAP_PROP_FPS, 30)


def post_process_mask(m):
    m = cv2.dilate(m, np.ones((10,10), np.uint8) , iterations=1)
    m = cv2.blur(m.astype(float), (10,20))
    return m

mask_queue = deque(maxlen=5)
while True:
    _, frame = realcam.read()
    if frame is None:
        break
    mask = model.predict_single(frame).get_mask(threshold=0.75).numpy()
    mask = mask.reshape((frame.shape[0], frame.shape[1])).astype(np.uint8)
    mask_queue.append(mask)
    mask = reduce(lambda a, b: cv2.bitwise_and(a, b), mask_queue)
    mask = post_process_mask(mask)

    inv_mask = 1-mask
    bl_frame = cv2.blur(frame, (50, 50))
    for c in range(frame.shape[2]):
        frame[:,:,c] = frame[:,:,c]*mask + bl_frame[:,:,c]*inv_mask

    frame = cv2.flip(frame, 1)
    #cv2.putText(frame, "hello, world", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fakecam.schedule_frame(frame)


