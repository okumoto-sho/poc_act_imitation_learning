import cv2 as cv
import time

from multiprocessing import Process, Queue, Pipe
from koch11.camera import Camera
from teleoperation_config import camera_config_1, camera_config_2


camera1 = Camera(**camera_config_1)
camera2 = Camera(**camera_config_2)

out1 = cv.VideoWriter("output1.mp4", cv.VideoWriter_fourcc(*"mp4v"), 120, (640, 480))
out2 = cv.VideoWriter("output2.mp4", cv.VideoWriter_fourcc(*"mp4v"), 120, (640, 480))

while True:
    start = time.time()

    img1 = camera1.read()
    img2 = camera2.read()

    cv.imshow("camera1", img1)
    cv.imshow("camera2", img2)
    out1.write(img1)
    out2.write(img2)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    print(1 / (time.time() - start))

out1.release()
out2.release()
