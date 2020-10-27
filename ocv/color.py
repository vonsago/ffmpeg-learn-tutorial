import cv2
import numpy as np
import matplotlib.pyplot as plt


def video_compare(video1, video2):
    capture = cv2.VideoCapture(video1)
    capture2 = cv2.VideoCapture(video2)
    counter = 0
    while True:
        f, frame = capture.read()
        f2, frame2 = capture2.read()
        try:
            res = frame - frame2
            if (np.count_nonzero(res) > 0):
                print('[-', frame, '--', frame2, '-]')
                counter += 1
        except Exception as e:
            print(e)


def process_image(file):
    test_png = cv2.imread(file)
    test_png = cv2.cvtColor(test_png, cv2.COLOR_BGR2RGB)
    plt.imshow(test_png)
    plt.show()


if __name__ == "__main__":
    video_compare("/Users/bilibili/Desktop/test1.ts", "/Users/bilibili/Desktop/test.ts")