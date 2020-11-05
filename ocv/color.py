import cv2
import numpy as np
import matplotlib.pyplot as plt


def video_compare(video1, video2):
    capture = cv2.VideoCapture(video1)
    capture2 = cv2.VideoCapture(video2)
    counter = [0, 0, 0]
    while True:
        f, frame = capture.read()
        f2, frame2 = capture2.read()
        try:
            # res = frame - frame2
            # # if (np.count_nonzero(res) > 0):
            # #     # print('[-', frame, '--', frame2, '-]')
            # #     counter += 1
            h = len(frame)
            w = len(frame[0])
            if (len(frame2)!= len(frame) or len(frame2[0])!=len(frame[0])):
                print("width*height error")
                return
            for he in range(len(frame)):
                for wi in range(len(frame[0])):
                    diff = frame[he][wi] - frame2[he][wi]
                    if(np.count_nonzero(diff)>0):
                        su_di = sum(diff)
                        if(su_di < 5):
                            counter[0]+=1
                        elif(su_di>=5 and su_di<=10):
                            counter[1]+=1
                        else:
                            counter[2]+=1
                        #print(frame[he][wi], frame2[he][wi], end=",")
                    #print()
        except Exception as e:
            print(e)
        print([i/(w*h) for i in counter])
        return capture


def process_image(file):
    test_png = cv2.imread(file)
    test_png = cv2.cvtColor(test_png, cv2.COLOR_BGR2RGB)
    plt.imshow(test_png)
    plt.show()


if __name__ == "__main__":
    video_compare("test1.ts", "test2.ts")
