import cv2
import socket
import pickle
import struct
import numpy as np

ip = '192.168.1.15'
port = 50001
tag = True
pts = np.zeros((4, 2), dtype=np.float32)
pts[0] = [0, 120]  # topLeft
pts[1] = [250, 0]  # topRight
pts[2] = [0, 620]  # bottomLeft
pts[3] = [250, 720]  # bottomRight

def perspective_transform(img, pts):
    topLeft = pts[0]
    topRight = pts[1]
    bottomLeft = pts[2]
    bottomRight = pts[3]

    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = max([int(w1), int(w2)])  # 두 좌우 거리간의 최대값이 서류의 폭
    height = max([int(h1), int(h2)])  # 두 상하 거리간의 최대값이 서류의 높이

    pts2 = np.float32([[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]])


    mtrx = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(img, mtrx, (width, height))
    cv2.imshow('transformed', result)

def getpos(img):
    img = cv2.resize(img, (1280, 720))

    while True:
        cv2.imshow("image", img)
        perspective_transform(img, pts)

        if cv2.waitKey() == ord('w'):
            pts[0][1] = pts[0][1] + 10
            pts[2][1] = pts[2][1] - 10
            print('w')
            print(pts)
        elif cv2.waitKey() == ord('a'):
            print('a')
        elif cv2.waitKey() == ord('s'):
            pts[0][1] = pts[0][1] - 10
            pts[2][1] = pts[2][1] + 10
            print('s')
            print(pts)
        elif cv2.waitKey() == ord('d'):
            print('d')


        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()  


if __name__=="__main__":

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((ip, port))
        print("Connect Success")
        retval, frame = capture.read()
        getpos(frame)

        while True:
            retval, frame1 = capture.read()
            retval, frame = cv2.imencode('.jpg', frame1, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame = pickle.dumps(frame)

            client_socket.sendall(struct.pack(">L", len(frame))+frame)

            cv2.imshow("View", frame1)
            perspective_transform(frame1, pts)


    capture.release()
    cv2.destroyAllWindows()