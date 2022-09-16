import cv2
import numpy as np

def perspective_transform(img, pts):
    topLeft = pts[0]
    topRight = pts[1]
    bottomLeft = pts[2]
    bottomRight = pts[3]

    # 변환 전 4개 좌표
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
    # print(pts1)

    # 변환 후 영상에 사용할 서류의 폭과 높이 계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = max([int(w1), int(w2)])  # 두 좌우 거리간의 최대값이 서류의 폭
    height = max([int(h1), int(h2)])  # 두 상하 거리간의 최대값이 서류의 높이
    # 변환 후 4개 좌표
    pts2 = np.float32([[0, 0], [width , 0],
                       [width , height ], [0, height ]])

    # 변환 행렬 계산
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    # print(mtrx)
    # 원근 변환 적용
    result = cv2.warpPerspective(img, mtrx, (width, height))
    cv2.imshow('transformed', result)

pts = np.zeros((4, 2), dtype=np.float32)
pts[0] = [1030, 0]  # topLeft
pts[1] = [1280, 120]  # topRight
pts[2] = [1030, 720]  # bottomLeft
pts[3] = [1280, 650]  # bottomRight
flag = 0

def getsize(capture1, capture2):
    #img = cv2.imread('bbox_2.jpg')
    re, img = capture1.read()
    re1,img1=capture2.read()

    img = cv2.resize(img, (1280, 720))
    img1= cv2.resize(img1, (1280, 720))
    
    while True:
        cv2.imshow("image", img)
        perspective_transform(img1, pts)

        if cv2.waitKey() == ord('w'):
            pts[0][1] = pts[0][1] + 20
            pts[2][1] = pts[2][1] - 20
            print('w')
            print(pts)
        elif cv2.waitKey() == ord('a'):
            pts[0][1] = pts[0][1] + 20
            pts[2][1] = pts[2][1] + 20
            pts[1][1] = pts[1][1] + 20
            pts[3][1] = pts[3][1] + 20
            print('a')
        elif cv2.waitKey() == ord('s'):
            pts[0][1] = pts[0][1] - 20
            pts[2][1] = pts[2][1] + 20
            print('s')
            print(pts)
        elif cv2.waitKey() == ord('d'):
            pts[0][1] = pts[0][1] + 20
            pts[2][1] = pts[2][1] - 20
            print('d')
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()
    flag = 1
    return flag

# Video resolution : 1280 * 720
cap_builtin = cv2.VideoCapture("./builtin.mp4")
cap_driver = cv2.VideoCapture("./driver.mp4")
flag = getsize(cap_driver,cap_builtin)

# cap2 = cv2.VideoCapture("220817_blackbox2.mp4")
while cap_driver.isOpened() and flag == 1 :
    ret, frame = cap_driver.read()
    ret1,frame1= cap_builtin.read()
    rows, cols = frame.shape[:2]

    # ret2, frame2 = cap2.read()

    cv2.imshow("video", frame)
    # cv2.imshow("video2", frame2)
    perspective_transform(frame1, pts)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if cv2.waitKey(1) & 0xFF == ord('w'):
    #     print('w')

    # keycode = cv2.waitKey(1)
    # if keycode == ord('w'):
    #     print('w')

cap_builtin.release()
cap_driver.release()
cv2.destroyAllWindows()
