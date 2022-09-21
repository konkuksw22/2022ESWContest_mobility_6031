import cv2
import numpy as np
import keyboard
import os

def perspective_transform(img, pts, title):

    pts1 = np.float32([pts[0], pts[1], pts[3], pts[2]])
    
    w1 = abs(pts[3][0] - pts[2][0])
    w2 = abs(pts[1][0] - pts[0][0])
    h1 = abs(pts[1][1] - pts[3][1])
    h2 = abs(pts[0][1] - pts[2][1])
    width = max([int(w1), int(w2)])  
    height = max([int(h1), int(h2)]) 

    pts2 = np.float32([[0, 0], [width , 0], [width , height], [0, height]])

    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
  
    result = cv2.warpPerspective(img, mtrx, (width, height))
    cv2.imshow(title, result)
    
def getsize(capture1,capture2):
    re_D, img_D = capture1.read()
    re_B, img_B = capture2.read()
    img_D = cv2.resize(img_D, (1280,720))
    img_B = cv2.resize(img_B, (1280,720))
    cv2.imshow("image", img_D)
    for i in range (3):
        while True:
            perspective_transform(img_B, dir[0],"tl")
            perspective_transform(img_B, dir[2],"tr")
            if keyboard.is_pressed("d"):
                if i==0:
                    dir[i][0][1] = dir[i][0][1] + 10
                    dir[i][2][1] = dir[i][2][1] - 10
                elif i==2:
                    dir[i][1][1] = dir[i][1][1] - 10
                    dir[i][3][1] = dir[i][3][1] + 10
            elif keyboard.is_pressed("w"):
                for j in range (0,4):
                    dir[i][j][1]=dir[i][j][1] + 10
            elif keyboard.is_pressed("a"):
                if i==0:
                    dir[i][0][1] = dir[i][0][1] - 10
                    dir[i][2][1] = dir[i][2][1] + 10
                elif i==2:
                    dir[i][1][1] = dir[i][1][1] + 10
                    dir[i][3][1] = dir[i][3][1] - 10
            elif keyboard.is_pressed("x"):
                for j in range (0,4):
                    dir[i][j][1]=dir[i][j][1] - 10
            if cv2.waitKey() == ord('q'):
                break
    
    np.save('dir.npy', dir)
    cv2.destroyAllWindows()
    flag = 1
    return flag

flag=0
if os.path.isfile("./dir.npy")==True:
    dir=np.load('dir.npy')
else :
    right=[[1030,0],[1280,120],[1030,720],[1280,650]]
    left=[[0,120],[250,0],[0,620],[250,720]]
    nom=[[0,0],[0,0],[0,0],[0,0]]
    dir=[left,nom,right]

cap_builtin = cv2.VideoCapture("./builtin_3.mp4")
cap_driver = cv2.VideoCapture("./builtin_3.mp4")
flag = getsize(cap_builtin,cap_driver)

# cap2 = cv2.VideoCapture("220817_blackbox2.mp4")
while cap_builtin.isOpened() and cap_driver.isOpened() and flag == 1 :
    ret, frame = cap_builtin.read()
    ret1, frame1 = cap_driver.read()
    rows, cols = frame.shape[:2]

    # ret2, frame2 = cap2.read()

    cv2.imshow("video", frame1)
    # cv2.imshow("video2", frame2)
    perspective_transform(frame, dir[0],"tl")
    perspective_transform(frame, dir[2],"tr")

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
