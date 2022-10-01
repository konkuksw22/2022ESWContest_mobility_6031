import numpy as np
import cv2
import glob
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./cali_com/*.png')
cnt=0
for fname in images:
    cnt+=1
    print(cnt)
    img = cv2.imread(fname)
    #img=cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print(fname)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(100)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
img = cv2.imread('./test_img/test1.png')
#img=cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_LINEAR)
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1)
# mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
# dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png',dst)
# print(mtx)
print(mtx)
print(dist)

dst = cv2.undistort(img, mtx, dist, None, newcameramtx) ## getOptimalNewCameraMatrix 함수를 쓰지 않은 이미지
dst2 = cv2.undistort(img, mtx, dist, newcameramtx) ## 함수를 쓴 이미지
cv2.imshow('num1', dst)
cv2.imshow('num2', dst2)
cv2.waitKey(0)

tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error
print("total error: ", tot_error/len(objpoints))

cv2.destroyAllWindows()