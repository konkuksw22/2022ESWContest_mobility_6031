from __future__ import division

from torch.autograd import Variable
from torch.cuda import FloatTensor
import torch.nn as nn

from darknet import Darknet, set_requires_grad
from shapely.geometry import Polygon, Point
from moviepy.editor import VideoFileClip
#from scipy.misc import imresize
from PIL import Image

import numpy as np
import os
import io
import cv2
import pafy
import argparse
import time
import serial
import math
from math import *
from util import *
from collections import deque

# Color
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
yellow = (0, 255, 255)
deepgray = (43, 43, 43)
dark = (1, 1, 1)
cyan = (255, 255, 0)
magenta = (255, 0, 255)
lime = (0, 255, 128)
purple = (255, 0, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN

# Global 함수 초기화
l_pos, r_pos, l_cent, r_cent = 0, 0, 0, 0 #왼쪽 점, 오른쪽 점, ? , ? 
uxhalf, uyhalf, dxhalf, dyhalf = 0, 0, 0, 0 #?
l_center, r_center, lane_center = ((0, 0)), ((0, 0)), ((0, 0)) #왼쪽 중앙, 오른쪽 중앙, 중앙
next_frame = (0, 0, 0, 0, 0, 0, 0, 0) #다음 프레임 저장

# 구문 저장
def arg_parse():
    parses = argparse.ArgumentParser(description='My capstone Design 2019')
    parses.add_argument("--roi", dest = 'roi', default = 0, help = "roi flag")
    parses.add_argument("--alpha", dest = 'alpha', default = 0, help = "center position add alpha")
    parses.add_argument("--video", dest = 'video', default = "./test_videos/drive_00.mp4")
    parses.add_argument("--url", dest = 'url', default = False, type = str, help="youtube url link")
    parses.add_argument("--com", dest = 'com', default = False, help = "Setting Arduino port", type = str)
    parses.add_argument("--brate", dest = 'brate', default = 9600, help = "Setting Arduino baudrate")
    return parses.parse_args()

"""
python main.py --com COM4 --video drive.mp4
python main.py --com COM4 --roi 1 --video drive06.mp4 --alpha 60
python main.py --com COM4 --url https://youtu.be/YsPdvvixYfo --roi 1 --alpha 60
"""

args = arg_parse()

# 동영상 저장
def save_video(filename, frame=30.0):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(filename, fourcc, frame, (1280,720))
    return out

# grayscale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# canny
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# gaussian_blur
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# ROI
def region_of_interest(img, vertices):
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255, ) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        # vertiecs로 만든 polygon으로 이미지의 ROI를 정하고 ROI 이외의 영역은 모두 검정색으로 정한다.

        masked_image = cv2.bitwise_and(img, mask)
        cv2.imshow("ROI", masked_image)
        return masked_image

# 기울기 구하기
def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

# x 절편 구하기
def get_x_point(slope, pointx, pointy):
    if (pointx+pointy/slope)>=0 and (pointx+pointy/slope)<1280:
        return int(pointx+pointy/slope)
    else:
        return 0

# 직선 그리기
def draw_lines(img, lines):
    global cache
    global first_frame
    global next_frame

    """초기화"""
    y_global_min = img.shape[0]
    y_max = img.shape[0]

    l_slope, r_slope = [], []
    l_lane, r_lane = [], []

    det_slope = 0.5
    α = 0.2

    height, width = img.shape[:2]

    """선을 인식하면 기울기 정도에 따라 오른쪽 차선인지, 왼쪽 차선인지 구별"""
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = get_slope(x1,y1,x2,y2)
                x_point=get_x_point(slope, x1, y1)
                #x 절편 따지기
                print(abs(math.atan(slope)))
                if abs(math.atan(slope))>80:
                    if x_point>360:
                        r_slope.append(slope) #기울기
                        r_lane.append(line) # 점들
                        print("right", slope)
                    elif x_point<360:
                        l_slope.append(slope)
                        l_lane.append(line)
                        print("left", slope)

        y_global_min = min(y1, y2, y_global_min) # 최소 y좌표 저장

    if (len(l_lane) == 0 or len(r_lane) == 0): # 오류 방지
        return 1

    """기울기와, 점 수치들 평균 값 구하기"""
    l_slope_mean = np.mean(l_slope, axis =0)
    r_slope_mean = np.mean(r_slope, axis =0)
    l_mean = np.mean(np.array(l_lane), axis=0)
    r_mean = np.mean(np.array(r_lane), axis=0)

    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1

    # y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    #이건 없어도 되는거 아닌가
    if np.isnan((y_global_min - l_b)/l_slope_mean) or \
    np.isnan((y_max - l_b)/l_slope_mean) or \
    np.isnan((y_global_min - r_b)/r_slope_mean) or \
    np.isnan((y_max - r_b)/r_slope_mean):
        return 1

    # x구하기
    l_x1 = int((y_global_min - l_b)/l_slope_mean)
    l_x2 = int((y_max - l_b)/l_slope_mean)
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)

    if l_x1 > r_x1: # Left line이 Right Line보다 오른쪽에 있는 경우 (Error)
        l_x1 = ((l_x1 + r_x1)/2)
        r_x1 = l_x1

        l_y1 = ((l_slope_mean * l_x1 ) + l_b)
        r_y1 = ((r_slope_mean * r_x1 ) + r_b)
        l_y2 = ((l_slope_mean * l_x2 ) + l_b)
        r_y2 = ((r_slope_mean * r_x2 ) + r_b)

    else: # l_x1 < r_x1 (Normal)
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max

    current_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype ="float32")

    if first_frame == 1:
        next_frame = current_frame
        first_frame = 0
    else:
        prev_frame = cache
        next_frame = (1-α)*prev_frame+α*current_frame

    global l_center
    global r_center
    global lane_center

    """next_frame = np.array([  0 ,   1 ,  2  ,  3  ,  4  ,  5  ,  6  ,  7  ], dtype ="float32")"""
    """next_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype ="float32")"""
    div = 2
    l_center = (int((next_frame[0] + next_frame[2]) / div), int((next_frame[1] + next_frame[3]) / div))
    r_center = (int((next_frame[4] + next_frame[6]) / div), int((next_frame[5] + next_frame[7]) / div))
    lane_center = (int((l_center[0] + r_center[0]) / div), int((l_center[1] + r_center[1]) / div))

    global uxhalf, uyhalf, dxhalf, dyhalf
    uxhalf = int((next_frame[2]+next_frame[6])/2)
    uyhalf = int((next_frame[3]+next_frame[7])/2)
    dxhalf = int((next_frame[0]+next_frame[4])/2)
    dyhalf = int((next_frame[1]+next_frame[5])/2)

    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), red, 2)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), red, 2)

    cache = next_frame

# 가로를 세로로 변경
def lane_pts():
    pts = np.array([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], [next_frame[6], next_frame[7]], [next_frame[4], next_frame[5]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    return pts

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, 1, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    dst=np.zeros_like(img)

    if lines is not None:
        for i in lines:
            cv2.line(dst, (int(i[0][0]), int(i[0][1])), (int(i[0][2]), int(i[0][3])), (255,255,255), 2)
        cv2.imshow("line", dst)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# 두개의 이미지를 더함
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    # result = imgA * a + imgB * b + c
    return cv2.addWeighted(initial_img, α, img, β, λ)

def get_pts(flag=0):
    # vertices1 = np.array([
    #             [230, 650],
    #             [620, 460],
    #             [670, 460],
    #             [1050, 650]
    #             ])

    #내가만든 ROI
    vertices1 = np.array([
                [260, 485],
                [550, 345],
                [645, 345],
                [900, 485],
    ])

    #그냥 단순 직사각형
    # vertices1 = np.array([
    #             [0, 455],
    #             [0, 340],
    #             [1280, 340],
    #             [1280, 455],
    # ])

    # vertices2 = np.array([
    #             [0, 720],
    #             [710, 400],
    #             [870, 400],
    #             [1280, 720]
    # ])

    vertices2 = np.array([
                [430, 755],
                [645, 580],
                [980, 580],
                [1410, 755]
    ])

    if flag == 0 : return vertices1
    if flag == 1 : return vertices2

hei = 25
alpha = int(args.alpha)
font_size = 1
""" 핸들 조종 및 위험 메세지 표시 """
def warning_text(image):
    whalf, height = 640, 720
    center = whalf - 5 + alpha
    angle = int(round(atan((dxhalf-(center))/120) * 180/np.pi, 3) * 3)

    m = 2
    limit = 0
    value = 0
    if angle > 90 : angle = 89
    if 90 > angle > limit :
        cv2.putText(image, 'WARNING : ', (10, hei*m), font, 0.8, red, font_size)
        cv2.putText(image, 'Turn Right', (150, hei*m), font, 0.8, red, font_size)
        value = angle

    if angle < -90 : angle = -89
    if -90 < angle < -limit:
        cv2.putText(image, 'WARNING : ', (10, hei*m), font, 0.8, red, font_size)
        cv2.putText(image, 'Turn Left', (150, hei*m), font, 0.8, red, font_size)
        value = -angle + 100

    elif angle == 0 :
        cv2.putText(image, 'WARNING : ', (10, hei*m), font, 0.8, white, font_size)
        cv2.putText(image, 'None', (150, hei*m), font, 0.8, white, font_size)
        value = 0

    # cv2.putText(image, 'angle = {0}'.format(angle), (10, hei*4), font, 0.7, white, font_size)
    # cv2.putText(image, 'value = {0}'.format(value), (10, hei*5), font, 0.7, white, font_size)
    # print(value)
    if mcu_port:
        mcu.write([value])

""" 현재 영상 프레임 표시 """
def show_fps(image, frames, start, color = white):
    now_fps = round(frames / (time.time() - start), 2)
    cv2.putText(image, "FPS : %.2f"%now_fps, (10, hei), font, 0.8, color, font_size)

""" Steering Wheel Control 시각화 """
def direction_line(image, height, whalf, color = yellow):
    cv2.line(image, (whalf-5+alpha, height), (whalf-5+alpha, 600), white, 2) # 방향 제어 기준선
    cv2.line(image, (whalf-5+alpha, height), (dxhalf, 600), red, 2) # 핸들 방향 제어
    cv2.circle(image, (whalf-5+alpha, height), 120, white, 2)

""" 왼쪽 차선, 오른쪽 차선, 그리고 차선의 중앙 지점 표시 """
def lane_position(image, gap = 20, length=20, thickness=2, color = red, bcolor = white): # length는 선의 위쪽 방향으로의 길이
    global l_cent, r_cent

    l_left = 300
    l_right = 520
    l_cent = int((l_left+l_right)/2)
    cv2.line(image, (l_center[0], l_center[1]+length), (l_center[0], l_center[1]-length), color, thickness)

    r_left = 730
    r_right = 950
    r_cent = int((r_left+r_right)/2)
    cv2.line(image, (r_center[0], r_center[1]+length), (r_center[0], r_center[1]-length), color, thickness)

""" 왼쪽 차선과 오른쪽 차선을 직선으로 표시 """
def draw_lanes(image, thickness = 3, color = red):
    cv2.line(image, (next_frame[0], next_frame[1]), (next_frame[2], next_frame[3]), red, 3)
    cv2.line(image, (next_frame[6], next_frame[7]), (next_frame[4], next_frame[5]), red, 3)

""" Arduino Map 함수 """
def map(x, in_min, in_max, out_min, out_max):
    return ((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

""" 두 차선이 Cross하는 지점을 계산 """
def lane_cross_point():
    """
    y = m(x-a) + b (m은 negative)
    y = n(x-e) + f (n은 positive)
    -> x = (am - b - en + f)/(m-n)
    """
    for seq in range(8):
        if next_frame[seq] == 0: # next_frame 중 하나라도 0이 존재하면 break
            return (0, 0)
        else:
            l_slope = get_slope(next_frame[0], next_frame[1], next_frame[2], next_frame[3])
            r_slope = get_slope(next_frame[6], next_frame[7], next_frame[4], next_frame[5])

            x = (next_frame[0]*l_slope - next_frame[1] - next_frame[6]*r_slope + next_frame[7])/(l_slope-r_slope)
            y = l_slope*(x-next_frame[0]) + next_frame[1]
            return int(x), int(y)

""" 시점변경 """
def perspective(image): # Bird's eye view
    pts1 = np.float32([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], [next_frame[4], next_frame[5]], [next_frame[6], next_frame[7]]])
    pts2 = np.float32([[425, 0], [425, 720], [855, 0], [855, 720]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (1280, 720))
    cv2.line(dst, (l_cent, 0), (l_cent, 720), red, 2)
    cv2.line(dst, (r_cent, 0), (r_cent, 720), red, 2)
    return dst























""" 차선 검출을 위한 이미지 전처리 """
def process_image(image):
    global first_frame

    #print(image.shape)
    #image = cv2.resize(image, (720, 1280, 3))
    height, width = image.shape[:2]

    kernel_size = 3

    # Canny Edge Detection Threshold
    low_thresh = 300
    high_thresh = 500

    rho = 2
    theta = np.pi/180
    #thresh = 100
    #min_line_len = 50
    #max_line_gap = 150
    min_line_len = 20
    max_line_gap = 150
    thresh = 100

    # 직선긋는것 까지 성공!
    # 젤 오른차선의 기울기 따지기, 그리고 수평 찾기 수평 사이 거리가 50픽셀 이런식으로 가까워야함. 그래야 신호등 혹은 정지선.

    gray_image = grayscale(image)

    # 시점 변환 - 버드 아이 뷰
    warped, M, Minv = pers_transform(image)

    # 색깔 추출
    mask_white = hls_thresh(warped) #하얀색 추출

    color_mask=np.zeros_like(mask_white, dtype="uint8") #노란색 추출
    img_hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV) # 더 넓은 폭의 노란색 범위를 얻기위해 HSV를 이용한다.
    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype = "uint8")
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    color_mask[(mask_white==1)|(mask_yellow>=1)]=255

    canny_edges = canny(color_mask, low_thresh, high_thresh)
    cv2.imshow("canny", canny_edges)

    line_image = hough_lines(canny_edges, rho, theta, thresh, min_line_len, max_line_gap)
    cv2.imshow("test", line_image)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    # cv2.polylines(result, vertices, True, (0, 255, 255)) # ROI mask

    #mask_yw_image = cv2.bitwise_and(grayscale(warped), color_mask.astype(np.uint8)) # Grayscale로 변환한 원본 이미지에서 흰색과 노란색만 추출
    #cv2.imshow("good", mask_yw_image)

    # gauss_gray = gaussian_blur(mask_yw_image, kernel_size)
    # sobelx = cv2.Sobel(color_mask, cv2.CV_64F, 1, 0, ksize=7) # Take the derivative in x
    # abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    # scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # grad_bin = np.zeros_like(scaled_sobel)
    # grad_bin[(scaled_sobel >= 100) & (scaled_sobel <= 200)] = 255 #이거 왜 binary가 안되는거지? (1)로 값이 안먹힘
    # cv2.imshow('x sobel', grad_bin)

    output_img=find_lanes(color_mask)
    cv2.imshow("pi", output_img)


    # vertices = [get_pts(flag = 0)]
    # roi_image = region_of_interest(canny_edges, vertices)


    return result, line_image

""" 차선 검출 결과물을 보여줌 """
def visualize(image, flg):
    height, width = image.shape[:2]
    whalf = int(width/2)
    hhalf = int(height/2)

    zeros = np.zeros_like(image)
    vertices = [get_pts(flag=flg)]
    pts = lane_pts()

    gap = 25
    max = 100 # 410 ~ 760
    limit = 30
    if not lane_center[1] < hhalf:
        """ 차선검출 영역 최대 길이 이상 지정 """
        if r_center[0]-l_center[0] > max:
            cv2.fillPoly(zeros, [pts], lime)
            lane_position(zeros)
            direction_line(zeros, height = height, whalf = whalf)

    """ Lane Detection ROI """
    # cv2.putText(zeros, 'ROI', (930, 650), font, 0.8, yellow, font_size)
    # cv2.polylines(zeros, vertices, True, (0, 255, 255))
    return zeros

# object detection start and end point
""" 객체 검출"""
def write(x, results, color = [126, 232, 229], font_color = red): # x = output
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    c1=(int(c1[0]), int(c1[1]))
    c2=(int(c2[0]), int(c2[1]))
    #print(x[1:3].item(), '\n')
    #정수형으로 변환해주기
    cls = int(x[-1]) # 마지막 Index

    image = results
    label = "{0}".format(classes[cls])

    vals = [0, 2, 3, 5, 7] #not 2 or 3 or 5 or 7: # 인식할 Vehicles를 지정 (2car, 7truck, 5bus, 3motorbike)
    for val in vals:
        if cls == val:
            if not abs(c1[0]-c2[0]) > 1000: # 과도한 Boxing 제외
                centx = int((c1[0]+c2[0])/2)
                centy = int((c1[1]+c2[1])/2)

                if cls == 0:
                    cv2.rectangle(image, c1, c2, blue, 1)

                    t_size = cv2. getTextSize(label, font2, 1, 1)[0]
                    c2 = c1[0] + t_size[0] + 3, c1[1] - t_size[1] - 4

                    cv2.rectangle(image, c1, (c2[0]-3, c2[1]+4), blue, -1)
                    cv2.putText(image, label, (c1[0], c1[1] - t_size[1] + 10), font2, 1, white, 1)

                else:
                    cv2.rectangle(image, c1, c2, red, 1) # 자동차 감지한 사각형

                    t_size = cv2.getTextSize(label, font2, 1, 1)[0]
                    c2 = c1[0] + t_size[0] + 3, c1[1] - t_size[1] - 4

                    cv2.rectangle(image, c1, (c2[0]-3, c2[1]+4), red, -1)
                    cv2.putText(image, label, (c1[0], c1[1] - t_size[1] + 10), font2, 1, white, 1)

    return image

def laneregion(image):
    zeros = np.zeros_like(image) # Object frame zero copy
    for seq in range(8):
        if next_frame[seq] == 0: # next_frame 중 하나라도 0이 존재하면 break
            pass
        else:
            cv2.fillPoly(zeros, [np.array([[next_frame[2], 720], [crossx, crossy], [next_frame[6], 720]], np.int32).reshape((-1, 1, 2))], lime)  # Cneter Region
            # cv2.line(zeros, (crossx, crossy), (crossx, 0), lime, 2)
            cv2.fillPoly(zeros, [np.array([[0, 720], [next_frame[2], 720], [crossx, crossy], [crossx, 0], [0, 0]], np.int32).reshape((-1, 1, 2))], red) # Left Region
            cv2.fillPoly(zeros, [np.array([[1280, 720], [next_frame[6], 720], [crossx, crossy], [crossx, 0], [1280, 0]], np.int32).reshape((-1, 1, 2))], blue) # Right Region
            # cv2.circle(zeros, (crossx, crossy), radius=6, color=red, thickness=-1)

    return zeros


"""--------------------------------------------------------따왔음--------------------------------------------------------------"""
# 차선 정보를 담기 위한 클래스 생성
class Line():
    def __init__(self, maxSamples=4):
        
        self.maxSamples = maxSamples 
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=self.maxSamples)
        # Polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # Polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        # Average x values of the fitted line over the last n iterations
        self.bestx = None
        # Was the line detected in the last iteration?
        self.detected = False 
        # Radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # Distance in meters of vehicle center from the line
        self.line_base_pos = None 
         
    def update_lane(self, ally, allx):
        # Updates lanes on every new frame
        # Mean x value 
        self.bestx = np.mean(allx, axis=0)
        # Fit 2nd order polynomial
        new_fit = np.polyfit(ally, allx, 2)
        # Update current fit
        self.current_fit = new_fit
        # Add the new fit to the queue
        self.recent_xfitted.append(self.current_fit)
        # Use the queue mean as the best fit
        self.best_fit = np.mean(self.recent_xfitted, axis=0)
        # meters per pixel in y dimension
        ym_per_pix = 30/720
        # meters per pixel in x dimension
        xm_per_pix = 3.7/700
        # Calculate radius of curvature
        fit_cr = np.polyfit(ally*ym_per_pix, allx*xm_per_pix, 2)
        y_eval = np.max(ally)
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])


# 시작점의 소스들을 목적지의 점으로 변환시킬때 모두 변환하기
def pers_transform(img, nx=9, ny=6):
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    #src = np.float32([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], [next_frame[4], next_frame[5]], [next_frame[6], next_frame[7]]])
    src = np.float32([[550, 345], [260, 485], [645, 345], [900, 485]])
    dst = np.float32([[425, 0], [425, 720], [855, 0], [855, 720]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst) 
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size) 
    # Return the resulting image and matrix
    Minv = cv2.getPerspectiveTransform(dst, src) 

    return warped, M, Minv


def hls_thresh(img, thresh_min=200, thresh_max=255):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float) 
    s_channel = hls[:,:,1]
    
    # Creating image masked in S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
    return s_binary

def canny_thresh(img, threshold_low, threshold_high):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gauss_gray = cv2.GaussianBlur(gray, (3, 3), 0)
    canny_edges = cv2.Canny(gauss_gray, threshold_low, threshold_high)
    return canny_edges

def dir_thresh(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.uint8) 
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    cv2.imshow("absgraddir", absgraddir)
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh_min) & (absgraddir <= thresh_max)] = 1

    # Return the binary image
    return binary_output

# 노란색 추출 조금 이상한데? 다른 방법 없나 찾아봐야 할 듯
def lab_b_channel(img, thresh=(190,255)):
    # Normalises and thresholds to the B channel
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.uint8) 
    lab_b = lab[:,:,2]+128
    # Don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    #  Apply a threshold
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    return binary_output

def sobel_thresh(img, sobel_kernel=3, orient='x', thresh_min=20, thresh_max=100):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float) 
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    else:
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in x
        abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))

    
    # Creathing img masked in x gradient
    grad_bin = np.zeros_like(scaled_sobel)
    grad_bin[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255 #이거 왜 binary가 안되는거지? (1)로 값이 안먹힘
    cv2.imshow("grad", grad_bin)

    
    return grad_bin

#ROI 제외하고 나머지 다 검정색으로 색칠
def mask_image(image):
    masked_image = np.copy(image)
    mask = np.zeros_like(masked_image)
    vertices = get_pts(flag=0)
    cv2.fillPoly(mask, np.int32([vertices]), (255,255,255))
    masked_edges = cv2.bitwise_and(masked_image, mask)
    return masked_edges
"""sobel 다시 해보기.... 왜 안돼 그리고 dir_bin 하기"""


def validate_lane_update(img, left_lane_inds, right_lane_inds):
    # Checks if detected lanes are good enough before updating
    img_size = (img.shape[1], img.shape[0])
    
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Extract left and right line pixel positions
    left_line_allx = nonzerox[left_lane_inds]
    left_line_ally = nonzeroy[left_lane_inds] 
    right_line_allx = nonzerox[right_lane_inds]
    right_line_ally = nonzeroy[right_lane_inds]
    
    # Discard lane detections that have very little points, 
    # as they tend to have unstable results in most cases
    if len(left_line_allx) <= 1800 or len(right_line_allx) <= 1800:
        left_line.detected = False
        right_line.detected = False
        return
    
    left_x_mean = np.mean(left_line_allx, axis=0)
    right_x_mean = np.mean(right_line_allx, axis=0)
    lane_width = np.subtract(right_x_mean, left_x_mean)
    
    # Discard the detections if lanes are not in their repective half of their screens
    if left_x_mean > 740 or right_x_mean < 740:
        left_line.detected = False
        right_line.detected = False
        return
    
    # Discard the detections if the lane width is too large or too small
    if  lane_width < 300 or lane_width > 800:
        left_line.detected = False
        right_line.detected = False
        return 
    
    # If this is the first detection or 
    # the detection is within the margin of the averaged n last lines 
    if left_line.bestx is None or np.abs(np.subtract(left_line.bestx, np.mean(left_line_allx, axis=0))) < 100:
        left_line.update_lane(left_line_ally, left_line_allx)
        left_line.detected = True
    else:
        left_line.detected = False
    if right_line.bestx is None or np.abs(np.subtract(right_line.bestx, np.mean(right_line_allx, axis=0))) < 100:
        right_line.update_lane(right_line_ally, right_line_allx)
        right_line.detected = True
    else:
        right_line.detected = False    
 
    # Calculate vehicle-lane offset
    xm_per_pix = 3.7/610 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    car_position = img_size[0]/2
    l_fit = left_line.current_fit
    r_fit = right_line.current_fit
    left_lane_base_pos = l_fit[0]*img_size[1]**2 + l_fit[1]*img_size[1] + l_fit[2]
    right_lane_base_pos = r_fit[0]*img_size[1]**2 + r_fit[1]*img_size[1] + r_fit[2]
    lane_center_position = (left_lane_base_pos + right_lane_base_pos) /2
    left_line.line_base_pos = (car_position - lane_center_position) * xm_per_pix +0.2
    right_line.line_base_pos = left_line.line_base_pos


def validate_lane_update(img, left_lane_inds, right_lane_inds):
    # Checks if detected lanes are good enough before updating
    img_size = (img.shape[1], img.shape[0])
    
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Extract left and right line pixel positions
    left_line_allx = nonzerox[left_lane_inds]
    left_line_ally = nonzeroy[left_lane_inds] 
    right_line_allx = nonzerox[right_lane_inds]
    right_line_ally = nonzeroy[right_lane_inds]
    
    # Discard lane detections that have very little points, 
    # as they tend to have unstable results in most cases
    if len(left_line_allx) <= 1800 or len(right_line_allx) <= 1800:
        left_line.detected = False
        right_line.detected = False
        return
    
    left_x_mean = np.mean(left_line_allx, axis=0)
    right_x_mean = np.mean(right_line_allx, axis=0)
    lane_width = np.subtract(right_x_mean, left_x_mean)
    
    # Discard the detections if lanes are not in their repective half of their screens
    if left_x_mean > 740 or right_x_mean < 740:
        left_line.detected = False
        right_line.detected = False
        return
    
    # Discard the detections if the lane width is too large or too small
    if  lane_width < 300 or lane_width > 800:
        left_line.detected = False
        right_line.detected = False
        return 
    
    # If this is the first detection or 
    # the detection is within the margin of the averaged n last lines 
    if left_line.bestx is None or np.abs(np.subtract(left_line.bestx, np.mean(left_line_allx, axis=0))) < 100:
        left_line.update_lane(left_line_ally, left_line_allx)
        left_line.detected = True
    else:
        left_line.detected = False
    if right_line.bestx is None or np.abs(np.subtract(right_line.bestx, np.mean(right_line_allx, axis=0))) < 100:
        right_line.update_lane(right_line_ally, right_line_allx)
        right_line.detected = True
    else:
        right_line.detected = False    
 
    # Calculate vehicle-lane offset
    xm_per_pix = 3.7/610 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    car_position = img_size[0]/2
    l_fit = left_line.current_fit
    r_fit = right_line.current_fit
    left_lane_base_pos = l_fit[0]*img_size[1]**2 + l_fit[1]*img_size[1] + l_fit[2]
    right_lane_base_pos = r_fit[0]*img_size[1]**2 + r_fit[1]*img_size[1] + r_fit[2]
    lane_center_position = (left_lane_base_pos + right_lane_base_pos) /2
    left_line.line_base_pos = (car_position - lane_center_position) * xm_per_pix +0.2
    right_line.line_base_pos = left_line.line_base_pos

def find_lanes(img):
    if left_line.detected and right_line.detected:  # Perform margin search if exists prior success.
        # Margin Search
        left_lane_inds, right_lane_inds,out_img = margin_search(img)
        # Update the lane detections
        validate_lane_update(img, left_lane_inds, right_lane_inds)
        
    else:  # Perform a full window search if no prior successful detections.
        # Window Search
        left_lane_inds, right_lane_inds,out_img = window_search(img)
        # Update the lane detections
        validate_lane_update(img, left_lane_inds, right_lane_inds)
    return out_img


def write_stats(img):
    font = cv2.FONT_HERSHEY_PLAIN
    size = 3
    weight = 2
    color = (255,255,255)
    
    radius_of_curvature = (right_line.radius_of_curvature + right_line.radius_of_curvature)/2
    cv2.putText(img,'Lane Curvature Radius: '+ '{0:.2f}'.format(radius_of_curvature)+'m',(30,60), font, size, color, weight)

    if (left_line.line_base_pos >=0):
        cv2.putText(img,'Vehicle is '+ '{0:.2f}'.format(left_line.line_base_pos*100)+'cm'+ ' Right of Center',(30,100), font, size, color, weight)
    else:
        cv2.putText(img,'Vehicle is '+ '{0:.2f}'.format(abs(left_line.line_base_pos)*100)+'cm' + ' Left of Center',(30,100), font, size, color, weight)
        
        
def draw_lane(undist, img, Minv):
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.stack((warp_zero, warp_zero, warp_zero), axis=-1)

    left_fit = left_line.best_fit
    right_fit = right_line.best_fit
    
    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (64, 224, 208))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.6, 0)
        write_stats(result)
        return result
    return undist

def assemble_img(warped, threshold_img, polynomial_img, lane_img):
    # Define output image
    # Main image
    img_out=np.zeros((720,1707,3), dtype=np.uint8)
    img_out[0:720,0:1280,:] = lane_img
    
    # Text formatting
    fontScale=1
    thickness=1
    fontFace = cv2.FONT_HERSHEY_PLAIN
    
    # Perspective transform image
    img_out[0:240,1281:1707,:] = cv2.resize(warped,(426,240))
    boxsize, _ = cv2.getTextSize("Transformed", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Transformed", (int(1494-boxsize[0]/2),40), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
   
    # Threshold image
    resized = cv2.resize(threshold_img,(426,240))
    resized=np.uint8(resized)
    gray_image = cv2.cvtColor(resized*255,cv2.COLOR_GRAY2RGB)
    img_out[241:481,1281:1707,:] = cv2.resize(gray_image,(426,240))
    boxsize, _ = cv2.getTextSize("Filtered", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Filtered", (int(1494-boxsize[0]/2),281), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
 
    # Polynomial lines
    img_out[480:720,1281:1707,:] = cv2.resize(polynomial_img*255,(426,240))
    boxsize, _ = cv2.getTextSize("Detected Lanes", fontFace, fontScale, thickness)
    cv2.putText(img_out, "Detected Lanes", (int(1494-boxsize[0]/2),521), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
    
    return img_out

def test_process(img):
      
    # Masking image
    masked = mask_image(img)
    
    # Perspective transform image
    warped, M, Minv = pers_transform(img) #masked 할수 있음
    cv2.imshow("warped",warped)
    
    # Colour thresholding in S channel
    s_bin = hls_thresh(warped)
    
    # Colour thresholding in B channel of LAB
    b_bin = lab_b_channel(warped, thresh = (185, 255))

    # Gradient thresholding with sobel x
    x_bin = sobel_thresh(warped, orient='x', thresh_min=20, thresh_max=100)
    
    # Gradient thresholding with sobel y
    y_bin = sobel_thresh(warped, orient='y', thresh_min=50, thresh_max=150)
    
    # # sobel x, y를 canny로 바꿔주기
    # canny_show=canny_thresh(warped, 130, 220)
    # cv2.imshow("canny", canny_show)
    
    # Direction of gradient thresholding
    dir_bin = dir_thresh(warped, thresh_min=1.5, thresh_max=np.pi/2) #thresh_min=0 이었음
    #cv2.imshow("dl", dir_bin)
    
    # # Combining both thresholds
    combined = np.zeros_like(x_bin)
    combined[(s_bin==1) | (b_bin == 1)] = 255
    cv2.imshow("combined", combined)
    
    return combined, warped, Minv



def window_search(binary_warped):
    # Take a histogram of the bottom half of the image
    bottom_half_y = binary_warped.shape[0]/2
    histogram = np.sum(binary_warped[int(bottom_half_y):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines

    #원래 기본
    # midpoint = np.int(histogram.shape[0]/2)
    # leftx_base = np.argmax(histogram[:midpoint])
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # 오른쪽만 찾기
    midpoint = np.int(histogram.shape[0]*3/4)
    leftx_base = np.argmax(histogram[360:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 500
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Generate black image and colour lane lines
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]
        
    # Draw polyline on image
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
    cv2.polylines(out_img, [right], False, (255,255,0), thickness=5)
    cv2.polylines(out_img, [left], False, (255,255,0), thickness=5)
    
    return left_lane_inds, right_lane_inds, out_img


def margin_search(binary_warped):
    # Performs window search on subsequent frame, given previous frame.
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 30

    left_lane_inds = ((nonzerox > (left_line.current_fit[0]*(nonzeroy**2) + left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] - margin)) & (nonzerox < (left_line.current_fit[0]*(nonzeroy**2) + left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_line.current_fit[0]*(nonzeroy**2) + right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] - margin)) & (nonzerox < (right_line.current_fit[0]*(nonzeroy**2) + right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Generate a blank image to draw on
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(out_img)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.intc([left_line_pts]), (0,255,0))
    cv2.fillPoly(window_img, np.intc([right_line_pts]), (0,255,0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]
        
    # Draw polyline on image
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
    cv2.polylines(out_img, [right], False, (1,1,0), thickness=5)
    cv2.polylines(out_img, [left], False, (1,1,0), thickness=5)
    
    return left_lane_inds, right_lane_inds, out_img

def process_img(img):
      
    # Undistorting image
    # undist = camera.undistort(img)
    
    # Masking image
    masked = mask_image(img)
    
    # Perspective transform image
    warped, M, Minv = pers_transform(img)
    
    # Colour thresholding in S channel
    s_bin = hls_thresh(warped)
    
    # Colour thresholding in B channel of LAB
    b_bin = lab_b_channel(warped, thresh = (185, 255))
    
    # Combining both thresholds
    # combined = np.zeros_like(s_bin)
    # combined[(s_bin==1) | (b_bin == 1)] = 1
    combined=s_bin
    
    # Find Lanes
    output_img = find_lanes(combined)
    
    # Draw lanes on image
    lane_img = draw_lane(img, combined, Minv);    
    
    result = assemble_img(warped, combined, output_img, lane_img)    
    
    return result

left_line = Line()
right_line = Line()
Rright_line=Line()



"""----------------------------------------------------------------------------------------------------------------------------"""



"""------------------------Data Directory------------------------------------"""
cfg = "cfg/yolov3.cfg"
weights = "weights/yolov3.weights"
names = "data/coco.names"

video_directory = ""
video = video_directory + args.video

# URL = "https://youtu.be/jieP4QkVze8"
# URL = "https://youtu.be/YsPdvvixYfo" roi = 1
url = args.url
if url:
    vpafy = pafy.new(url)
    play = vpafy.getbest(preftype = "mp4")

"""--------------------------Changeable Variables----------------------------"""
frames = 0
first_frame = 1

start = 0
batch_size = 1
confidence = 0.8 # 신뢰도
nms_thesh = 0.3 # Intersection of union의 범위를 설정해줌 (낮을수록 box 개수가 작아짐)
resol = 416 # 해상도

whalf, height = 640, 720

num_classes = 12
print("[INFO] Reading configure file")
model = Darknet(cfg)
print("[INFO] Reading weights file")
model.load_weights(weights) 
print("[INFO] Reading classes file")
classes = load_classes(names)
set_requires_grad(model, False)
print("[INFO] Network successfully loaded!")

mcu_port = args.com
mcu_brate = args.brate # Baud rate
if mcu_port:
    mcu = serial.Serial(mcu_port, mcu_brate, timeout = 1)
    mcu.timeout = None

model.net_info["height"] = resol
input_dim = int(model.net_info["height"])
assert input_dim % 32 == 0
assert input_dim > 32

"""--------------------------Video test--------------------------------------"""
torch.cuda.empty_cache()

# CUDA = False
CUDA = torch.cuda.is_available()
if CUDA:
    model.cuda()
model.eval()

start = time.time()


if url:
    cap = cv2.VideoCapture(play.url)
else: # python main.py --com COM4 --youtube
    cap = cv2.VideoCapture(video)
print("\n[INFO] Video and Camera is now ready to show.")

clip1 = save_video('./out_videos/' + args.video.split("/")[-1], 12.0) # result 영상 저장
print('./out_videos/' + args.video.split("/")[-1])
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # zerof = laneregion(frame)
        # show = cv2.addWeighted(frame, 1, zerof, 0.6, 0)
        width, height = frame.shape[:2]

        cv2.rectangle(frame, (0,0), (300, 130), dark, -1)
        show_fps(frame, frames, start, color = yellow)
        warning_text(frame)

        """------------------------- Lane Detection -------------------------"""
        cpframe = frame.copy() # Lane frame copy
        prc_img, hough = process_image(cpframe)


        result_img_detection=process_img(cpframe)
        cv2.imshow("result_img_detect", result_img_detection)

        #cv2.imshow("test", prc_img)
        lane_detection = visualize(prc_img, args.roi)
        

        """------------------------ Object Detection ------------------------"""
        cnt = 0 # Car count
        vals = [2, 3, 5, 7]
        l_cnt, r_cnt, c_cnt = 0, 0, 0
        if frames %3 == 0: # Frame 높히기; 눈속임
            prep_frame = prep_image(frame, input_dim)
            frame_dim = frame.shape[1], frame.shape[0]
            frame_dim = torch.FloatTensor(frame_dim).repeat(1, 2)

            if CUDA:
                frame_dim = frame_dim.cuda()
                prep_frame = prep_frame.cuda()

            with torch.no_grad():
                output = model(Variable(prep_frame, True), CUDA)
            output = write_results(output, confidence, num_classes, nms_thesh)

            if type(output) is not int:
                frame_dim = frame_dim.repeat(output.size(0), 1)
                scaling_factor = torch.min(resol/frame_dim, 1)[0].view(-1, 1)

                output[:, [1, 3]] -= (input_dim - scaling_factor * frame_dim[:, 0].view(-1, 1))/2
                output[:, [2, 4]] -= (input_dim - scaling_factor * frame_dim[:, 1].view(-1, 1))/2
                output[:, 1:5] /= scaling_factor

                for i in range(output.shape[0]):
                    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, frame_dim[i,0])
                    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, frame_dim[i,1])

                zero_frame = np.zeros_like(frame) # Object frame zero copy
                list(write(x, zero_frame) for x in output) # list(map(lambda x: write(x, frame), output))

                crossx, crossy = lane_cross_point()

                l_poly = Polygon([(next_frame[0], next_frame[1]), (crossx, crossy), (crossx, 0), (0, 0), (0, 720)])
                r_poly = Polygon([(next_frame[6], next_frame[7]), (crossx, crossy), (crossx, 0), (1280, 0), (1280, 720)])
                c_poly = Polygon([(next_frame[0], next_frame[1]), (crossx, crossy), (next_frame[6], next_frame[7])]) # Center Polygon

                for x in output:
                    c1 = tuple(x[1:3].int())
                    c2 = tuple(x[3:5].int())
                    c1=(int(c1[0]), int(c1[1]))
                    c2=(int(c2[0]), int(c2[1]))
                    centx = int((c1[0]+c2[0])/2)
                    centy = int((c1[1]+c2[1])/2)
                    label = "{0}".format(classes[int(x[-1])])

                    carbox = Polygon([(c1[0], c1[0]), (c1[0], c1[1]), (c1[1], c1[1]), (c1[1], c1[0])])
                    carcent = Point((centx, centy)) # Car Center point
                    carundcent = Point((centx, c2[1]))
                    carupcent = Point((centx, c1[1]))

                    """ 차의 중앙 지점과 겹치는 곳이 있으면 그곳이 차의 위치 """
                    for val in vals:
                        if int(x[-1]) == val:
                            cnt += 1
                            if l_poly.intersects(carcent):
                                l_cnt += 1
                            if r_poly.intersects(carcent):
                                r_cnt += 1
                            if c_poly.intersects(carcent):
                                c_cnt += 1
                                if c_cnt > 1 : c_cnt = 1

                                # 앞 차량과의 거리계산
                                pl = carundcent.distance(Point(whalf-5, 720))
                                dist = (pl * 1.8 / (next_frame[6] - next_frame[2])) * 180/np.pi
                                dist = round(map(dist, 20, 40, 10, 70), 2)

                                # 앞 차량의 Detection Box----------------------------
                                cv2.rectangle(frame, c1, c2, blue, 1)

                                t_size = cv2.getTextSize(label, font2, 1, 1)[0]
                                c2 = c1[0] + t_size[0], c1[1] - t_size[1]

                                cv2.rectangle(frame, c1, c2, blue, -1)
                                #---------------------------------------------------

                                cv2.line(frame, (centx, c1[1]), (centx, c1[1]-120), purple, 1)
                                cv2.line(frame, (centx-50, c1[1]-120), (centx+40, c1[1]-120), purple, 1)
                                cv2.putText(frame, "{} m".format(dist), (centx-45, c1[1]-130), font, 0.6, purple, 1)

                            if l_cnt or r_cnt or c_cnt:
                                cnt = l_cnt + c_cnt + r_cnt

                object_detection = cv2.add(frame, zero_frame)
                lane_detection = cv2.addWeighted(object_detection, 1, lane_detection, 0.5, 0)

            else:
                lane_detection = cv2.addWeighted(frame, 1, lane_detection, 0.5, 0)

            cv2.putText(lane_detection, 'vehicles counting : {}'.format(cnt), (10, 75), font, 0.8, white, 1)
            cv2.putText(lane_detection, 'L = {0} / F = {2} / R = {1}'.format(l_cnt, r_cnt, c_cnt), (10, 100), font, 0.7, white, font_size)


            """------------------------- Result -----------------------------"""
            import screeninfo

            screen_id = 0

            screen = screeninfo.get_monitors()[screen_id]
            width, height = screen.width, screen.height
            #print("width :", width, "\nheight :", height, "\n\nlane : ", lane_detection.shape)
            #lane_detection = cv2.resize(lane_detection, (height, width, 3))
            cv2.namedWindow('lane_detection', cv2.WND_PROP_FULLSCREEN)
            # cv2.moveWindow('lane_detection', screen.x - 1, screen.y - 1)
            cv2.setWindowProperty('lane_detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Result", lane_detection)

            clip1.write(lane_detection)

        # lane_detection = cv2.addWeighted(frame, 1, lane_detection, 0.5, 0)
        # cv2.imshow("hough", lane_detection)
        frames += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
clip1.release()
cv2.destroyAllWindows()
