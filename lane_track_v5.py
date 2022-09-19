import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path

import pandas as pd
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# #횡단보도용
# from utils_cross.datasets import letterbox as letterbox_c
# from utils_cross.general import non_max_suppression as non_max_suppression_c
# from utils_cross.plots import Annotator as Annotator_c

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])



"""따오기"""
from shapely.geometry import Polygon, Point
from moviepy.editor import VideoFileClip
from sklearn.cluster import AgglomerativeClustering
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
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
from math import *

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
R_right_line=(0,0,0,0)

# 데이터마다 변경해야 할 변수 
img_width, img_height=1280, 720  #이미지 가로, 세로
middle_point=(595, 340)  #시야로 확인한 중간지점
#middle_point=(0,0)
h1_y=450 #h1 직선의 y좌표
h2_y =345 #h2 직선의 y좌표

X_x1=[]
y_x2=[]
lines_queue=[]

X_x1_=[]
y_x2_=[]
lines_queue_=[]

# lane type : 변수 # (-1 : 노란색선, 0 : 미측정, 1 : 흰색 점선, 2 : 흰색 실선)
left_type, right_type, r_right_type=0, 0, 0
R_right_detect=0 # (0: 맨 오른차선 인식 안함, 1 : 맨 오른차선 인식)
count_frames_to_update=[0,0,0,0,0]
current_lane=[0, 0, 0, 0, 0, 0, 0, 0]
last_current=0

#multi_roi 변수들
lane_width=290 #이는 측정한 값을 넣기로 함. 
scan_hwidth=50 #ROI 가로 너비

l_l_Roi=np.array([[512-lane_width-scan_hwidth, img_height], [512-lane_width-scan_hwidth, 0], [512-lane_width+scan_hwidth, 0], [512-lane_width+scan_hwidth, img_height]])
l_Roi=np.array([[512-scan_hwidth, img_height], [512-scan_hwidth, 0], [512+scan_hwidth, 0], [512+scan_hwidth, img_height]])
r_Roi=np.array([[768-scan_hwidth, img_height], [768-scan_hwidth, 0], [768+scan_hwidth, 0], [768+scan_hwidth, img_height]])
r_r_Roi=np.array([[768+lane_width-scan_hwidth, img_height], [768+lane_width-scan_hwidth, 0], [768+lane_width+scan_hwidth, 0], [768+lane_width+scan_hwidth, img_height]])


first_frame = 1
lane_detected=0
is_turn_right=0

"""
python main.py --com COM4 --video drive.mp4
python main.py --com COM4 --roi 1 --video drive06.mp4 --alpha 60
python main.py --com COM4 --url https://youtu.be/YsPdvvixYfo --roi 1 --alpha 60
"""


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
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

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
        return masked_image

# white color 추출
def hls_thresh(img, thresh_min=200, thresh_max=255):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float) 
    s_channel = hls[:,:,1]
    
    # Creating image masked in S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 255
    return s_binary

# yellow color 추출
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
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 255
    return binary_output



# 기울기 구하기
def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

# 직선 그리기
def draw_lines(img, lines, ori_img):
    global cache
    global first_frame
    global next_frame
    global lines_queue

    """초기화"""
    y_global_min = img.shape[0]
    #y_max = img.shape[0]
    y_max = 485 #ROI의 마지막

    l_slope, r_slope = [], []
    l_lane, r_lane = [], []

    det_slope = 0.5
    α = 0.2

    """선을 인식하면 기울기 정도에 따라 오른쪽 차선인지, 왼쪽 차선인지 구별"""
    if lines is not None:
        #temp_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        temp_img=np.copy(ori_img)
        extract_lines=find_scatter(lines,temp_img)
        if extract_lines is not None:
            for line in extract_lines:
                for x1,y1,x2,y2 in line:
                    slope = get_slope(x1,y1,x2,y2)
                    if slope > det_slope:
                        r_slope.append(slope) #기울기
                        r_lane.append(line) # 점들
                    elif slope < -det_slope:
                        l_slope.append(slope)
                        l_lane.append(line)

        # # 평행 직선 그리기
        # if abs(slope)>0.7:
        #     temp_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        #     find_scatter(lines,temp_img )

            y_global_min = min(y1, y2, y_global_min) # 최소 y좌표 저장

    if (len(l_lane) == 0 or len(r_lane) == 0): # 오류 방지, 동시에 2개의 차선이 모두 인식되어야 함. 
        return None

    """기울기와, 점 수치들 평균 값 구하기"""
    l_slope_mean = np.mean(l_slope, axis =0)
    r_slope_mean = np.mean(r_slope, axis =0)
    l_mean = np.mean(np.array(l_lane), axis=0)
    r_mean = np.mean(np.array(r_lane), axis=0)

    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return None

    # y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    #이건 없어도 되는거 아닌가
    if np.isnan((y_global_min - l_b)/l_slope_mean) or \
    np.isnan((y_max - l_b)/l_slope_mean) or \
    np.isnan((y_global_min - r_b)/r_slope_mean) or \
    np.isnan((y_max - r_b)/r_slope_mean):
        return None

    # x구하기
    l_x1 = int((y_global_min - l_b)/l_slope_mean)
    l_x2 = int((y_max - l_b)/l_slope_mean)
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)

    if abs(l_x1-r_x1)>100: #맨 위쪽 차폭 제한
        return None

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
    return extract_lines

# 가로를 세로로 변경
def lane_pts():
    pts = np.array([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], [next_frame[6], next_frame[7]], [next_frame[4], next_frame[5]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    return pts

"""sequential RANSAC을 통한 직선 검출"""
def find_scatter(lines, img):
    global X_x1, y_x2, lines_queue
    append_line_num=0
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = get_slope(x1,y1,x2,y2)
                h1_line_x1=((h1_y-y1)/slope+x1)-middle_point[0]
                h2_line_x2=((h2_y-y1)/slope+x1)-middle_point[0]
                if (abs(h1_line_x1) <(img_width/2)) and (abs(h2_line_x2) <(img_width/2)) and abs(slope)<10 and abs(h1_line_x1-h2_line_x2)<(img_width/2): #slope조건 빠지면 양옆 차선(4개) 모두 인식함
                    append_line_num+=1
                    X_x1.append(np.intc([h1_line_x1]))
                    y_x2.append(np.intc([h2_line_x2]))



        #queue 구조 생성 및 삭제            
        if len(lines_queue)<20:
            lines_queue.append(append_line_num)
            delete_lines=0
        else:
            delete_lines=lines_queue.pop(0) #queue 자료구조에서 삭제해야할 lines수
            lines_queue.append(append_line_num)

        for i in range(delete_lines):
            X_x1.pop(0)
            y_x2.pop(0)
        # X_x1.append(np.intc([0]))
        # y_x2.append(np.intc([0]))
    
    if len(lines_queue)>=20:
        plt.scatter(X_x1, y_x2, color="yellowgreen", marker=".", label="Inliers")

        ransac = linear_model.RANSACRegressor(residual_threshold=20)
        ransac.fit(X_x1, y_x2)
        inlier_mask = ransac.inlier_mask_
        X_x1_inlier=np.array(X_x1)[inlier_mask]
        y_x2_inlier=np.array(y_x2)[inlier_mask]
        #plt.scatter(X_x1_inlier, y_x2_inlier, color="red", marker=".", label="Inliers")
        #plt.xlim([-640, 640])
        #plt.ylim([-640, 640])

        extract_parallel_point=[]

        for i in range(len(X_x1_inlier)):
            cv2.line(img, (X_x1_inlier[i][0]+middle_point[0], h1_y), (y_x2_inlier[i][0]+middle_point[0], h2_y), red, 2)
            extract_parallel_point.append([[X_x1_inlier[i][0]+middle_point[0], h1_y,y_x2_inlier[i][0]+middle_point[0], h2_y ]])

        """여기있는 값을 가지고 한번 차선을 만들어보자"""
        #plt.show()
        cv2.imshow("lineline", img)
        return np.array(extract_parallel_point)





def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, ori_image):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img2=np.copy(ori_image)
    extract_lines=draw_lines(line_img, lines, line_img2)
    return line_img, extract_lines

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
                [625, 345],
                [900, 485],
    ])

    #사각형 ROI
    vertices2 = np.array([
                [260, 485],
                [0, 455],
                [0, 345],
                [1280, 345],
                [1280, 455],
                [900, 485],
    ])

    # #내가만든 ROI2(for multi_lane)
    # vertices1 = np.array([
    #             [0, 450],
    #             [485, 340],
    #             [645, 340],
    #             [845, 450],
    # ])

    # #직사각형 ROI
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

   # vertices2 = np.array([
   #             [430, 755],
   #             [645, 580],
   #             [980, 580],
   #             [1410, 755]
   # ])

    if flag == 0 : return vertices1
    if flag == 1 : return vertices2

hei = 25
alpha = 0
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

""" 현재 영상 프레임 표시 """
def show_fps(image, frames, start, color = white):
    now_fps = round(frames / (time.time() - start), 2)
    cv2.putText(image, "FPS : %.2f"%now_fps, (10, hei), font, 0.8, color, font_size)

""" Steering Wheel Control 시각화 """
def direction_line(image, height, whalf, color = yellow):
    cv2.line(image, (whalf-10+alpha, height), (whalf-10+alpha, 600), white, 2) # 방향 제어 기준선
    cv2.line(image, (whalf-10+alpha, height), (dxhalf, 600), red, 2) # 핸들 방향 제어
    cv2.circle(image, (whalf-10+alpha, height), 120, white, 2)

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
def per_transform(image): # Bird's eye view
    pts1 = np.float32([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], [next_frame[4], next_frame[5]], [next_frame[6], next_frame[7]]])
    pts2 = np.float32([[512, 0], [512, 720], [768, 0], [768, 720]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (1280, 720))
    Minv=cv2.getPerspectiveTransform(pts2, pts1) 
    # cv2.line(dst, (l_cent, 0), (l_cent, 720), red, 2)
    # cv2.line(dst, (r_cent, 0), (r_cent, 720), red, 2)
    # cv2.imshow("l_cent", dst)
    return dst, M, Minv

def clust(image):
    ve=get_pts()
    interested = region_of_interest(image,[ve])
    wheres = np.where(interested > 80)
    clustered = AgglomerativeClustering(3).fit_predict(wheres[1].reshape([-1,1]))

    plt.figure(figsize = (15, 7))
    plt.imshow(interested)
    colors = ['r','g','b']
    for i in range(3):
        plt.scatter(wheres[1][clustered==i],wheres[0][clustered==i],label=i,color=colors[i])
    plt.show()
    plt.show()

def point_Minv_point(input_point, Minv):
    new_lines=[]
    for i in range(len(input_point)):
        x=input_point[i][0]
        y=input_point[i][1]
        arr_result=Minv@np.array([[x], [y], [1]]) #내적, perspective -> 기존 좌표로 변경하기
        arr_weight=arr_result[2][0]
        new_lines.append([arr_result[0][0]/arr_weight, arr_result[1][0]/arr_weight ])
    out_lines=np.array(new_lines)
    #print(out_lines)
    return out_lines

def make_ROI(image, Minv): #이거 undistor이 필요함. 무조건. 
    ROI_image = np.copy(image)

    #warped 이미지에서 window search 적용해보기
    roi_mask=np.zeros_like(image)
    cv2.fillPoly(roi_mask, [l_l_Roi], (255, 255, 255))
    cv2.fillPoly(roi_mask, [l_Roi], (255, 255, 255))
    cv2.fillPoly(roi_mask, [r_Roi], (255, 255, 255))
    cv2.fillPoly(roi_mask, [r_r_Roi], (255, 255, 255))
    warped_roi_mask=cv2.bitwise_and(roi_mask,ROI_image )
    cv2.imshow("masked warped", warped_roi_mask)

    # warped 이미지에서 ROI 나타내기
    cv2.polylines(ROI_image, [l_l_Roi], True, lime)
    cv2.polylines(ROI_image, [l_Roi], True, lime)
    cv2.polylines(ROI_image, [r_Roi], True, lime)
    cv2.polylines(ROI_image, [r_r_Roi], True, lime)


    cv2.imshow("ROI_image_multi_lane", ROI_image)
    return warped_roi_mask, point_Minv_point(l_l_Roi, Minv), point_Minv_point(l_Roi, Minv), point_Minv_point(r_Roi, Minv), point_Minv_point(r_r_Roi, Minv)


def make_outline(image, kernel_size, low_thresh, high_thresh):
    gauss_gray = gaussian_blur(image, kernel_size) 
    canny_edges = canny(gauss_gray, low_thresh, high_thresh)
    return canny_edges

"""---------------------------------------------------------------------------------------------------------"""

def window_search(binary_warped):
    # Take a histogram of the bottom half of the image
    #binary_warped_temp= cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #out_img=np.copy(binary_warped)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    #midpoint = np.int(896)
    leftx_base = np.argmax(histogram[np.int(384):midpoint])+np.int(384)
    rightx_base = np.argmax(histogram[midpoint:np.int(896)]) + midpoint
    r_rightx_base = np.argmax(histogram[np.int(896):]) + np.int(896)
    #print("left", np.max(histogram[:midpoint]), "right", np.max(histogram[midpoint:np.int(896)]), "R_right", np.max(histogram[np.int(896):]))

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
    r_rightx_current=r_rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 100
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    r_right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        win_xRright_low = r_rightx_current - margin
        win_xRright_high = r_rightx_current + margin        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        good_Rright_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xRright_low) & (nonzerox < win_xRright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        r_right_lane_inds.append(good_Rright_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        if len(good_Rright_inds) > minpix:        
            r_rightx_current = np.int(np.mean(nonzerox[good_Rright_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    r_right_lane_inds = np.concatenate(r_right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    r_rightx = nonzerox[r_right_lane_inds]
    r_righty = nonzeroy[r_right_lane_inds] 


    # #차선 판별
    # min_lane_point=min(np.max(histogram[:midpoint]), np.max(histogram[midpoint:np.int(896)]))
    # if np.max(histogram[np.int(896):])>min_lane_point/2: #차선인가
    #     print("맨 오른차선 인식")
    
        
    return lefty, righty, r_righty, out_img



def detect_lane_and_type(mask_white, mask_yellow, M):
    global left_type, right_type, r_right_type # (-1 : 노란색선, 0 : 미측정, 1 : 흰색 점선, 2 : 흰색 실선)
    Wleft, Wright, Wr_right, Wout_img=window_search(mask_white)
    cv2.imshow("Wout_img", Wout_img)
    Yleft, Yright, Yr_right, Yout_img=window_search(mask_yellow)
    cv2.imshow("Yout_img", Yout_img)

    if len(Wleft)==0 or len(Wright)==0 or len(Wr_right)==0:
        return 1
    left_ratio=len(Yleft)/len(Wleft)
    right_ratio=len(Yright)/len(Wright)
    r_right_ratio=len(Yr_right)/len(Wr_right)

    if left_ratio>0.1 : 
        left_type=-1
    else:
        left_type=1 #여기서 흰색 실선일 수도 있지만 이는 배재한다. (추후 점선 실선 구분법 알고리즘 넣을것임)

    if R_right_detect==0: #맨 오른 차선 인식 못할 때
        if right_ratio>0.1 : #노란색 인식 했더라면
            right_type=-1
            r_right_type=0
        else:
            right_type=2
            r_right_type=0
    else : #맨 오른 차선 인식할 때
        if r_right_ratio>0.1 : #노란색 인식 했더라면
            right_type=1
            r_right_type=-1
        else : #노란색 인식 안했더라면
            right_type=1
            r_right_type=2

    #line이 맞는지 확인

    #점선인지, 직선인지

    #노란색이 섞여 있는지

def make_colorMask(image, roi_image, thresh):
    color_mask=np.zeros_like(image, dtype="uint8") #color 추출 마스크 생성
    mask_white = hls_thresh(roi_image) #하얀색 추출
    mask_yellow = lab_b_channel(roi_image, thresh) #노란색도 thresh 값 설정해야함
    cv2.imshow("mask_yellow", mask_yellow)
    #cv2.imshow("yello", mask_yellow)
    color_mask[(mask_white>=1)|(mask_yellow>=1)]=(255, 255, 255) #색 마스크 추출
    cv2.imshow("color_mask", color_mask)
    return mask_white, mask_yellow, color_mask

def find_parallel_line(img, rho, theta, threshold, min_line_len, max_line_gap, ori_image):
    canny_edges = canny(img, 150, 200)
    lines = cv2.HoughLinesP(canny_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros_like(ori_image)
    global X_x1_, y_x2_, lines_queue_

    append_line_num=0
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = get_slope(x1,y1,x2,y2)
                h1_line_x1=((0-y1)/slope+x1)-middle_point[0]
                h2_line_x2=((img_height-y1)/slope+x1)-middle_point[0]
                if (abs(h1_line_x1) <(img_width/2)) and (abs(h2_line_x2) <(img_width/2)) and abs(h1_line_x1-h2_line_x2)<(100): #slope조건 빠지면 양옆 차선(4개) 모두 인식함
                    append_line_num+=1
                    X_x1_.append(np.intc([h1_line_x1]))
                    y_x2_.append(np.intc([h2_line_x2]))

                #queue 구조 생성 및 삭제            
        if len(lines_queue_)<2:
            lines_queue_.append(append_line_num)
            delete_lines=0
        else:
            delete_lines=lines_queue_.pop(0) #queue 자료구조에서 삭제해야할 lines수
            lines_queue_.append(append_line_num)

        for i in range(delete_lines):
            X_x1_.pop(0)
            y_x2_.pop(0)

    if len(lines_queue_)>=2:
        #plt.scatter(X_x1_, y_x2_, color="yellowgreen", marker=".", label="Inliers")

        ransac_ = linear_model.RANSACRegressor(residual_threshold=5)
        ransac_.fit(X_x1_, y_x2_)
        inlier_mask = ransac_.inlier_mask_
        X_x1_inlier=np.array(X_x1_)[inlier_mask]
        y_x2_inlier=np.array(y_x2_)[inlier_mask]
        #plt.scatter(X_x1_inlier, y_x2_inlier, color="red", marker=".", label="Inliers")
        #plt.xlim([-640, 640])
        #plt.ylim([-640, 640])

        extract_parallel_point=[]

        for i in range(len(X_x1_inlier)):
            cv2.line(img, (X_x1_inlier[i][0]+middle_point[0], 0), (y_x2_inlier[i][0]+middle_point[0], img_height), red, 2)
            extract_parallel_point.append([[X_x1_inlier[i][0]+middle_point[0], 0,y_x2_inlier[i][0]+middle_point[0], img_height ]])

        """여기있는 값을 가지고 한번 차선을 만들어보자"""

        cv2.imshow("par", img)
        #if frames%32==0:
            #plt.show()
        return np.array(extract_parallel_point)
        
def draw_multi_lane(image, Minv):
    # (-1 : 노란색선, 0 : 미측정, 1 : 흰색 점선, 2 : 흰색 실선)
    global current_lane, last_current
    current=0
    copy_img=np.copy(image)
    if left_type==-1:
        cv2.line(copy_img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), red, 2)
    elif left_type==1:
        cv2.line(copy_img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), blue, 2)
    elif left_type==2:
        cv2.line(copy_img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), purple, 2)

    x1=R_right_line[0]
    y1=R_right_line[1]
    x2=R_right_line[2]
    y2=R_right_line[3]
    arr_result1=Minv@np.array([[x1], [y1], [1]])
    arr_w1=arr_result1[2][0]
    arr_result2=Minv@np.array([[x2], [y2], [1]])
    arr_w2=arr_result2[2][0]
    x1=arr_result1[0][0]/arr_w1
    y1=arr_result1[1][0]/arr_w1
    x2=arr_result2[0][0]/arr_w2
    y2=arr_result2[1][0]/arr_w2

    if R_right_detect==1:
        if r_right_type==-1:
            current=1
            cv2.line(copy_img, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
            cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), blue, 2)
        elif r_right_type==1:
            if right_type==-1:
                current=0
                cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), red, 2)
            else:
                current=2
                cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), blue, 2)
                cv2.line(copy_img, (int(x1), int(y1)), (int(x2), int(y2)), blue, 2)
        elif r_right_type==2:
            if right_type==-1:
                current=0
                cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), red, 2)
            else:
                current=1
                cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), blue, 2)
                cv2.line(copy_img, (int(x1), int(y1)), (int(x2), int(y2)), purple, 2)
    else : 
        if right_type==-1:
            current=0
            cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), red, 2)
        elif right_type==1: #다음과 같은 조건은 있을 수 없다
            current=1
            cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), blue, 2)
        elif right_type==2:
            current=0
            cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), purple, 2)

    current_lane.append(current)
    current_lane.pop(0)

    if sum(current_lane)/8>=current:
        result_current=current
    else:
        result_current=last_current

    cv2.imshow("copy_img_result", copy_img)
    last_current=result_current
    return copy_img, result_current


""" 차선 검출을 위한 이미지 전처리 """
def process_image(image):
    global first_frame, R_right_detect, count_frames_to_update, R_right_line

    """---------------Step 0: 이미지 전처리를 위한 변수 설정---------------------"""
    #이미지 크기 변경 -오류 (혹여나 다른 크기의 사이즈가 있을 수 있으니)
    #image = cv2.resize(image, (720, 1280))
    image=cv2.resize(image, dsize=(1280, 720), interpolation=cv2.INTER_LINEAR)

    #image preprocessing 변수들
    kernel_size = 3
    # Canny Edge Detection Threshold
    low_thresh = 150
    high_thresh = 200
    rho = 2
    theta = np.pi/180
    min_line_len = 50
    max_line_gap = 150
    thresh = 50

    #ROI 정하기
    if lane_detected==1:  #==0으로 바꿔야 하긴 한데 일단 놔두기. 
        vertices = [get_pts(flag = 0)]
    else:
        vertices=[get_pts(flag = 1)]

    roi_image = region_of_interest(image, vertices)

    """---------------Step 1: 주행 차선 인식하기------------------"""
    #Canny를 사용해 윤곽선 검출
    outline_image=make_outline(roi_image, kernel_size, low_thresh, high_thresh)

    #2개의 차선 검출(직선 검출)
    line_image, extract_lines = hough_lines(outline_image, rho, theta, thresh, min_line_len, max_line_gap, roi_image)
    #여기서 2개 차선에 대한 정보는 """next_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype ="float32")""" 로 저장.
    cv2.imshow("extract_only_2_line_of_lane", line_image)

    #2개 차선, 원본 이미지와 합치기
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)

    """---------------Step 2: 다중 차선 인식하기--------------------"""
    #다중 차선 ROI 생성을 위한 시점 변환
    warped, M, Minv = per_transform(roi_image) 

    #다중 차선 ROI 생성 (원본 이미지에 대한 ROI의 vertices임)
    roi_generation_img, ROI_vertices_l_l, ROI_vertices_l, ROI_vertices_r, ROI_vertices_r_r=make_ROI(warped, Minv)

    #실제 차선에서 ROI 영역 표시
    multi_roi_img=np.copy(image)
    cv2.polylines(multi_roi_img, [np.intc(ROI_vertices_l_l)], True, lime)
    cv2.polylines(multi_roi_img, [np.intc(ROI_vertices_l)], True, (0, 255, 255))
    cv2.polylines(multi_roi_img, [np.intc(ROI_vertices_r)], True, (0, 255, 255))
    cv2.polylines(multi_roi_img, [np.intc(ROI_vertices_r_r)], True, (0, 255, 255))
    cv2.imshow("multi-roi", multi_roi_img)

    #색깔 마스크 생성
    mask_white, mask_yellow, color_mask=make_colorMask(image, roi_generation_img, thresh=(120, 245))

    #인식한 ROI속 검출된 pixel이 차선의 pixel인지 검증하는 단계
    if seen>20:
        #맨 오른쪽 ROI에 color_mask로 직선이 검출되는지 확인
        extracted_parallal_line=find_parallel_line(color_mask, rho, theta, thresh, min_line_len, max_line_gap, roi_image)
        if extracted_parallal_line is not None:
            for line in extracted_parallal_line:
                for x1, y1, x2, y2 in line:
                    if x2>(768+lane_width-scan_hwidth+1) and x1>(768+lane_width-scan_hwidth+1):
                        if R_right_detect==0: #맨 오른 차선을 인식 안할 때, 
                            count_frames_to_update.append(1)
                            count_frames_to_update.pop(0)
                            if sum(count_frames_to_update)>=5: #5프레임 모두 맨 오른차선 인식했을 때,
                                R_right_detect=1
                                R_right_line = np.array([x1, y1, x2, y2], dtype ="float32")
                                #print("detect most right lane")
                        else: #맨 오른 차선 인식했을 때
                            R_right_line = np.array([x1, y1, x2, y2], dtype ="float32")
                            count_frames_to_update.append(0)
                            count_frames_to_update.pop(0)
                            if sum(count_frames_to_update)<=0: #5프레임 모두 맨 오른차선 인식했을 때,
                                R_right_detect=0


    """---------------Step 3: 현재 상태 인식 및 결과 판단--------------------"""
    #감지한 직선이 어떤 차선인지 확인, 차선이라면 어떤 종류의 차선인지
    detect_lane_and_type(mask_white, mask_yellow, M) 
    multi_result, my_current_lane=draw_multi_lane(image, Minv)
    #print(my_current_lane)

    return result, line_image

""" 차선 검출 결과물을 보여줌 """
def visualize_func(image, flg):
    height, width = image.shape[:2]
    whalf = int(width/2)
    hhalf = int(height/2)

    zeros = np.zeros_like(image)
    vertices = [get_pts(flag=flg)]
    pts = lane_pts() #차선 정보 세로로 수정 #채우기 위해서

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
    cv2.putText(zeros, 'ROI', (930, 650), font, 0.8, yellow, font_size)
    cv2.polylines(zeros, vertices, True, (0, 255, 255))
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

"""------------따왔음 끝---------------"""

MODEL_PATH = 'weights/best.pt'












@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5n.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=[0, 1, 2, 3, 5, 7],  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        count=False,  # get counts of every obhects

):
    global seen

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    classes=[0, 1, 2, 3, 5, 7]
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0

        frame_height = im0s.shape[0]
        frame_width = im0s.shape[1]
        middle_line = frame_width//2

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # """횡단보도 용 preprocess"""
        # img_input = letterbox_c(im0s, imgsz[0], stride=stride)[0]
        # img_input = img_input.transpose((2, 0, 1))[::-1]
        # img_input = np.ascontiguousarray(img_input)
        # img_input = torch.from_numpy(img_input).to(device)
        # img_input = img_input.float()
        # img_input /= 255.
        # img_input = img_input.unsqueeze(0)

        # #inference
        # pred_c = model(img_input, augment=False, visualize=False)[0]
        # pred_c = torch.unsqueeze(pred_c, 0)

        # # postprocess
        # pred_c = non_max_suppression_c(pred_c, conf_thres, iou_thres,None, agnostic_nms, max_det=max_det)[0]

        # pred_c = pred_c.cpu().numpy()

        # pred_c[:, :4] = scale_coords(img_input.shape[2:], pred_c[:, :4], im0s.shape).round()
        # boxes_cross, confidences_cross, class_ids_cross = [], [], []

        # # Visualize
        # annotator_cross = Annotator_c(im0s.copy(), line_width=3, example=str(class_names), font='data/malgun.ttf')

        # cw_x1, cw_x2 = None, None # 횡단보도 좌측(cw_x1), 우측(cw_x2) 좌표
        # for p in pred_c:
        #     class_name = class_names[int(p[5])]
        #     x1, y1, x2, y2 = p[:4]

        #     annotator_cross.box_label([x1, y1, x2, y2], '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])
        #     print(class_name)

        #     if class_name == '횡단보도':
        #         cw_x1, cw_x2 = x1, x2
        #         print("횡단보도")
        # result_img = annotator_cross.result()
        # cv2.imshow('result_cross', result_img)

        # """횡단보도 끝"""


        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop


            annotator = Annotator(im0, line_width=2, pil=not ascii)
            
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            """이미지 읽기 시작"""
            cpframe = im0.copy() # Lane frame copy
            prc_img, hough = process_image(cpframe)
            #cv2.imshow("test", prc_img)
            lane_detection = visualize_func(prc_img, 1)


            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        bbox_left, bbox_top, bbox_right, bbox_bottom = bboxes 

                        # 맨 밑변의 아래가 필요
                        vehicle_side = -2
                        if bbox_right < middle_line:
                            vehicle_side = 0  ## LEFT
                        if bbox_left > middle_line:
                            vehicle_side = 1  ## RIGHT

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path+'.txt', 'a') as f:
                                f.write(('%g ' * 12 + '\n') % (frame_idx + 1, cls, id, int(vehicle_side), bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                        #bbox_bottom 아래

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')


            if count:
                itemDict={}
                ## NOTE: this works only if save-txt is true
                try:
                    df = pd.read_csv(txt_path +'.txt' , header=None, delim_whitespace=True)
                    df = df.iloc[:,0:3]
                    df.columns=["frameid" ,"class","trackid"]
                    df = df[['class','trackid']]
                    df = (df.groupby('trackid')['class']
                            .apply(list)
                            .apply(lambda x:sorted(x))
                            ).reset_index()

                    df.colums = ["trackid","class"]
                    df['class']=df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                    vc = df['class'].value_counts()
                    vc = dict(vc)

                    vc2 = {}
                    for key, val in enumerate(names):
                        vc2[key] = val
                    itemDict = dict((vc2[key], value) for (key, value) in vc.items())
                    itemDict  = dict(sorted(itemDict.items(), key=lambda item: item[0]))
                    # print(itemDict)

                except:
                    pass

                if save_txt:
                    ## overlay
                    display = im0.copy()
                    h, w = im0.shape[0], im0.shape[1]
                    x1 = 10
                    y1 = 10
                    x2 = 10
                    y2 = 70

                    txt_size = cv2.getTextSize(str(itemDict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(im0, (x1, y1 + 1), (txt_size[0] * 2, y2),(0, 0, 0),-1)
                    cv2.putText(im0, '{}'.format(itemDict), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,0.7, (210, 210, 210), 2)
                    cv2.addWeighted(im0, 0.7, display, 1 - 0.7, 0, im0)

            
            """여기까지가 이미지 인식 끝"""


            #current frame // tesing
            cv2.imwrite('testing.jpg',im0)


            if show_vid:
                # cv2.imshow(str(p), im0)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                resized_img=cv2.resize(im0, dsize=(1280, 720), interpolation=cv2.INTER_LINEAR)
                lane_detection = cv2.addWeighted(resized_img, 1, lane_detection, 0.5, 0)
                cv2.imshow("Result", lane_detection)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            prev_frames[i] = curr_frames[i]

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(lane_detection)


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--count', action='store_true', help='display all MOT counts results on screen')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    # Load model (횡단보도)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model_cross = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
    class_names = ['횡단보도', '빨간불', '초록불'] # model.names
    stride = int(model_cross.stride.max())
    colors = ((50, 50, 50), (0, 0, 255), (0, 255, 0)) # (gray, red, green)
    main(opt)

"""python lane_track_v5.py --source test_videos\drive_00.mp4 --yolo-weights weights/yolov5n.pt --img 640 --show-vid --save-vid"""