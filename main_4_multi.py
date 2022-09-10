from __future__ import division
from array import array
from msilib import sequence
from turtle import color

from torch.autograd import Variable
from torch.cuda import FloatTensor
import torch.nn as nn

from darknet import Darknet, set_requires_grad
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
from util import *

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

# 데이터마다 변경해야 할 변수 
img_width, img_height=1280, 720  #이미지 가로, 세로
middle_point=(595, 340)  #시야로 확인한 중간지점
#middle_point=(0,0)
h1_y=450 #h1 직선의 y좌표
h2_y =335 #h2 직선의 y좌표

X_x1=[]
y_x2=[]

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
        return masked_image

# white color 추출
def hls_thresh(img, thresh_min=200, thresh_max=255):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float) 
    s_channel = hls[:,:,1]
    
    # Creating image masked in S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
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
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    return binary_output



# 기울기 구하기
def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

# 직선 그리기
def draw_lines(img, lines):
    global cache
    global first_frame
    global next_frame

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
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = get_slope(x1,y1,x2,y2)
                if slope > det_slope:
                    r_slope.append(slope) #기울기
                    r_lane.append(line) # 점들
                elif slope < -det_slope:
                    l_slope.append(slope)
                    l_lane.append(line)

                # 평행 직선 그리기
#                if abs(slope)>0.5:
#                    temp_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#                    find_scatter(lines,temp_img )

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

"""sequential RANSAC을 통한 직선 검출"""
def find_scatter(lines, img):
    global X_x1, y_x2
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = get_slope(x1,y1,x2,y2)
                h1_line_x1=((h1_y-y1)/slope+x1)-middle_point[0]
                h2_line_x2=((h2_y-y1)/slope+x1)-middle_point[0]
                if (abs(h1_line_x1) <(img_width/2)) and (abs(h2_line_x2) <(img_width/2)):
                    X_x1.append(np.intc([h1_line_x1]))
                    y_x2.append(np.intc([h2_line_x2]))
        X_x1.append(np.intc([0]))
        y_x2.append(np.intc([0]))
    
    if frames %20==0:
        plt.scatter(X_x1, y_x2, color="yellowgreen", marker=".", label="Inliers")

        ransac = linear_model.RANSACRegressor()
        ransac.fit(X_x1, y_x2)
        inlier_mask = ransac.inlier_mask_
        X_x1_inlier=np.array(X_x1)[inlier_mask]
        y_x2_inlier=np.array(y_x2)[inlier_mask]
        plt.scatter(X_x1_inlier, y_x2_inlier, color="red", marker=".", label="Inliers")
        plt.xlim([-640, 640])
        plt.ylim([-640, 640])

        for i in range(len(X_x1_inlier)):
            cv2.line(img, (X_x1_inlier[i][0]+middle_point[0], h1_y), (y_x2_inlier[i][0]+middle_point[0], h2_y), red, 2)

        cv2.imshow("lineline", img)


        plt.show()
        X_x1=[]
        y_x2=[]





def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
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
    cv2.line(dst, (l_cent, 0), (l_cent, 720), red, 2)
    cv2.line(dst, (r_cent, 0), (r_cent, 720), red, 2)
    cv2.imshow("l_cent", dst)
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
    lane_width=290 #이는 측정한 값을 넣기로 함. 
    scan_hwidth=40
    ROI_image = np.copy(image)

    r_r_Roi=np.array([[512-lane_width-scan_hwidth, img_height], [512-lane_width-scan_hwidth, 0], [512-lane_width+scan_hwidth, 0], [512-lane_width+scan_hwidth, img_height]])
    r_Roi=np.array([[512-scan_hwidth, img_height], [512-scan_hwidth, 0], [512+scan_hwidth, 0], [512+scan_hwidth, img_height]])
    l_Roi=np.array([[768-scan_hwidth, img_height], [768-scan_hwidth, 0], [768+scan_hwidth, 0], [768+scan_hwidth, img_height]])
    l_l_Roi=np.array([[768+lane_width-scan_hwidth, img_height], [768+lane_width-scan_hwidth, 0], [768+lane_width+scan_hwidth, 0], [768+lane_width+scan_hwidth, img_height]])

    cv2.polylines(ROI_image, [r_r_Roi], True, 255)
    cv2.polylines(ROI_image, [r_Roi], True, 255)
    cv2.polylines(ROI_image, [l_Roi], True, 255)
    cv2.polylines(ROI_image, [l_l_Roi], True, 255)

    out_point=point_Minv_point(r_r_Roi, Minv)

    cv2.imshow("wow", ROI_image)
    return ROI_image, point_Minv_point(r_r_Roi, Minv), point_Minv_point(r_Roi, Minv), point_Minv_point(l_Roi, Minv), point_Minv_point(l_l_Roi, Minv)

# def detect_lane_and_type(input_img,mask_white, mask_yellow, roi_vertices):


""" 차선 검출을 위한 이미지 전처리 """
def process_image(image):
    global first_frame

    #print(image.shape)
    #image = cv2.resize(image, (720, 1280, 3))
    height, width = image.shape[:2]

    kernel_size = 3

    # Canny Edge Detection Threshold
    low_thresh = 150
    high_thresh = 200

    rho = 2
    theta = np.pi/180
    #thresh = 100
    #min_line_len = 50
    #max_line_gap = 150
    min_line_len = 1
    max_line_gap = 50
    thresh = 50

    #ROI 생성
    if lane_detected==1:  #==0으로 바꿔야 하긴 한데 일단 놔두기. 
        vertices = [get_pts(flag = 0)]
    else:
        vertices=[get_pts(flag = 1)]

    roi_image = region_of_interest(image, vertices)

    #색깔 마스크 생성
    color_mask=np.zeros_like(image, dtype="uint8") #color 추출 마스크 생성
    mask_white = hls_thresh(roi_image) #하얀색 추출
    mask_yellow = lab_b_channel(roi_image, thresh = (185, 245)) #노란색도 thresh 값 설정해야함
    #cv2.imshow("yello", mask_yellow)
    color_mask[(mask_white==1)|(mask_yellow>=1)]=255 #색 마스크 추출
    cv2.imshow("color_mask", color_mask)

    #직선 검출
    canny_edges = canny(color_mask, low_thresh, high_thresh)
    warped, M, Minv = per_transform(canny_edges)  #Multi -lane 감지하기
    roi_generation_img, ROI_vertices_r_r, ROI_vertices_r, ROI_vertices_l, ROI_vertices_l_l=make_ROI(warped, Minv)
    multi_roi_img=np.copy(image)
    cv2.polylines(multi_roi_img, [np.intc(ROI_vertices_r_r)], True, lime)
    cv2.polylines(multi_roi_img, [np.intc(ROI_vertices_r)], True, (0, 255, 255))
    cv2.polylines(multi_roi_img, [np.intc(ROI_vertices_l)], True, (0, 255, 255))
    cv2.polylines(multi_roi_img, [np.intc(ROI_vertices_l_l)], True, (0, 255, 255))
    cv2.imshow("multi-roi", multi_roi_img)

    line_image = hough_lines(canny_edges, rho, theta, thresh, min_line_len, max_line_gap) #직선 검출

    cv2.imshow("test", line_image)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    # cv2.polylines(result, vertices, True, (0, 255, 255)) # ROI mask

    return result, line_image

""" 차선 검출 결과물을 보여줌 """
def visualize(image, flg):
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
lane_detected=0

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
print(video)
print("\n[INFO] Video and Camera is now ready to show.")

clip1 = save_video('./out_videos/' + args.video.split("/")[-1], 12.0) # result 영상 저장
print('./out_videos/' + args.video.split("/")[-1])
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        #zerof = laneregion(frame)
        # show = cv2.addWeighted(frame, 1, zerof, 0.6, 0)
        width, height = frame.shape[:2]

        cv2.rectangle(frame, (0,0), (300, 130), dark, -1)
        show_fps(frame, frames, start, color = yellow)
        warning_text(frame)

        """------------------------- Lane Detection -------------------------"""
        cpframe = frame.copy() # Lane frame copy
        prc_img, hough = process_image(cpframe)
        #cv2.imshow("test", prc_img)
        lane_detection = visualize(prc_img, 1)
        

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
