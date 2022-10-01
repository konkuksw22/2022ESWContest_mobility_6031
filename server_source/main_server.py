# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import pandas as pd
from collections import Counter
from sklearn import linear_model
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from math import *
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from socket import *
import pickle
import time

ip='203.252.164.24'
port=8608

# clientSocket=socket(AF_INET, SOCK_STREAM)
# clientSocket.connect((ip,port))
print("Connect Success")

def listsend(cmd, client):
    # sentence="".join(map(str,cmd))
    # print(sentence)
    # print(cmd)
    # clientSocket.sendall(sentence.encode("utf-8"))
    clientSocket.sendall(cmd.encode("utf-8"))
    return 0

MODEL_PATH = 'weights/best.pt'

#---------------변수들------------------
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
middle_point=(704, 620)  #시야로 확인한 중간지점 #(middle_x, roi_bottom)
#middle_point=(0,0)
h1_y=720 #h1 직선의 y좌표
h2_y =520 #h2 직선의 y좌표

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

l_l_Roi=np.array([[512-lane_width+20-scan_hwidth, img_height], [512-lane_width-scan_hwidth+20, 0], [512-lane_width+scan_hwidth+20, 0], [512-lane_width+scan_hwidth+20, img_height]])
l_Roi=np.array([[512-scan_hwidth, img_height], [512-scan_hwidth, 0], [512+scan_hwidth, 0], [512+scan_hwidth, img_height]])
r_Roi=np.array([[768-scan_hwidth, img_height], [768-scan_hwidth, 0], [768+scan_hwidth, 0], [768+scan_hwidth, img_height]])
r_r_Roi=np.array([[768+lane_width+90-scan_hwidth, img_height], [768+lane_width-scan_hwidth+90, 0], [768+lane_width+scan_hwidth+90, 0], [768+lane_width+scan_hwidth+90, img_height]])

first_frame = 1
lane_detected=0
start=0
is_lane_center=0

hei = 25
alpha = 0 #changable 
font_size = 1
seen=0
lane_change_detected=0
whalf, height = 640, 720
cswalk_detected=0
cswalk_box=[0,0,0,0]

""" 핸들 조종 및 위험 메세지 표시 """
def warning_text(image):
    whalf, height = 640, 720
    center = whalf+50+alpha
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

""" 차선 검출을 위한 이미지 전처리 """
def process_image(image):
    global first_frame, R_right_detect, count_frames_to_update, R_right_line

    """---------------Step 0: 이미지 전처리를 위한 변수 설정---------------------"""
    #이미지 크기 변경 -오류 (혹여나 다른 크기의 사이즈가 있을 수 있으니)
    #image = cv2.resize(image, (720, 1280))

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
    vertices=[get_pts(flag = 1)]

    roi_image = region_of_interest(image, vertices)

    """---------------Step 1: 주행 차선 인식하기------------------"""
    #Canny를 사용해 윤곽선 검출
    outline_image=make_outline(roi_image, kernel_size, low_thresh, high_thresh)

    #2개의 차선 검출(직선 검출)
    line_image, extract_lines = hough_lines(outline_image, rho, theta, thresh, min_line_len, max_line_gap, roi_image) #직선 검출, 2개 차선 가져오기
    #여기서 2개 차선에 대한 정보는 """next_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype ="float32")""" 로 저장.
    #cv2.imshow("extract_only_2_line_of_lane", line_image)

    #2개 차선, 원본 이미지와 합치기
    #result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)

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
    # cv2.imshow("show_multi_roi", multi_roi_img)

    #색깔 마스크 생성
    mask_white, mask_yellow, color_mask=make_colorMask(image, roi_generation_img, thresh=(120, 245))

    """---------------Step 3: 현재 상태 인식 및 결과 판단--------------------"""
    #감지한 직선이 어떤 차선인지 확인, 차선이라면 어떤 종류의 차선인지
    #left_type, right_type, r_right_type # (-1 : 노란색선, 0 : 미측정, 1 : 흰색 점선, 2 : 흰색 실선)으로 결과가 출력
    detect_lane_and_type(mask_white, mask_yellow, M)

    #인식한 ROI속 검출된 pixel이 차선의 pixel인지 검증하는 단계
    #R_right_detect=> 맨 오른차선 인식했는지 여부, R_right_line => 측정한 R의 직선 좌표
    if right_type!=-1:
        if seen>20:
            #print("detectedtedteeteete")
            #맨 오른쪽 ROI에 color_mask로 직선이 검출되는지 확인
            extracted_parallal_line=find_parallel_line(color_mask, rho, theta, thresh, min_line_len, max_line_gap, roi_image)
            if extracted_parallal_line is not None:
                for line in extracted_parallal_line:
                    for x1, y1, x2, y2 in line:
                        if x2>(768+lane_width-scan_hwidth+2) and x1>(768+lane_width-scan_hwidth+2):
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
    multi_result_img, my_current_lane=draw_multi_lane(image, Minv) #멀티 lane이 보임

    # print(R_right_line)
    #return result, line_image
    # print(my_current_lane)
    return my_current_lane, multi_result_img #현재 내 차선 위치, 여러 차선이 보여주는 이미지

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

def make_ROI(image, Minv): #이거 undistor이 필요함. 무조건. 
    ROI_image = np.copy(image)

    #warped 이미지에서 window search 적용해보기
    roi_mask=np.zeros_like(image)
    cv2.fillPoly(roi_mask, [l_l_Roi], (255, 255, 255))
    cv2.fillPoly(roi_mask, [l_Roi], (255, 255, 255))
    cv2.fillPoly(roi_mask, [r_Roi], (255, 255, 255))
    cv2.fillPoly(roi_mask, [r_r_Roi], (255, 255, 255))
    warped_roi_mask=cv2.bitwise_and(roi_mask,ROI_image )
    # cv2.imshow("masked warped", warped_roi_mask)

    # warped 이미지에서 ROI 나타내기
    cv2.polylines(ROI_image, [l_l_Roi], True, lime)
    cv2.polylines(ROI_image, [l_Roi], True, lime)
    cv2.polylines(ROI_image, [r_Roi], True, lime)
    cv2.polylines(ROI_image, [r_r_Roi], True, lime)


    # cv2.imshow("ROI_image_multi_lane", ROI_image)
    return warped_roi_mask, point_Minv_point(l_l_Roi, Minv), point_Minv_point(l_Roi, Minv), point_Minv_point(r_Roi, Minv), point_Minv_point(r_r_Roi, Minv)

def make_outline(image, kernel_size, low_thresh, high_thresh):
    gauss_gray = gaussian_blur(image, kernel_size) 
    canny_edges = canny(gauss_gray, low_thresh, high_thresh)
    return canny_edges

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, ori_image):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img2=np.copy(ori_image)
    extract_lines=draw_lines(line_img, lines, line_img2)
    return line_img, extract_lines #line이 찍힌 이미지, 모든 추출한 선들

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

def make_colorMask(image, roi_image, thresh):
    color_mask=np.zeros_like(image, dtype="uint8") #color 추출 마스크 생성
    mask_white = hls_thresh(roi_image) #하얀색 추출
    mask_yellow = lab_b_channel(roi_image, thresh) #노란색도 thresh 값 설정해야함
    #cv2.imshow("mask_white", mask_white)
    #cv2.imshow("yello", mask_yellow)
    color_mask[(mask_white>=1)|(mask_yellow>=1)]=(255, 255, 255) #색 마스크 추출
    #cv2.imshow("color_mask", color_mask)
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
                h1_line_x1=((0-y1)/(slope+0.0001)+x1)-middle_point[0]
                h2_line_x2=((img_height-y1)/(slope+0.0001)+x1)-middle_point[0]
                # if (abs(h1_line_x1) <(img_width/2)) and (abs(h2_line_x2) <(img_width/2)) and abs(h1_line_x1-h2_line_x2)<(100): #slope조건 빠지면 양옆 차선(4개) 모두 인식함
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
        try:
            ransac_.fit(X_x1_, y_x2_)
        except ValueError:
            return None

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

        # cv2.imshow("par", img)
        #if frames%32==0:
            #plt.show()
        return np.array(extract_parallel_point)

def detect_lane_and_type(mask_white, mask_yellow, M):
    global left_type, right_type, r_right_type # (-1 : 노란색선, 0 : 미측정, 1 : 흰색 점선, 2 : 흰색 실선)
    Wleft, Wright, Wr_right, Wout_img=window_search(mask_white) #흰색 픽셀만 있는 점 좌표 모두 가져오기
    # cv2.imshow("Wout_img", Wout_img)
    Yleft, Yright, Yr_right, Yout_img=window_search(mask_yellow) #노란색 픽셀만 있는 점 좌표 모두 가져오기
    # cv2.imshow("Yout_img", Yout_img)

    if len(Wleft)==0 or len(Wright)==0 or len(Wr_right)==0:
        return 1

    #(해야할일)해당 선에 있는 직선만 따와야할 것 같은데 안했네(window_search에서 나중에 하자. )
    left_ratio=len(Yleft)/len(Wleft)
    right_ratio=len(Yright)/len(Wright)
    r_right_ratio=len(Yr_right)/len(Wr_right)

    if left_ratio>0.1 : 
        left_type=-1
    else:
        left_type=1 #여기서 흰색 실선일 수도 있지만 이는 배재한다. (추후 점선 실선 구분법 알고리즘 넣을것임)

    if right_ratio>0.1:
        right_type=-1
        r_right_type = 0
    else:
        right_type=1
        if r_right_ratio>0.1:
            r_right_type=-1
        elif len(Wr_right)>10:
            r_right_type=1
        else:
            r_right_type=0


    if R_right_detect==1: #맨 오른 차선 인식 때
        if r_right_ratio>0.1 : #노란색 인식 했더라면
            right_type=1
            r_right_type=-1
        else : #노란색 인식 안했더라면
            right_type=1
            r_right_type=2

    #line이 맞는지 확인

    #점선인지, 직선인지

    #노란색이 섞여 있는지

def draw_multi_lane(image, Minv):
    # (-1 : 노란색선, 0 : 미측정, 1 : 흰색 점선, 2 : 흰색 실선)
    global current_lane, last_current
    current=0 #현재 내가 있는 차선(결과)
    copy_img=np.zeros_like(image)
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

    if right_type==-1:
        current=0
        cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), red, 2)
    else:
        if r_right_type==-1:
            current=1
            cv2.line(copy_img, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
            cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), blue, 2)
            r_right=(int(x1), int(y1)), (int(x2), int(y2))
        else:
            current=2
            cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), blue, 2)
            cv2.line(copy_img, (int(x1), int(y1)), (int(x2), int(y2)), blue, 2)
            r_right = (int(x1), int(y1)), (int(x2), int(y2))

    # if R_right_detect==1: #맨 오른차선 detected
    #     if r_right_type==-1:
    #         current=1
    #         cv2.line(copy_img, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
    #         cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), blue, 2)
    #         r_right=(int(x1), int(y1)), (int(x2), int(y2))
    #     elif r_right_type==1:
    #         if right_type==-1:
    #             current=0
    #             cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), red, 2)
    #             r_right=None
    #         else:
    #             current=2
    #             cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), blue, 2)
    #             cv2.line(copy_img, (int(x1), int(y1)), (int(x2), int(y2)), blue, 2)
    #             r_right=(int(x1), int(y1)), (int(x2), int(y2))
    #     elif r_right_type==2:
    #         if right_type==-1:
    #             current=0
    #             cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), red, 2)
    #             r_right=None
    #         else:
    #             current=1
    #             cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), blue, 2)
    #             cv2.line(copy_img, (int(x1), int(y1)), (int(x2), int(y2)), purple, 2)
    #             r_right=(int(x1), int(y1)), (int(x2), int(y2))
    # else :
    #     current=0
    #     cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), red, 2)
        # if right_type==-1:
        #     current=0
        #     cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), red, 2)
        #     r_right=None
        # elif right_type==1: #다음과 같은 조건은 있을 수 없다
        #     current=1
        #     cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), blue, 2)
        #     r_right=None
        # elif right_type==2:
        #     current=0
        #     cv2.line(copy_img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), purple, 2)
        #     r_right=None


    current_lane.append(current)
    current_lane.pop(0)

    # if (sum(current_lane)/8>=current) and (lane_change_detected==1):
    if (sum(current_lane) / 8 >= current):
        result_current=current #현재 최종 위치
    else:
        result_current=last_current

    last_current=result_current
    return copy_img, result_current

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

# 직선 그리기
def draw_lines(img, lines, ori_img):
    global cache
    global first_frame
    global next_frame
    global lines_queue

    """초기화"""
    y_global_min = img.shape[0]
    #y_max = img.shape[0]
    y_max = 720 #ROI의 마지막

    l_slope, r_slope = [], []
    l_lane, r_lane = [], []

    det_slope = 0.2
    α = 0.2
    #cv2.imshow("gray", img)
    r_max=0
    r_min=img_width
    l_min=img_width
    l_max=0

    """선을 인식하면 기울기 정도에 따라 오른쪽 차선인지, 왼쪽 차선인지 구별"""
    if lines is not None:
        #temp_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        temp_img=np.copy(ori_img)
        extract_lines=find_scatter(lines,temp_img)
        #extract_lines=lines
        if extract_lines is not None:
            for line in extract_lines:
                for x1,y1,x2,y2 in line:
                    slope = get_slope(x1 - middle_point[0], y1, x2 - middle_point[0], y2)
                    if slope>det_slope: #right
                        r_max=max(r_max, x1)
                        r_min=min(r_min, x1)
                    elif slope < -det_slope: #left
                        l_max = max(l_max, x1)
                        l_min = min(l_min, x1)
            r_mean=(r_max+r_min)/2
            l_mean=(l_max+l_min)/2
            for line in extract_lines:
                for x1, y1, x2, y2 in line:
                    slope = get_slope(x1-middle_point[0],y1,x2-middle_point[0],y2)
                    if slope > det_slope:
                        if lane_change_detected==1:
                            r_slope.append(slope)  # 기울기
                            r_lane.append(line)  # 점들
                        else:
                            if x1<r_mean:
                                r_slope.append(slope) #기울기
                                r_lane.append(line) # 점들
                    elif slope < -det_slope:
                        if lane_change_detected==1:
                            l_slope.append(slope)
                            l_lane.append(line)
                        else:
                            if x1>l_mean:
                                l_slope.append(slope)
                                l_lane.append(line)

        # # 평행 직선 그리기
        # if abs(slope)>0.7:
        #     temp_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        #     find_scatter(lines,temp_img )

            y_global_min = min(y1, y2, y_global_min) # 최소 y좌표 저장
    #print(len(l_lane), len(r_lane))
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
    # print(l_b, r_b)

    deg_l=np.arctan2(l_slope_mean, 1)*180/np.pi-10
    deg_r = np.arctan2(r_slope_mean, 1) * 180 / np.pi - 10
    #print(deg_l, deg_r)
    if abs(abs(deg_r)-abs(deg_l))>40:
        return None

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

    if abs(l_x1-r_x1)>300 or abs(l_x2-r_x2)<50: #맨 위쪽 차폭 제한
        return None

    # if is_lane_center == 1:
    #     if (abs(l_x2-middle_point[0])<20) or (abs(r_x2-middle_point[0])< 20):
    #         return None

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


    #middle cross point of y has to over h2_y
    x_ = int((l_x1 * l_slope_mean - l_y1 - r_x2 * r_slope_mean + r_y2) / (l_slope_mean - r_slope_mean))
    y_ = int(l_slope_mean * (x_ - l_x1) + l_y1)
    if (y_>h2_y):
        return None
    #print("/t", x_, y_)
    # cv2.circle(img, (x_, y_), 2, red, 2)
    # cv2.imshow("gogogo", img)

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

"""sequential RANSAC을 통한 직선 검출"""
def find_scatter(lines, img):
    global X_x1, y_x2, lines_queue

    append_line_num=0
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = get_slope(x1,y1,x2,y2)
                h1_line_x1=((h1_y-y1)/(slope+0.001)+x1)-middle_point[0]
                h2_line_x2=((h2_y-y1)/(slope+0.001)+x1)-middle_point[0]
                if (abs(h1_line_x1) <(img_width/2)) and (abs(h2_line_x2) <(img_width/2)) and abs(h1_line_x1-h2_line_x2)<(img_width/2): #slope조건 빠지면 양옆 차선(4개) 모두 인식함
                    if abs(h1_line_x1)<600 and abs(h2_line_x2)<200:
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
        #plt.scatter(X_x1, y_x2, color="yellowgreen", marker=".", label="Inliers")

        ransac = linear_model.RANSACRegressor(residual_threshold=20)
        try:
            ransac.fit(X_x1, y_x2)
        except ValueError:
            return None
        inlier_mask = ransac.inlier_mask_
        X_x1_inlier=np.array(X_x1)[inlier_mask]
        y_x2_inlier=np.array(y_x2)[inlier_mask]
        # plt.scatter(X_x1_inlier, y_x2_inlier, color="red", marker=".", label="Inliers")
        # plt.xlim([-640, 640])
        # plt.ylim([-640, 640])

        extract_parallel_point=[]

        for i in range(len(X_x1_inlier)):
            cv2.line(img, (X_x1_inlier[i][0]+middle_point[0], h1_y), (y_x2_inlier[i][0]+middle_point[0], h2_y), red, 2)
            extract_parallel_point.append([[X_x1_inlier[i][0]+middle_point[0], h1_y,y_x2_inlier[i][0]+middle_point[0], h2_y ]])

        """여기있는 값을 가지고 한번 차선을 만들어보자"""
        # plt.show()
        # cv2.imshow("lineline", img)
        return np.array(extract_parallel_point)

def window_search(binary_warped):
    # Take a histogram of the bottom half of the image
    #binary_warped_temp= cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #out_img=np.copy(binary_warped)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int32(histogram.shape[0]/2)
    #midpoint = np.int(896)
    leftx_base = np.argmax(histogram[np.int32(384):midpoint])+np.int32(384)
    rightx_base = np.argmax(histogram[midpoint:np.int32(896)]) + midpoint
    r_rightx_base = np.argmax(histogram[np.int32(896):]) + np.int32(896)
    #print("left", np.max(histogram[:midpoint]), "right", np.max(histogram[midpoint:np.int(896)]), "R_right", np.max(histogram[np.int(896):]))

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int32(binary_warped.shape[0]/nwindows)
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
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        if len(good_Rright_inds) > minpix:        
            r_rightx_current = np.int32(np.mean(nonzerox[good_Rright_inds]))

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

""" 차선 검출 결과물을 보여줌 """
def visualize_f(image, flg):
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
            center_line=direction_line(zeros, height = height, whalf = whalf)
            lane_position(zeros, center=center_line)  #핸들을 돌렸는지 확인함. 

    """ Lane Detection ROI """
    cv2.putText(zeros, 'ROI', (930, 650), font, 0.8, yellow, font_size)
    cv2.polylines(zeros, vertices, True, (0, 255, 255))
    result=weighted_img(zeros, image, α=0.8, β=1., λ=0.)
    return result

# 가로를 세로로 변경
def lane_pts():
    pts = np.array([[next_frame[0], next_frame[1]], [next_frame[2], next_frame[3]], [next_frame[6], next_frame[7]], [next_frame[4], next_frame[5]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    return pts

""" Steering Wheel Control 시각화 """
def direction_line(image, height, whalf, color = yellow):
    global is_lane_center
    cv2.line(image, (whalf+50+alpha, height), (whalf+50+alpha, 600), white, 2) # 방향 제어 기준선
    cv2.line(image, (whalf+50+alpha, height), (dxhalf, 600), red, 2) # 핸들 방향 제어
    cv2.circle(image, (whalf+50+alpha, height), 120, white, 2)

    c1_width=(dxhalf-whalf+50+alpha)
    c2_height=(height-600)
    deg=np.arctan2(c2_height, c1_width)*180/np.pi-10
    if abs(deg-90)<25:
        is_lane_center=1
    else:
        is_lane_center=0
    print("deg", deg)
    return whalf+50+alpha

""" 왼쪽 차선, 오른쪽 차선, 그리고 차선의 중앙 지점 표시 """
def lane_position(image, gap = 20, length=20, thickness=2, color = red, bcolor = white, center=img_width/2): # length는 선의 위쪽 방향으로의 길이
    global l_cent, r_cent, lane_change_detected

    if lane_change_detected==0:
        if (l_center[0]>center-15) or r_center[0]<center+15:
            lane_change_detected=1
    else:
        if (l_center[0]<center-30) and r_center[0]>center+30:
            lane_change_detected=0
    cv2.line(image, (l_center[0], l_center[1]+length), (l_center[0], l_center[1]-length), color, thickness)
    cv2.line(image, (r_center[0], r_center[1]+length), (r_center[0], r_center[1]-length), color, thickness)
    # print("lane changed?", lane_change_detected)

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


# white color 추출
def hls_thresh(img, thresh_min=200, thresh_max=255):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float_)
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

# 두개의 이미지를 더함
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    # result = imgA * a + imgB * b + c
    return cv2.addWeighted(initial_img, α, img, β, λ)

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

# 기울기 구하기
def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1+0.001)

def map(x, in_min, in_max, out_min, out_max):
    return ((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

def get_pts(flag=0):
    # vertices1 = np.array([
    #             [230, 650],
    #             [620, 460],
    #             [670, 460],
    #             [1050, 650]
    #             ])

    #내가만든 ROI
    vertices1 = np.array([
                [0, 720],
                [0, 550],
                [1280, 550],
                [1280, 720],
    ])

    #사각형 ROI
    vertices2 = np.array([
                [48, 620],
                [48, 500],
                [1246, 500],
                [1246, 620],
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

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=[0, 1, 2, 3, 5, 7],  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global seen
    start = time.time()
    #영상 로드
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    classes_cross = None  # filter by class
    #classes=[0, 1, 2, 3, 5, 7] #사람, 자전거, 차, 오토바이, 버스, 트럭
    classes = [0, 1, 2, 3]  # 사람, 자전거, 차, 오토바이, 버스, 트럭
    classes_name=["사람", "자전거", "차", "오토바이","0", "버스", "0", "트럭"]

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # 횡단보도 모델 로드
    device_c = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(MODEL_PATH, map_location=device_c)
    #model_c = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
    class_names_c = ['횡단보도', '빨간불', '초록불', '사람', '자전거', '자동차', '오토바이', '버스', '트럭'] # model.names
    #stride_c = int(model_c.stride.max())
    colors_c = ((50, 50, 50), (0, 0, 255), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)) # (gray, red, green)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    windows, dt =[], (Profile(), Profile(), Profile())
    test_val = 0
    for path, im, im0s, vid_cap, s in dataset:
        # print(test_val)
        test_val = test_val + 1
        """-----------------------------------------Pre-Process-----------------------------------------"""
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        vertices = [get_pts(flag=1)]
        im0s_for_cross = region_of_interest(im0s, vertices)
        #횡단보도 preprocessing
        # img_input_c = letterbox(im0s_for_cross, 640, stride=stride_c)[0]
        # img_input_c = img_input_c.transpose((2, 0, 1))[::-1]
        # img_input_c = np.ascontiguousarray(img_input_c)
        # img_input_c = torch.from_numpy(img_input_c).to(device_c)
        # img_input_c = img_input_c.float()
        # img_input_c /= 255.
        # img_input_c = img_input_c.unsqueeze(0)

        """-----------------------------------------Inference-----------------------------------------"""
        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            # 횡단보도 Inference
            #pred_c = model_c(img_input_c, augment=False, visualize=False)[0]

        """-----------------------------------------Apply NMS(non_max_suppression)----------------------"""
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # 횡단보도 Postprocess
            # pred_c = non_max_suppression(pred_c, 0.5, 0.45, classes_cross, False, max_det=1000)[0]
            # pred_c = pred_c.cpu().numpy()
            # pred_c[:, :4] = scale_coords(img_input_c.shape[2:], pred_c[:, :4], im0s.shape).round()
            boxes_c, confidences_c, class_ids_c = [], [], []

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        """----------------------------------------각각 다른 detections 시작-----------------------------"""
        ################################횡단보도 추출하기################################
        cw_x1, cw_x2 = None, None # 횡단보도 좌측(cw_x1), 우측(cw_x2) 좌표

        # for p in pred_c:
        #     global cswalk_detected, cswalk_box
        #     class_name_c = class_names_c[int(p[5])]
        #     x1, y1, x2, y2 = p[:4]
        #
        #
        #     # workspace
        #     #annotator.box_label([x1, y1, x2, y2], '%s %d' % (class_name_c, float(p[4]) * 100), color=colors_c[int(p[5])])
        #     #annotator.box_label(bboxes, label, color=colors(c, True))
        #
        #     if class_name_c == '횡단보도':
        #         cswalk_detected=1
        #         #cw_x1, cw_x2 = x1, x2
        #         print(class_name_c)
        #         cswalk_box=p[:4]
        #     else :
        #         cswalk_detected=0
        #
        #     #우회전 시, 초록불 감지시, 알람
        #     if cswalk_detected==1:
        #         if class_name_c=="초록불":
        #             print("잠시 멈췄다가 가세요")


        # 기존 코드
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            #이미지 크기 변경하기
            im0=cv2.resize(im0, dsize=(1280, 720), interpolation=cv2.INTER_LINEAR)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop

            zeros_copy=np.zeros_like(im0)
            annotator = Annotator(zeros_copy, line_width=line_thickness, example=str(names))

            """---------------------------------영상 데이터 수집 & 차선 수집 시작---------------------------"""
            ################################틀만들기##############################
            #frame_height, frame_width
            # show_fps(im0, seen, start, color = yellow)
            # warning_text(im0)
            # cv2.imshow("frame", im0)
            frame=im0.copy()
            """------------------------- Lane Detection -------------------------"""
            my_current_lane, multi_lane_img = process_image(frame) #현재 내가 있는 차선 출력, 다중 차선 이미지 출력
            # cv2.imshow("multi_lane_img", multi_lane_img)
            lane_detection = visualize_f(multi_lane_img, 1)  #차선은 직선형태로 next_frames으로 보내짐. 
            # 검은 화면 차선정보만 있는 화면이 lane_detection
            # cv2.imshow("lane_detection_v2", lane_detection)

            """------------------------- 객체 인식 시작 -------------------------"""
            obj_cnt = 0 # Car count
            l_cnt, r_cnt, c_cnt = 0, 0, 0

            L_car = L_person = L_bicycle = L_motorcycle = H_car = H_person = H_bicycle = H_motorcycle = R_person = R_car = R_bicycle = R_motorcycle = 0
            right_stat = 1 if my_current_lane == 0 else 0

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                crossx, crossy=lane_cross_point()
                l_poly = Polygon([(next_frame[0], next_frame[1]), (crossx, crossy), (crossx, 0), (0, 0), (0, 720)])
                r_poly = Polygon([(next_frame[6], next_frame[7]), (crossx, crossy), (crossx, 0), (1280, 0), (1280, 720)])
                c_poly = Polygon([(next_frame[0], next_frame[1]), (crossx, crossy), (next_frame[6], next_frame[7])]) # Center Polygon

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    bbox_left, bbox_top, bbox_right, bbox_bottom = xyxy #x1, y1, x2, y2
                    c1=(int(bbox_left), int(bbox_top))
                    c2=(int(bbox_right), int(bbox_bottom))
                    centx=int((bbox_left+bbox_right)/2)
                    centy=int((bbox_top+bbox_bottom)/2)
                    obj_cent=Point((centx, centy))
                    obj_bot_cent=Point((centx, c2[1]))

                    label = "{0}".format(classes_name[int(cls)])

                    obj_cnt += 1
                    if l_poly.intersects(obj_cent):
                        l_cnt += 1
                    if r_poly.intersects(obj_cent):
                        r_cnt += 1
                    if c_poly.intersects(obj_cent):
                        c_cnt += 1
                        if c_cnt > 1 : c_cnt = 1

                        # 앞 차량과의 거리계산
                        pl = obj_bot_cent.distance(Point(whalf+50+alpha, 720))
                        if (next_frame[6] - next_frame[2])!=0:
                            dist = (pl * 1.8 / (next_frame[6] - next_frame[2])) * 180/np.pi
                            dist = round(map(dist, 20, 40, 10, 70), 2)
                        else:
                            dist=0

                        # 앞 차량의 Detection Box----------------------------
                        cv2.rectangle(frame, c1, c2, blue, 1)

                        t_size = cv2.getTextSize(label, font2, 1, 1)[0]
                        c2_ = c1[0] + t_size[0], c1[1] - t_size[1]

                        cv2.rectangle(frame, c1, c2_, blue, -1)
                        #---------------------------------------------------

                        cv2.line(frame, (centx, c1[1]), (centx, c1[1]-120), purple, 1)
                        cv2.line(frame, (centx-50, c1[1]-120), (centx+40, c1[1]-120), purple, 1)
                        cv2.putText(frame, "{} m".format(dist), (centx-45, c1[1]-130), font, 0.6, purple, 1)

                    if l_cnt or r_cnt or c_cnt:
                        cnt = l_cnt + c_cnt + r_cnt

                    #횡단보도에 사람이 있는지 확인
                    # if cswalk_detected==1:
                    #     cs_x1, cs_y1, cs_x2, cs_y2 = cswalk_box
                    #     # cv2.putText(frame, "crossWalk!", (400, 400), font, 0.6, red, 2)
                    #     #우회전 시 신호등 초록불 감지하면 알람
                    #
                    #     #횡단보도 감지 안에 사람이 있으면 알람
                    #     if int(cls)==0:
                    #         if (cs_x1<=centx) and (cs_x2>=centx):
                    #             if (cs_y2<=c2[1]) and (cs_y1>=c2[1]):
                    #                 print("횡단보도 사람 감지")
                    # else:
                    #     cv2.putText(frame, " ", (400, 400), font, 0.6, red, 2)

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        if xywh[1] > 0.4 and xywh[1] < 0.8 :    # Hill
                            annotator.box_label(xyxy, label, color=colors(0, True))
                            if cls == 0:
                                H_person = 1
                            if cls == 2 or cls == 5 or cls == 7:
                                H_car = 1
                            if cls == 1 or cls == 3:
                                H_bicycle = 1
                            if cls == 3:
                                H_motorcycle = 1
                        if xywh[0] < 0.2:   # left
                            annotator.box_label(xyxy, label, color=colors(4, True))
                            if cls == 0:
                                L_person = 1
                            if cls == 2 or cls == 5 or cls == 7:
                                L_car = 1
                            if cls == 1 or cls == 3:
                                L_bicycle = 1
                            if cls == 3:
                                L_motorcycle = 1
                        if xywh[0] > 0.8:
                            if bbox_top>200:
                                annotator.box_label(xyxy, label, color=colors(7, True))
                                if cls == 0:
                                    R_person = 1
                                if cls == 2 or cls == 5 or cls == 7:
                                    R_car = 1
                                if cls == 1 or cls == 3:
                                    R_bicycle = 1
                                if cls == 3:
                                    R_motorcycle = 1
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                object_detect_result_img=annotator.result()
                object_detection = cv2.add(frame, object_detect_result_img)
                # cv2.imshow("object",object_detection )
                final_detected_img = cv2.addWeighted(object_detection, 1, lane_detection, 0.5, 0)

            else:
                final_detected_img = cv2.addWeighted(frame, 1, lane_detection, 0.5, 0)

            cv2.rectangle(final_detected_img, (0, 720 // 10 * 5), (1280, 720 // 10 * 8), (0, 255, 0),
                          3)  # Hill
            cv2.rectangle(final_detected_img, (0, 0), (1280 // 5, 720), (0, 0, 255),
                          3)  # Left A pillar
            cv2.rectangle(final_detected_img, (1280 // 5 * 4, 0), (1280, 720), (255, 0, 0),
                          3)  # Right A pillar
            cv2.rectangle(final_detected_img, (0,0), (330, 130), dark, -1)

            cv2.putText(final_detected_img, 'current_lane : rightmost', (10, 50), font, 0.8, red,
                        1) if my_current_lane == 0 else cv2.putText(final_detected_img,
                                                                    'current_lane : {}'.format(my_current_lane),
                                                                    (10, 50), font, 0.8, white, 1)
            cv2.putText(final_detected_img, 'object counting : {}'.format(obj_cnt), (10, 75), font, 0.8, white, 1)
            cv2.putText(final_detected_img, 'L = {0} / F = {2} / R = {1}'.format(l_cnt, r_cnt, c_cnt), (10, 100), font, 0.7, white, font_size)

            # print('A' + str(L_person) + str(L_car) + str(L_bicycle) + 'H' + str(H_person) + str(H_car) + str(
            #     H_bicycle) + 'R' + str(R_person) + str(R_car) + str(R_bicycle))
            # message = str(L_person) + str(L_car) + str(L_bicycle) + str(H_person) + str(H_car) + str(H_bicycle) + str(
            #     R_person) + str(R_car) + str(R_bicycle)
            message = str(L_car) + str(L_person) + str(L_bicycle) + str(R_car) + str(R_person) + str(R_bicycle) + str(
                H_car) + str(H_person) + str(H_bicycle) + str(right_stat)
            message = str(L_car) + str(L_person) + str(L_bicycle) + str(R_car) + str(R_person) + str(R_bicycle) + str(
                H_car) + str(H_person) + str(H_bicycle) + str(right_stat)
            # message = [L_car, L_person, L_bicycle, R_car, R_person, R_bicycle,
            #     H_car, H_person, H_bicycle, right_stat]
            # message = [L_car, L_person, L_bicycle, R_car, R_person, R_bicycle,
            #            H_car, H_person, H_bicycle]
            print(message)
            # if (test_val % 3 == 0):
            #     listsend(message, clientSocket)
            #listsend(message, clientSocket)
            # print(right_stat)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow("final_img", final_detected_img)
                # cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
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
                    vid_writer[i].write(final_detected_img)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    clientSocket.close()


"""python main_server_rev1.py --source Hill_3.avi --weights weights/yolov5n.pt --img 640 --view-img"""