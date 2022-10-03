# Team JJIDLE
Built in Driving Assitant   
내장형 블랙박스를 이용한 운전자 시야확장 시스템   
> 블랙박스가 운전자에 비해 시야가 더 넓다는 것을 이용함.     
> (1) 교차로에서 우회전 시, 운전자보다 사람을 먼저 인식해 경고로 사고 예방   
> (2) 경사로에서 보이지 않는 시야각으로 발생할 수 있는 사고 예방   
> (3) A 필러 부근 보이지 않는 시야 확장    

* 교차로 우회전 상황   

|운전자 시점|블랙박스 시점 영상 처리|
|--|--|
|![](./img/best_DMoment2.jpg)|![](./img/best_Moment2.jpg)|  

* 경사로 상황   

|운전자 시점|블랙박스 시점 영상 처리|
|--|--|
|![](./img/best_DMoment_hill.jpg)|![](./img/best_Moment_hill.jpg)|  

참고 : red box : 왼쪽 A필러 구역, blue box : 오른쪽 A필러 구역, green box : 전방 구역   

>- 교차로 우회전 경우, 차량이 맨 오른쪽에 있음을 상황 인지 후, 오른쪽 A필러 구역(Blue box area)의 사람을 감지해 운전자에게 alert   
>- 경사로 경우, 기울기 센서로 해당 차량이 경사로에 있음을 상황 인지 후, 전방 구역(Green box area)의 사람과 물체를 감지해 운전자에게 alert

## <div align="center">Team member</div>

| 이름   | 메일               | 역할 |
| ------ | ------------------ | ------ |
| 정우진 | woo9904@konkuk.ac.kr | 기획, 총괄 및 개발<br/>우회전 인식 시스템 개발 <br/>object detecting 알고리즘 개발|
| 박승철 | psc0526@konkuk.ac.kr | A필러 시점변환 개발<br/>서버 환경 구축 & 서버 코드 개발<br/>이미지 전처리 개발 |
| 신지혜 | long0404@konkuk.ac.kr | H/W개발 – Raspberry pi 4& 서버간 통신 환경 구축<br/>웹 스트리밍 서비스 개발<br/>센서, 소리 및 LED제어 |
| 이서연 | seoyeon8167@konkuk.ac.kr |A필러 시점변환 영상처리 및 코드 최적화 <br/> Object Detection 최적 모델 선별 <br/> 데이터 선별 |

## <div align="center">Quick Start Examples</div>
### 주의사항

> 실행은 
> 
> - server : CUDA 필요
> - [YOLOV5](https://github.com/ultralytics/yolov5) 설치 필요 
> - 기타 소프트 웨어는 requirments.txt 참고

<details open>
<summary>1. Server Computer Install</summary>

객체 검출의 bashline 코드는 이 [Yolov5](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) 코드를 참고하였다.    
Environment 또한 동일하게 
[**Python>=3.7.0**](https://www.python.org/) 환경에, 
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/)가 필요하다.   
리눅스 OS 환경에 설치하였다. 

> 1. baseline으로 사용한 yolov5 6.2ver을 clone 한다.
```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

> 2. 본 코드의 코드를 clone 한다. 
```bash
cd ../
git clone https://github.com/konkuksw22/esw22_JJIDLE5  # clone
```

> 3. "server_source" directory의 main_server.py 코드를 yolov5 폴더에 복사한다. 
```bash
cd esw22_JJIDLE5/server_source
cp main_server.py ../yolov5
```

> 4. 코드 실행을 위한 코드 수정 및 후 처리   
> CUDA 설치 필요. 
> yolov5/utils/dataloaders.py 변경
```bash
cd esw22_JJIDLE5
rm ../../yolov5/utils/dataloaders.py
cp server_source/dataloaders.py ../yolov5/utils
```
이로서 서버에서 필요한 준비는 모두 끝났다. 
</details>

<details open>
<summary>2. Raspberry Pi Install</summary>

본 GitHub의 코드를 clone하여 "raspberryPi4_source" directory의 코드를 설치한다. 

```bash
git clone https://github.com/konkuksw22/esw22_JJIDLE5  # clone
cd raspberryPi4_source
```
</details>

<details open>
<summary>3. 프로그램 사용하기 </summary>

> 1. Rpi4 준비   
> webstream 통신을 위한 ip를 accelerometer.py에 입력한다.
```bash
host = '203.###.###.##' #여기에 입력
port = int(sys.argv[1])
```
> Web server streaming 활성화, perspective_transform 영상 송출, 하드웨어 제어 파일을 동시에 실행한다.
```Shell
$ ./Alltime.sh 
```
> 2. Server 준비
> RPi 4와 연결을 하여 영상 이미지 받을 준비를 한다. 이때 통신을 위한 ip를 입력한다. 
```bash
ip='203.###.##.##' #여기에 입력
port=____
```
> 실시간 영상을 받아 서버에서 이미지 처리를 하고 판단 결과를 다시 RPI에 토큰으로 전송한다. 
> 저장된 영상으로 실행시키는 방법
```Shell
$ python main_server.py --source "source_video_name" --weights weights/yolov5n.pt --img 640 --view-img
```
> 실시간 영상으로 실행시키는 방법
```Shell
$ python main_server.py --source 0 --weights weights/yolov5n.pt --img 640 --view-img
```
</details>

## <div align="center">documents</div>

### github Tree

```bash
 │ README.md
 │ 
 ├── raspberryPi3_source
 │		   ├── wandlab-cv-streamer-master
 │		   │		├── wandlab.cv.streamer.py
 │		   │		└── webcam.test.py
 │		   ├── Alltime.sh
 │		   ├── accelerometer.py
 │		   └── perspective_transform.py
 ├── server_source
 │	   ├── camera_cal           #for camera calibration sources
 │     │	  	├── imags.png
 │	   ├── calibration_code.py  #for camera calibration
 │	   ├── dataloaders.py       #for image preprocessing
 │	   ├── main_server.py       #for server computer 
 │		 └── socketserver01.py    
 └── 일지
```

### 전체 구성도
![](./img/main_algorithm.jpg)

### 하드웨어 구성도
![](./img/hardware.jpg)


### 알고리즘

#### 1. server
#### 1-1. 파일 구성
```Shell
$ python main_server.py --source source_video_name --weights weights/yolov5n.pt --img 640 --view-img
  ㄴ main_server.py
      ㄴ /utils/dataloaders.py
```

##### 1-2. 함수 설명 및 특징
- Send Tokens to RPi
  - Uncomment lines
  ```Shell
  clientSocket.connect((ip,port))
  listsend(message, clientSocket)
  ```
- dataloaders.py (Update : 2022.9.27. 01:00)
  - Image resize (dst : 1280 * 720)
  - Histogram equalization (Normalize Intensity)
  - Undistortion
  ```Shell
  # -----------------------------------------------------------------------------------------------
  cv2.imshow("Original Image", im0)
  im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2YUV)  # Convert colorspace to YUV, seperate Intensity channel
  print(im0.shape)
  im0[:,:,0] = cv2.equalizeHist(im0[:,:,0])   # Normalize Intensity
  im0 = cv2.cvtColor(im0, cv2.COLOR_YUV2BGR)  # Convert to RGB
  cv2.imshow("Equalized Image", im0)
  # -----------------------------------------------------------------------------------------------
  height, width = im0.shape[:2]
  newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1)
  im0 = cv2.undistort(im0, mtx, dist, None, newcameramtx)
  cv2.imshow("Undistored Image", im0)
  # -----------------------------------------------------------------------------------------------
  ```

#### 2. Raspberry pi 4
##### 2-1. 파일 구성
```Shell
$./Alltime.sh
  ㄴ Web Streamer
    ㄴ ./wandlab-cv-streamer-master
  ㄴ perspective transform
  ㄴ Main HW ( + Socket)
```
##### 2-2. 특징

- Web Streamer : `./wandlab-cv-streamer-master/wandlab-cv-streamer.py`
    - `wandlab-cv-streamer-master` 경로 내 타 python 파일 `import` 중
- Perspective Transform : `./perspective/t.py (option)`
    - sudo 권한으로 실행 (`Alltime.sh`에 해당 내용 포함됨)
    - option :
        - '0' : 초기설정(방향값 .txt에 저장과정 포함)
        - '1' : 기존 설정값으로 transform 바로 진행 (`Alltime.sh`에는 이 옵션으로 포함됨)
- Main HW (+ Socket) : `./accelerometer.py`
    - 파라미터 1개 `$ChatPort` 필요(shell에서 지정됨)
    - Socket, ADXL, Switch, LED 등 Main 출력부 제어 포함
    - Speaker 제어 포함 

client로부터 받은 문자열을 3개(pillar, hill, right)로 나눈 후 각각에 맞는 HW 및 처리 진행





### Computing Power
영상 처리를 위한 서버컴퓨터와, 영상을 찍는데 사용한 Dashcam의 사양에 대해 적어본다. 

```
 1. Server Computer
 CPU : Intel i9-11900K
 RAM : DDR4 32GB RAM
 SSD : 1TB M.2 NVME SSD
 VGA : Nvidia Geforce RTX3090
 CUDA : CUDA 11.7

  2. Dash cam 
  model : AMS7 FF 블랙박스 / FULL HD 
  렌즈 : 2.0M Pixel, F2.0 CMOS
  화각 : 약 130도 (대각기준)
  해상도 및 프레임 : 1920x1080 (최대 30프레임)
  영상/음성 코덱 : H.264

```

## HW
> 하드웨어 구성
```
 Raspberry Pi 4B+
 ADXL345
 JLED-bar-10
 Logitech Brio 4K PRO Web Cam
 RPi 160도 광각 카메라 모듈 5MP
```

## TODOS

- 실시간 영상처리 최적화
- 객체 인식 학습
- 최적의 이미지 전처리 
