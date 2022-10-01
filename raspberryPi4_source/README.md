## Socket and HW
---

### Server(RPI)
```Shell
$./Alltime.sh
  ㄴ Web Streamer
    ㄴ ./wandlab-cv-streamer-master
  ㄴ perspective transform
  ㄴ Main HW ( + Socket)
```
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
    - Speaker 제어 포함 예정 (~9/26)

client로부터 받은 문자열을 3개(pillar, hill, right)로 나눈 후 각각에 맞는 HW 및 처리 진행

### Clinet(workstation)
```Shell
$ python3 socketserver01.py
```
yolo와 합친 코드로 변경 필요  
현재 `socketserver01.py`는 단순 통신 테스트를 위한 내용  
인식 결과값을 리스트(문자열)로 받아 server로 send

---

## Run YOLO & Lane Detection via Server Computer

Run after building YOLOv5 environment.

```Shell
$ python main_server.py --source source_video_name --weights weights/yolov5n.pt --img 640 --view-img
  ㄴ main_server.py
      ㄴ /utils/dataloaders.py
```
- Send Tokens to RPi
  - Uncomment lines
  ```Shell
  # clientSocket.connect((ip,port))
  # listsend(message, clientSocket)
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
