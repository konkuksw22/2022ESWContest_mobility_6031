# Perspective_final
### 목표
A필러에 띄우는 영상이 운전자의 시점과 일치할 수 있도록 양쪽 A필러 영상을 조절할 수 있도록 시점 변환을 진행한다.<br><br>

### 기능
#### 1) 양쪽 A필러
- 이미지 끝의 좌표를 잡고 `getPerspectiveTransform` 함수를 통해 변환행렬을 구한 뒤, 
`warpPerspective` 함수를 통해 시점 변환<br><br>
- 양쪽 A필러의 영상은 Buit_in cam의 영상, 가운데 영상은 운전자 시점 영상 송출<br>

#### 2) waxd 키
- `getsize`함수를 사용해 시점 변환 진행
- `keyboard.is_pressed`를 이용해 키보드 눌림이 감지되면 A필러의 영상 변환.
- 왼쪽 A필러 영상 조절 후, 오른쪽 A필러 영상 조절
- w: 좌표 위로 올리기, a: 사진 왼쪽으로 기울이기,<br> x: 좌표 아래로 내리기, d: 사진 오른쪽으로 기울이기
- 매번 shift하는 번거로움 → 변환배열 저장, 파일 실행 시 option
  - 0: 시점 변환 진행
  - 1 or default: 저장된 배열 값을 불러와 이전과 동일하게 자동 변환

#### 3) RPi 영상 딜레이
- RPi에서 `cv2.imshow`의 딜레이가 너무 느려 양쪽 A필러와 운전자 시점 영상을 `np.concatenate` 함수를 사용해 하나의 창으로 합쳐 띄움