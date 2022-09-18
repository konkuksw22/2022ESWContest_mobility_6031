# 시점 변환
<목표><br>
A필러에 띄우는 영상이 운전자의 시점과 일치할 수 있도록 양쪽 A필러 영상을 조절할 수 있도록 한다.

## 문제 사항
원본 파일 origin을 기준으로<br>
  1. 왼쪽, 오른쪽 창 모두 띄우도록 하기<br>
  2. A필러의 창과 운전자 시점의 영상 달리하기<br>
  3. Wasd 키 delay 줄이기<br>
  4. wasd 키 활성화<br>
  5. 코드 최적화<br>

## 수정 사항
1. 오른쪽 A필러 띄우기<br>
  왼쪽 창을 띄우는 것과 마찬가지로 이미지 끝의 좌표를 잡고
  getPerspectiveTransform 함수를 통해 변환행렬을 구한 뒤,<br>
  warpPerspective 함수를 통해 시점 변환<br>
2. A필러의 창과 운전자 시점의 영상 달리하기
  중간에 보이는 창은<br>
  운전자 시점의 영상, 양 옆의 A필러 부분은 빌트인 캠 영상.<br>
  ->문제점 : 적절한 영상을 찾기 힘듬<br>
      1) 웹캠 영상과 빌트인 캠의 영상의 구도가 너무 다름<br>
      2) 웹캠 영상이 너무 밝아 영상의 편집이 필요<br>
3. wasd 키 delay<br>
  wasd 키를 사용해 사용자의 시점에 맞도록 개인적으로 조절<br>
  ->문제점
  
