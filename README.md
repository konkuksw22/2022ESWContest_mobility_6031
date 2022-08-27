# 우측 깜빡이, 차량 및 사람 인식
목표는 다음과 같다.   
차량이 교차로에서 우측 깜빡이를 통해 우회전시,    
횡단보도에 건너고 있는 사람이나 차량을 감지하기.   
1. 교차로 인식이 중요
2. 우측 차량 인식이 중요

* **제작자 : 정우진(Woo)**
* **제작 일 : 22.08.15 ~**
* **보고서 작성 일 : 22.08.27**


## 세부목표

1. 교차로 인식
>  - (1차 시도) 차선이 끊기는걸 확인하기 
>  -> 차선이 끊긴 것인지 빛으로 인한 차선이 끊긴 것처럼 보이는 것인지 확인하기 어려움
>  - (2차 시도) 정지선 및 횡단보도 인식
>  -> 정지선 감지 대신해서 횡단보도 인식은 yolo로 인식하기로 함
>  - (3차 시도) 자신이 현재 몇차선에 있는지 확인해서, 맨 끝차선에 있다면 우측 깜빡이를 통해 확인
>  -> 가장 젤 좋은 방법이라고 생각. 

몇차선에 있는지 확인하는 방법도 좋지만 (기술 고난이도), 그러나 (8월 23일) 그냥 차선 변경하기 위한 우측 깜빡이만 켜도 사물인식 하는 것으로 일단 하기로 함.    

2. 횡단보도에 건너고 있는 사람이나 차량을 감지
> - 오른쪽이나 왼쪽 차나 사람 인식
> -> 단순 화면의 중앙에서 왼쪽, 오른쪽을 구별해 사물을 인식할 수 있지만 (주의 경고와 함께)
> - 사물인식 움직임도 확인해 사물이 움직이는 것도 구별할 수 있으면 좋음. (특히 사람)
> - 또는 차선을 인식한 것을 통해 오른쪽, 정면, 왼쪽 차량 또는 사물 인식도 추가적으로 가능할 수 있음. 



## 일지
(1차 목표, 교차로 인식 성공 시키기)
* ~8월 12일
> 해당 날짜까지 1차 시도를 위해 관련된 자료 조사. 
> houghline search, window search, lanenet, 3d lane net 등등

* ~8월 17일
> houghline search, yolov3을 이용해 사물 인식하는 것을 성공

* ~8월 22일
> window search방법 연구, 효과적인 lane detect 방법 연구

* ~8월 29일
> 차선 인식에 어려움을 갖자, 우측깜빡이를 킬 때만 주변 사물을 인식하는 모드로 변경하기로 함(난이도는 많이 낮아짐)
> 더욱 효과적인 lane detect감지 방법 연구

## 연구 내용
1) hough line 장점
차량이나 도로 노면 표시가 아닌 경우,   
차선이 잘 보이는 경우, (차선만) 엉뚱한 차선 인식하는 빈도가 줆.   

다만 차선을 정확하게 인식하고자, window search 기법으로 변경하고자 하였음. 

2) window search 장점
비교적 정확한 범위의 차선을 인식하지만,   
주변 환경에 매우 민감하게 반응함.   

-> 이를 막고자 차선을 하나의 식으로 표현해 예상하고 확인하는 기법으로 변경
(window search, margin search)

-> 도로 노면 표시를 차선으로 인식해 차선인식의 정확도가 떨어지는 경우가 있어, 도로 노면 제거 기법 연구
(단순 차선을 인식하면 중앙을 지우는 기법 적용, 그 이후 후속과제로 군집화를 통해 노면을 제거하고자 함. )

-> 이후 hough line과 window search 를 혼합하는 기법을 고안하고자 함. 

## 현재 최고의 알고리즘
1. 교차로 인식
방안 1) 단순 우측 깜빡이 -> 사물 인식
후속 1) 우측 깜빡이 -> 사물 움직임 감지
후속 2) 우측 깜빡이 -> 사물 위치 감지 -> 사물 움직임 감지

방안 2) 차선인식 -> 현재 차선 위치 인식 -> 깜빡이 -> 사물인식   

차선인식 알고리즘 방안   
버드 아이 뷰 -> 색 인식 mask (하얀색 & 노란색 추출) -> Canny -> 앞 차선만 ROI로 따오기 -> houghline 으로 일정 기울기 이상 직선만 가져오기-> 직선에서 window search -> 얻은 직선 차선으로 중앙 차선 ROI로 제거해서 노면 표지 제거 

---------------------------------------------------------
