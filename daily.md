# 진행상황 기록하기
## Rubiks Cube in Manipulator
---
## 22.04
- tensorflow를 이용해 동작 학습시키고 이동하게 하기
- Gripper 교체 모델 고안하기
- Master & Slave 조작해 두 개의 Manipulator 동시 작동하게 하기

### 04.02
> open_manipulator_recording_trajectory.cpp : 동작 완료  
> 모션을 읽어 txt 파일로 저장할 수 있음  
> PiCAM+RPI에서 이미지를 얻을 수 있음  
>> MasterPC로 넘겨주는 작업이 필요함

> open_manipulator_play_recorded_trajectory.cpp  
> cmake 결과 오류가 남.
>> 코드 상에서 obj, glue, eclipse 가 없는 등 오류가 발생함.  
>> 받아오는 이미지(Object)가 없어서 나는 오류인 듯 보임.  
>> 헤더 파일에 해당 내용이 없다는 오류가 발생함.  
>> _glue_와 _eclipse_의 의미가 무엇인지, 그리고 왜 헤더에 없는지 알아보기

> 라즈베리파이에서 이미지를 받아오기  
> ros를 통해 이미지를 송수신, 비디오를 송수신하는 방법 찾아보기
>> 참고 블로그에서는 WebCam을 사용함.  
>> 혹은 myo를 사용했으나 이는 조금 더 고려할 필요가 있음