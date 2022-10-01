# Team JJIDLE

## 팀원

| 이름   | 메일               | 역할 |
| ------ | ------------------ | ------ |
| 정우진 | woo9904@konkuk.ac.kr | 기획, 총괄 및 개발<br/>우회전 인식 시스템 개발 <br/>object detecting 알고리즘 개발|
| 박승철 | psc0526@konkuk.ac.kr | A필러 부분 시점변환 개발<br/>서버 환경 구축 & 서버 코드 개발<br/>이미지 전처리 개발 |
| 신지혜 | long0404@konkuk.ac.kr | H/W개발 – 라즈베리파이 & 서버간 통신 시스템 개발<br/>언어관련 딥러닝 클라이언트 개발<br/>센서 및 소리 및 LED제어 |
| 이서연 | seoyeon8167@konkuk.ac.kr | H/W개발 – 라즈베리파이 & 서버간 통신 시스템 개발<br/>model 검색 및 알고리즘 최적화 <br/>다양한 모드 연동|

## <div align="center">Quick Start Examples</div>

<details open>
<summary>1. Server Computer Install</summary>

객체 검출의 bashline 코드는 이 [Yolov5](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) 코드를 참고하였다.    
Environment 또한 동일하게 
[**Python>=3.7.0**](https://www.python.org/) 환경에, 
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/)가 필요하다.

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
</details>

<details open>
<summary>2. Raspberry Pi Install</summary>

본 GitHub의 코드를 clone하여 "raspberryPi4_source" directory의 코드를 설치한다. 

```bash
git clone https://github.com/konkuksw22/esw22_JJIDLE5  # clone
cd raspberryPi4_source
```
</details>

## github Tree

```bash
│  README.md
│  
├─camera_cal
│      img.png
```



## 주의사항

> 실행은 
>
> - server : CUDA 필요
> - [YOLOV5](https://github.com/ultralytics/yolov5) 설치 필요 
> - raspberry pi, server pip install -r requirments.txt



## Kaldi

### 학습데이터

#### 1. AM

> KSponSpeech (1000 hrs) + OpenSLR (51.6 hrs)의 음향 데이터를 이용

![](./imgs/graph_AM.png)

##### 1-1. Sox Pipeline

> ```/opt/zeroth/s5/local/nnet3/multi_condition/run_ivector_common.sh.md``` 참조



#### 2. LM

![](./imgs/graph_LM.png)

> 모두의 말뭉치, AI HUB의 말뭉치 및 Web Scrapping을 이용하여 말뭉치를 구성하였으며
>
> 2.5억개의 문장으로 제작하였다.	

##### 2-1. ARPA 제작

> ```/opt/zeroth/s5/data/local/lm/buildLM/run_task.sh``` 실행

##### 2-2. Pruning

> ```/opt/zeroth/s5/data/local/lm/buildLM/_scripts_/buildNGRAM.sh``` 참조



### 사용방법

- 학습방법

  > ```/opt/zeroth/s5/run_kspon.sh``` 실행  및 markdown file 참조
  >
  > - 불필요한 파일 제거를 위해 ``` /opt/zeroth/s5/utils/remove_data.sh``` ```실행
  >
  > - **실행 전 Stage 변수 확인**

- 학습 파일 추출 방법

  > ```/opt/zeroth/s5/local/export.sh``` 참조

- 단어 추가 방법

  > ``` /opt/zeroth/s5/kusw_extend_vocab_demo.sh.md``` 참조
  >
  > ❗ **주의사항**❗
  >
  > 기존에 학습시켜놓은 Morfessor Model을 이용하여 새로운 단어를 추출하는 과정이 있기때문에, 형태소 분리과정에서 단어가 거의 중복되어 사라질 가능성이 높으니, 추가를 원하는 단어는 ```/opt/zeroth/s5/data/local/lm/buildLM/_scripts_/gen_Pronounciate.py``` 로 제작
  >
  > > **발음 생성의 원리는 다음과 같음**
  > >
  > > ```
  > > CHOSUNG_LIST =  [u'ㄱ', u'ㄲ', u'ㄴ', u'ㄷ', u'ㄸ', u'ㄹ', u'ㅁ', u'ㅂ', u'ㅃ', u'ㅅ',\
  > >                  u'ㅆ', u'ㅇ', u'ㅈ', u'ㅉ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ']
  > > JUNGSUNG_LIST = [u'ㅏ', u'ㅐ', u'ㅑ', u'ㅒ', u'ㅓ', u'ㅔ', u'ㅕ', u'ㅖ', u'ㅗ', u'ㅘ',\
  > >                  u'ㅙ', u'ㅚ', u'ㅛ', u'ㅜ', u'ㅝ', u'ㅞ', u'ㅟ', u'ㅠ', u'ㅡ', u'ㅢ', u'ㅣ']
  > > JONGSUNG_LIST = [u'_', u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ', u'ㄹ', u'ㄺ',\
  > >                  u'ㄻ', u'ㄼ', u'ㄽ', u'ㄾ', u'ㄿ', u'ㅀ', u'ㅁ', u'ㅂ', u'ㅄ', u'ㅅ',\
  > >                  u'ㅆ', u'ㅇ', u'ㅈ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ']
  > > 
  > > CHOSUNG_SYM =  [u'g', u'gg', u'n', u'd', u'dd', u'l', u'm', u'b', u'bb', u's',\
  > >                 u'ss', u'', u'j', u'jj', u'ch', u'kh', u't', u'p', u'h']
  > > JUNGSUNG_SYM = [u'a', u'ae', u'ya', u'yae', u'eo', u'e', u'yeo', u'ye', u'o', u'wa',\
  > >                 u'wae', u'oe', u'yo', u'u', u'wo', u'we', u'wi', u'yu', u'eu', u'ui', u'i']
  > > JONGSUNG_SYM = [u'', u'g2', u'', u'', u'n2', u'', u'', u'd2', u'l2', u'',\
  > >                 u'', u'', u'', u'', u'', u'', u'm2', u'b2', u'', u'',\
  > >                 u'', u'ng', u'', u'', u'', u'', u'', u'']
  > > 
  > > CHOSUNG_SYM_IPA =  [u'k', u'k͈', u'n', u't', u't͈', u'ɾ', u'm', u'p', u'p͈', u'sʰ',\
  > >                     u's͈', u'', u't͡ɕ', u't͡ɕ͈', u't͡ɕʰ', u'kʰ', u'tʰ', u'pʰ', u'h']
  > > JUNGSUNG_SYM_IPA = [u'a', u'ɛ', u'ja̠', u'jɛ̝', u'ʌ̹', u'e', u'jʌ', u'je', u'o', u'wa',\
  > >                     u'wɛ̝', u'we', u'jo', u'u', u'wʌ', u'we', u'y', u'ju', u'ɯ', u'ɰi', u'i']
  > > JONGSUNG_SYM_IPA = [u'', u'k̚', u'', u'', u'n', u'', u'', u't̚', u'ɭ', u'',\
  > >                     u'', u'', u'', u'', u'', u'', u'm', u'p̚', u'', u'',\
  > >                     u'', u'ŋ', u'', u'', u'', u'', u'', u'']
  > > ```
  > >
  > > [출처] https://github.com/goodatlas/zeroth/tree/master/s5/data/local/lm

✔ **이외의 파일 사용 방법은 각 마크다운(.md) 파일 참조**



## 서버 구성도

![](./imgs/TCP.png)

> 서버는 main server, TTS & Sentimental server, Spacing client, Chatting client, TTS receive client, Voice send client 여섯 개의 프로그램으로 구성되어있다. 연결 구조는 첨부한 사진과 같으며, 동작을 위해서는 아래와 같은 절차를 진행한다.


> cc파일의 경우 사용을 위해서는 make를 통해 build하는 과정이 필요하다.


```sh
telnet ${서버주소}
```

> 1. telnet으로 서버에 접속하고 ID와 Passwd를 입력한다

```sh
cd /opt
sudo sh allStart.sh
```

> 2. /opt 디렉토리로 이동해서 allStart.sh 파일을 실행시키고 connect success가 뜰 때까지 기다린다

```sh
sh start.sh
```

> 3. ctrl + alt+ T를 이용해 새 터미널 창을 열고 start.sh파일을 실행시킨다.
> 4. 이후 로그인 창이 뜨면 로그인한 뒤 프로그램을 사용한다.


## 딥러닝 클라이언트 구조

### E2E Text To Speech Model

#### 1. Text2Mel

##### 1-1. Tacotron2

> https://github.com/NVIDIA/tacotron2의 Model 사용

![](./imgs/Tacotron2.PNG)

> Text2Mel을 진행해주는 Seq2Seq모델의 구조를 기반으로 하는 Neural Network이다. 문자 임베딩을 Mel-Spectrogram에 맵핑하는 반복적인 구조로 이루어져있으며, Encoder와 
> Decoder를 연결하는 Location-Sensitive Attention이 있다. 이때 Decoder에서 AutoRegressive 
> RNN을 포함하는데 이와 같은 이유로, 추론 속도가 떨어지는 특징을 가진다.



##### 1-2. GST

![](./imgs/GST.PNG)

> 다양한 길이의 오디오 입력의 운율을 Reference Encoder(참조 인코더)를 통해 고정된 길이의 
> 벡터로 변환한다. 이후 학습 과정에서 Tacotron 인코더단의 출력과 concatenate하여 Tacotron 
> 모델의 Attention의 입력으로 사용하여 목표 문장의 Style을 Embedding을 한다. 이후의 Style Embedding Vector는 Text2Mel 모델을 사용할 때 임베딩 된 Character와 함께 
> add 또는 concatenate하여 Style에 맞는 Mel-spectrogram을 제작하는데 사용된다.



##### 1-3. FastSpeech

![](./imgs/etts_graph.png)

> FastSpeech 모델은 Text2Mel 작업을 위한 Neural Network이며, 같은 작업을 하는 Tacotron2
> 는 Regressive한 구조를 가지는데 비해 FastSpeech는 Non-autoRegressive한 구조를 가져 훨씬 
> 더 빠른 inference가 가능해 해당 모델을 사용하기로 하였다. 전체 구조는 위의 그림과 같으며, Tacotron2-GST를 통해 학습하였던 Style Vector를 가져와 
> Character Embedding Vector와 add연산을 진행하여 추론에 사용한다.



### Korean Spacing Model

![spacing model]
> 실시간으로 들어오는 문장에 대해 처리하는 만큼 정확도를 유지한 채 빠른 속도로 동작하기 위해 CNN모델을 사용했다.
> 띄어쓰기가 잘 되어있는 한국어 문장에 랜덤하게 공백을 삭제/ 추가하고 원래의 문장으로 북구하기 위한 label을 생성함으로써 학습시켰다.
> 0, 1, 2의 라벨을 갖게되며 0은 현상유지, 1은 띄어쓰기 추가, 2는 띄어쓰기 삭제를 의미한다.
> 공백 삭제의 확률은 0.15, 공백 추가의 확률은 0.5로, 형태소 단위로 띄어쓰기가 되는 kaldi의 환경에 맞추어 공백 추가 확률을 높게 조정했다.
> 학습한 모델의 성능은 GTX1080Ti SLI환경에서 아래의 표와 같았다.



|               case               | 정확도 | 추론 시간 |
| :------------------------------: | :----: | :-------: |
|     공백을 모두 제거한 문장      | 0.9442 |  0.088ms  |
| 모든 음절마다 공백을 추가한 문장 | 0.9539 |  0.088ms  |



### KoBERT Sentimental Classification

> https://github.com/SKTBrain/KoBERT의 PreTrained BERT 사용

![](./imgs/koBERT.PNG)





## Requirement

> ESPnet 0.7.0
>
> Kaldi >= 5.4
>
> Pytorch >= 1.7.0



### Computing Power

```
1. Work Station (Main)
CPU: Intel(R) Xeon(R) CPU E5-2640 v3
VGA: GTX1080Ti SLI
RAM: 32GB (+ Swap Memory 96GB)
HDD: SSD 1TB / HDD 3TB
CUDA 10.1
CUDNN 7.6.5

2. Work Station2 (Sub)
CPU: ntel(R) Xeon(R) CPU W-2223
VGA: RTX3080
RAM: 64GB (+ Swap Memory 128GB)
HDD: SSD 1TB / HDD 4TB
CUDA 11.1
CUDNN 8.0.5
```

## 채팅 UI 구성 및 사용법
![](./imgs/Login.png)

>Login Interface
>- Email, Password 입력 후 Enter or Login 버튼 클릭
>- Tab은 Tkinter에서 사용 불가능하여 Space로 대체

![](./imgs/Main_Chatting.png)
>Main Chatting Interface
>- Home 버튼을 누르면 Login 화면으로 나가진다.
>- Menu 버튼을 누르면 Menu 팝업창이 나온다.
>- 하단에는 채팅을 입력하는 Chat 인터페이스가 있다.
>- 중앙에는 채팅 로그를 띄워주는 인터페이스가 있다.

![](./imgs/Menu.png)
>Menu Interface
>- 성별을 선택하여 TTS의 성별을 결정 가능하다.
>- Scale을 선택하여 TTS의 감정 감도를 결정 가능하다.
>- Information을 확인할 수 있다.

## HW
>VVS의 하드웨어 구성
>- Raspberry Pi 4B+
>- HiFiBerry DAC+ ADC pro
>- USB SoundCard

![](./imgs/RPI_Case.png)
>3D Printer 케이스 도면  
>- 발열 문제가 심해 쿨러를 별도로 설치하기 위한 케이스를 도면을 제작함

## TODOS

- server, client 고급화
- raspberryPi/src/client_text.py 코드 최적화
- Large HCLG graph 제작
