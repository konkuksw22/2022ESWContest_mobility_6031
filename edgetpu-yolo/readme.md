## Introduction

In this repository we'll explore how to run a state-of-the-art object detection mode, [Yolov5](https://github.com/ultralytics/yolov5), on the [Google Coral EdgeTPU](coral.ai/). 

```
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install -y git curl gnupg

# Install PyCoral (you don't need to do this on a Coral Board)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
sudo apt-get update
sudo apt-get install -y gasket-dkms libedgetpu1-std python3-pycoral

# Get Python dependencies
sudo apt-get install -y python3 python3-pip
pip3 install --upgrade pip setuptools wheel
python3 -m pip install numpy
python3 -m pip install opencv-python-headless
python3 -m pip install tqdm pyyaml

# Clone this repository
git clone https://github.com/jveitchmichaelis/edgetpu-yolo
cd edgetpu-yolo

# Run the test script
python3 detect.py -m yolov5s-int8-96_edgetpu.tflite --bench_image

# use webcam
python3 detect.py -m yolov5s-int8-96_edgetpu.tflite --stream
```

detect.py는 이미지 창을 띄우지 않는 파일, detect1.py는 detect.py에 실시간 화면 창을 띄우고 bounding box까지 표시한 파일
