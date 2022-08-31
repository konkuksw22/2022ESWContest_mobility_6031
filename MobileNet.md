TF Lite Object Detection Models on the RPi4 with Coral Accelerator
---
github: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md#section-1---how-to-set-up-and-run-tensorflow-lite-object-detection-models-on-the-raspberry-pi
<br><br>
### 1. without coral accelerator
~~~
sudo apt-get update
sudo apt-get dist-upgrade
~~~
~~~
git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git
~~~
~~~
mv TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi tflite1
cd tflite1
~~~
~~~
sudo pip3 install virtualenv
~~~
~~~
python3 -m venv tflite1-env
~~~
~~~
source tflite1-env/bin/activate
~~~
~~~
bash get_pi_requirements.sh
~~~
~~~
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
~~~
~~~
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model
~~~
~~~
python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model
~~~

### 2. with coral accelerator
#### set up
~~~
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
~~~
~~~
sudo apt-get install libedgetpu1-max
~~~
<br>
1. coral.ai: https://coral.ai/docs/accelerator/get-started/

~~~
sudo apt-get install python3-pycoral
~~~
~~~
mkdir google-coral && cd google-coral
git clone https://github.com/google-coral/examples-camera --depth 1
~~~
~~~
cd examples-camera
sh download_models.sh
~~~
~~~
cd raspicam
bash install_requirements.sh
~~~
~~~
python3 classify_capture.py
~~~
-> default는 `mobilenet_v2_1.0_224_quant_edgetpu.tflite`
<br>-> 이게 첫번째 방법

 <br><br>
2. [github](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md#section-2---run-edge-tpu-object-detection-models-on-the-raspberry-pi-using-the-coral-usb-accelerator "깃허브")
~~~
cd /home/pi/tflite1
source tflite1-env/bin/activate
~~~
~~~
wget https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
mv mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite Sample_TFLite_model/edgetpu.tflite
~~~
~~~
python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model --edgetpu
~~~
