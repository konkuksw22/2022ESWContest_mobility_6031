# Alltime.sh  final
#!/bin/bash

IP=203.252.164.24
ChatPort=8608

# Path

# web server streaming
gnome-terminal --working-directory=/home/pi/wandlab-cv-streamer/ -- python3 wandlab.cv.streamer.py
echo "--------------------------------"
echo "Web Server is Streaming"
echo "--------------------------------"
# Perspective Transform - A pillar
gnome-terminal --working-directory=/home/pi/perspective/ -- sudo python3 t.py 1
echo "--------------------------------"
echo "A pillar Transform is playing"
echo "--------------------------------"
# Token Server & Main HW
python3 /home/pi/accelerometer.py $ChatPort
# NeoPixel OFF (Temp)
python3 /home/pi/testneopixel.py