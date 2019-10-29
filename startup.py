import os
import sys
import time


os.system("screen -d -m -S mav python MAVProxy/MAVProxy/mavproxy.py --mav10 --master=/dev/ttyUSB0 --out 127.0.0.1:14552 --out 127.0.0.1:14557 --out 127.0.0.1:14550 --out 127.0.0.1:14540 --out 10.42.0.52:14570 --out 10.42.0.80:14570 --out 10.42.0.204:14570 --baudrate 921600 --aircraft MyCopter")
time.sleep(120)
### wait till mavproxy is up.
os.system("screen -d -m -S app python3.6 main.py --connect 127.0.0.1:14550 --check_avoidance True --avoidance_distance 6 --avoidance_alt 3 --rtl_min_time 20 --continue_min_time 8 --obstacle_altitude_range 5 --float_after_time 5 --run_number 30000 --use_blob ''  ")

