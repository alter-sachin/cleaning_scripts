# cleaning_scripts

http://cvit.iiit.ac.in/projects/IMFDB/
Please download the above dataset. ..

# Collect input videos :

1. known faces + unknown faces from canteen cam :

2. known + unknown from pi : 

3. unknown faces from canteen cam:


# Creating classifiers : 
All these classifiers are saved at : tida-face-recognition/trained-models/classifier_name
we are saving the classifiers at t-2 folder....please name them to reflect the classes and yes/no for landmarkblur script run or not....
For each class run mtcnn :

1. 14 classes, landmark blur done :  experiment 1 > results folder. contains images.
call this 14_lb_done 
python facenet/src/classifier.py TRAIN dataset/14_lb_done/  trained-models/facenet/casiabollyself1.pb trained-models/facenet/my_tida_classifier.pkl-casiabollyself-14_lb_done --batch_size 1000 

python facenet/src/classifier.py CLASSIFY dataset/14_lb_done/  trained-models/facenet/casiabollyself1.pb trained-models/facenet/my_tida_classifier.pkl-casiabollyself-14_lb_done --batch_size 1000 


2. 34 classes , landmark blur not done : 
34_lb_not_done

python facenet/src/classifier.py TRAIN dataset/34_lb_not_done/  trained-models/facenet/casiabollyself1.pb trained-models/facenet/my_tida_classifier.pkl-casiabollyself-34_lb_not_done --batch_size 1000 

python facenet/src/classifier.py CLASSIFY dataset/34_lb_not_done/  trained-models/facenet/casiabollyself1.pb trained-models/facenet/my_tida_classifier.pkl-casiabollyself-34_lb_not_done --batch_size 1000 

3. 34 classes, landmark blur done :
34_lb_done
python facenet/src/classifier.py TRAIN dataset/34_lb_done/  trained-models/facenet/casiabollyself1.pb trained-models/facenet/my_tida_classifier.pkl-casiabollyself-34_lb_not_done --batch_size 1000 

python facenet/src/classifier.py TRAIN dataset/34_lb_done/  trained-models/facenet/casiabollyself1.pb trained-models/facenet/my_tida_classifier.pkl-casiabollyself-34_lb_done --batch_size 1000 

4. 100 classes , landmark blur done :
100_lb_done


5. 83 done , landmark blur done : 
python facenet/src/classifier.py TRAIN dataset/200above/  trained-models/facenet/casiabollyself1.pb trained-models/facenet/my_tida_classifier.pkl-casiabollyself-83+71_lb_not_done --batch_size 1000

python facenet/src/classifier.py CLASSIFY dataset/200above/  trained-models/facenet/casiabollyself1.pb trained-models/facenet/my_tida_classifier.pkl-casiabollyself-83+71_lb_not_done --batch_size 1000 


python facenet/src/classifier.py CLASSIFY dataset/200above/this trained-models/facenet/casiabollyself1.pb trained-models/facenet/my_tida_classifier.pkl-casiabollyself-83_lb_not_done --batch_size 1000


















raspivid -o - -t 9999999 -w 1280 -h 720 --hflip | cvlc -vvv stream:///dev/stdin --sout '#rtp{sdp=rtsp://192.168.7.96:8800/}' :demux=h264
