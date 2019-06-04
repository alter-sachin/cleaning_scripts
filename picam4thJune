import face_recognition
import cv2
import numpy as np
import os
from threading import Thread
import pickle
from websocket import create_connection
import time

import cv2
import imagezmq
from queue import Queue

from queue import *
import collections
from datetime import timedelta

q_count = 0
detect_q = collections.deque(maxlen = 2)

detect_time = collections.deque(maxlen=2)

count = 0
WS ='ws://118.185.61.235:8090'
now_frame = ''


def on_message(message):
    print(len(str(message)))
    # pipe.stdin.write(message)


def on_error(error):
    print(error)


def on_close():
    print("### closed ###")


def on_open():
    def run(*args):
        for i in range(3):
            time.sleep(1)
            ws.send("Hello %d" % i)
        time.sleep(1)
        ws.close()
        print("thread terminating...")
    thread.start_new_thread(run, ())


def checkLock(lst): 
    if len(detect_q) == 2 :
        ele = lst[0]
        chk = True
        for item in lst:
            if ele !=item:
                chk = False
                break;
        if (chk == True):
            print("Equal")
            return True
        else:
            print("Not Equal")
            return False
    else:
        return False
        # Comparing each element with first item  .....if one at a time....

def threadFrameGet(threadname,q):
    # Initialize some variables
    #create_connection(WS)   
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    global count
    global detect_time
    #try:
    while True:
        # Grab a single frame of video
        count = count + 1
        print(time.ctime())
        #difference = timeDelta(detect_time[0],detect_time[1])
        #print(difference)
        #print(count)
        #ret, frame = video_capture.read()
        rpi_name, image = image_hub.recv_image()
        print(rpi_name)
        #cv2.imshow(rpi_name, image) # 1 window for each RPi
        #cv2.waitKey(1)
        #img = cv2.imread(image,0)
        #print(image)
        #global now_frame 
        #now_frame = image
        frame = image
        #cv2.imwrite("hello.png",image)
        image_hub.send_reply(b'OK')
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        #print("printing frame")
        #print(rgb_small_frame)
        # Only process every other frame of video to save time
        if process_this_frame:
            print("pushing frame")
            q.put(rgb_small_frame)
            #time.sleep(1/10000)
        #    person_name = recognise_person(rgb_small_frame)
        #    print(person_name)
            #call the other function here..... that contains the below lines.
        process_this_frame = not process_this_frame
            #return rgb_small_frame
    #except KeyboardInterrupt:
    #    print("GoodBye")
    #finally:
        # Release handle to the webcam
    #    video_capture.release()
    #    cv2.destroyAllWindows()


def recognise_person(threadname,q):  #
    #access names that are keys
    #global now_frame
    global q_count
    global detect_q
    while True:
        print("inside REKO")
        frame = q.get()
        if frame is None: continue
        #print(frame)
        rgb_small_frame = frame
        total_face_names = list(all_face_encodings.keys())
        # access values of encodings as a numpy array.
        total_face_encodings = np.array(list(all_face_encodings.values()))
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame,number_of_times_to_upsample=1,model="cnn")
        print(face_locations)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        #print(face_encodings)
        #global count
        #count = count + 1
        #print(count)
        face_names = []
        for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(total_face_encodings, face_encoding,tolerance = 0.55)
            #print(matches)
            name = "Unknown"

            #use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(total_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = total_face_names[best_match_index]
                lock_status = False
                ####################
                if(q_count < 2):
                    detect_q.append(name)
                    print(detect_q)
                    print(str(q_count) + "inside if")
                    q_count = q_count+1
                else:
                    detect_q.append(name)
                    print(detect_q)
                    print(str(q_count) + "inside else")
                    lock_status = checkLock(detect_q)
                    print(lock_status)
                    print(type(lock_status))
                    if lock_status:
                        socket_test('UnLock')
                        
                        time.sleep(3) ## time for which you want the door to be unlocked
                        print(name)
                        detect_q.clear()
                        #detect_q.clear()
                        socket_test('Lock')
                        break                
            else:
                #return None
                socket_test('Lock')
                continue



def socket_test(send="Lock"):
    global WS
    print("reached here")
    ws = create_connection(WS)
    if ws:
        print("will unlock now")
        ws.send(send)


def writeFrame(frame,name):

    #write text onto the image and display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        print(name)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)

        #Display the resulting image
        cv2.imwrite('test1/Video'+str(count)+'.png', frame)


if __name__ =="__main__":
    #pick up the face encodings saved in the pickle file, which is saved as a dictionary.
    #socket_test('Lock')
    
    print("loading saved encodings")
    with open('saved_encoding/dataset.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)

    #video_capture = cv2.VideoCapture('http://118.185.61.234:8000/stream.mjpg')
    
    #video_capture = cv2.VideoCapture('rtmp://35.212.176.30:1935/myapp/example-stream')
    image_hub = imagezmq.ImageHub()
    queue = Queue()
    thread1 = Thread(target=threadFrameGet, args=("Thread-1",queue))
    thread2 = Thread(target=recognise_person,args=("Thread-2",queue))
    #thread3 = Thread(target=checkLock,args=("Thread-3"),queue)
    thread1.start()
    thread2.start()
    #thread3.start()
    #thread1.join()
    #thread2.join()
    # Get a reference to picam  ## also try using rtsp
    #process 1 should collect video frames and calculate its encoding.
    # process two must do all the matching using the frames collected
    
