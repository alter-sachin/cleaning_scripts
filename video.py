#!/usr/bin/env python
import cv2
import numpy as np
import pandas as pd
import os
import pickle
import time
import sys
import tensorflow as tf
from scipy import misc
from queue import *
import collections



from facenet.src import facenet
from facenet.src.align import detect_face
BASE_DIR = '/home/ubuntu/tida'
#BASE_DIR = 'home/ubuntu'  #pretrained casia

#modeldir = BASE_DIR+'models/facenet/20190412-142315' ###pretrainedcasia
modeldir = BASE_DIR + '/tida-face-recognition/trained-models/facenet/20190408-095443/' ####casiabollyself



classifier_filename = BASE_DIR + \
	'/tida-face-recognition/trained-models/facenet/my_tida_classifier.pkl-casiabollyself-10teamhall_sidesremoved'#casiabollyself


#classifier_filename = BASE_DIR + \
#	'/tida-face-recognition/trained-models/20180408-102900/my_tida_classifier.pkl-casia-10' ###pretrained

npy = ''
train_img ="/home/ubuntu/tida/tida-face-recognition/dataset/raw"
# URL = 'rtmp://35.212.198.245:1935/myapp/example-stream'
# URL = 'http://192.168.7.96:8000/stream.mjpg'
URL = 'http://118.185.61.234:8000/stream.mjpg'
#URL = 'video.avi'
q_count = 0
detect_q = collections.deque(maxlen = 3)
# WS = 'ws://192.168.7.96:8084'
WS = 'ws://118.185.61.234:8085'
# import subprocess
import asyncio
# import websockets

from websocket import create_connection


def on_message(ws, message):
	print(len(str(message)))
	# pipe.stdin.write(message)


def on_error(ws, error):
	print(error)


def on_close(ws):
	print("### closed ###")


def on_open(ws):
	def run(*args):
		for i in range(3):
			time.sleep(1)
			ws.send("Hello %d" % i)
		time.sleep(1)
		ws.close()
		print("thread terminating...")
	thread.start_new_thread(run, ())

class UsbCamera(object):

	""" Init camera """

	def __init__(self):
		self.c=0
		self.create_dict()
		self.ws = create_connection(WS)
#		self.ws = None
		#self.lst =[]
		#self.ckeckLock(lst)
		print('initializing FR')
		self.init_face_recognition()
		# select first video device in system
		print('initializing CAMERA')
	#	self.cam = cv2.VideoCapture(URL)
		# set camera resolution
		self.w = 320
		self.h = 240
		
		# set crop factor
		# self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
		# self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
		# load cascade file
		#self.face_cascade = cv2.CascadeClassifier('face.xml')
		# start_server = websockets.serve(self.socket_test, 'ws://192.168.7.96', 8084)
		# asyncio.get_event_loop().run_until_complete(start_server)
		# asyncio.get_event_loop().run_forever()
		# self.init_websocket()
		# self.socket_test()


	def set_resolution(self, new_w, new_h):
		"""
		functionality: Change camera resolution
		inputs: new_w, new_h - with and height of picture, must be int
		returns: None ore raise exception
		"""
		if isinstance(new_h, int) and isinstance(new_w, int):
			# check if args are int and correct
			if (new_w <= 800) and (new_h <= 600) and \
			   (new_w > 0) and (new_h > 0):
				self.h = new_h
				self.w = new_w
			else:
				# bad params
				raise Exception('Bad resolution')
		else:
			# bad params
			raise Exception('Not int value')

	def checkLock(self,lst): 
		#if len(detect_q) == 3 :

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
    	# Comparing each element with first item  .....if one at a time....
    

	def get_frame(self, fdenable):
		"""
		functionality: Gets frame from camera and try to find feces on it
		:return: byte array of jpeg encoded camera frame
		"""
		success, image = self.cam.read()
		if success:
			# scale image
			image = cv2.resize(image, (self.w, self.h))
			try:
				self.detect_face(image)
			except:
				print("An exception occurred")

		else:
			image = np.zeros((self.h, self.w, 3), np.uint8)
			cv2.putText(image, 'No camera', (40, 60),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
		# encoding picture to jpeg
		ret, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()

	

	def detect_face(self, frame):
		# print("Read new frame", time.time())
		# frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter('outputgreat.avi',fourcc,24,(1280,720))
		global q_count
		global detect_q
  		### check queue if queue has 3 of the same name then unlock.
		# return
		
		if self.c % 1 == 0:
			if frame.ndim == 2:
				frame = facenet.to_rgb(frame)
			frame = frame[:, :, 0:3]
			bounding_boxes, _ = detect_face.detect_face(
				frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
			nrof_faces = bounding_boxes.shape[0]
			print('Detected_FaceNum: %d' % nrof_faces)
			if nrof_faces > 0:
				det = bounding_boxes[:, 0:4]
				img_size = np.asarray(frame.shape)[0:2]
				#fourcc = cv2.VideoWriter_fourcc(*'AVI')
				#out = cv2.VideoWriter('output.avi',fourcc)
				cropped = []
				scaled = []
				scaled_reshape = []
				bb = np.zeros((nrof_faces, 4), dtype=np.int32)

				for i in range(nrof_faces):
					emb_array = np.zeros((1, self.embedding_size))

					bb[i][0] = det[i][0]
					bb[i][1] = det[i][1]
					bb[i][2] = det[i][2]
					bb[i][3] = det[i][3]

					# inner exception
					if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
						print('Face is very close!')
						continue

					cropped.append(
						frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
					cropped[i] = facenet.flip(cropped[i], False)
					scaled.append(misc.imresize(
						cropped[i], (self.image_size, self.image_size), interp='bilinear'))
					scaled[i] = cv2.resize(scaled[i], (self.input_image_size, self.input_image_size),
										interpolation=cv2.INTER_CUBIC)
					scaled[i] = facenet.prewhiten(scaled[i])
					scaled_reshape.append(
						scaled[i].reshape(-1, self.input_image_size, self.input_image_size, 3))
					feed_dict = {
						self.images_placeholder: scaled_reshape[i], self.phase_train_placeholder: False}
					emb_array[0, :] = self.sess.run(
						self.embeddings, feed_dict=feed_dict)
					predictions = self.model.predict_proba(emb_array)
					best_class_indices = np.argmax(predictions, axis=1)
					best_class_probabilities = predictions[np.arange(
						len(best_class_indices)), best_class_indices]
					# print("predictions")

					# boxing face
					cv2.rectangle(
						frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
					print('at_frame',str(self.c),self.Name_Dict[self.class_names[best_class_indices[0]]],' with accuracy ',best_class_probabilities)
					if best_class_probabilities > 0.01:
						
						# plot result idx under box
						text_x = bb[i][0]
						text_y = bb[i][3] + 20
						# result_names = self.Name_Dict[self.class_names[best_class_indices[0]]]
						result_names = self.Name_Dict[self.class_names[best_class_indices[0]]]

						lock_status = False
						####################
						if(q_count < 3):
							detect_q.append(result_names)
							print(detect_q)
							print(str(q_count) + "inside if")
							q_count = q_count+1
						else:
							detect_q.append(result_names)
							print(detect_q)
							print(str(q_count) + "inside else")
							lock_status = self.checkLock(detect_q)
							print(lock_status)
							print(type(lock_status))
							if lock_status:
								self.socket_test('UnLock')
								time.sleep(3) ## time for which you want the door to be unlocked
								print("writing true frame")
								print('at_frame',str(self.c),result_names, ' with accuracy ',
								best_class_probabilities)
								cv2.putText(frame, result_names+ "-" + str(best_class_probabilities), (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=1, lineType=1)
						
								status = cv2.imwrite("/home/ubuntu/tida/tida-2/casiabollyself-83_lb_not_done/inputvideo/yesframe"+str(self.c)+".png",frame)
								detect_q.clear()
								self.socket_test('Lock')
							break
						###################


						print('at_frame',str(self.c),result_names, ' with accuracy ',
						best_class_probabilities)
						cv2.putText(frame, result_names+ "-" + str(best_class_probabilities), (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
									1, (0, 0, 255), thickness=1, lineType=1)
						
						#out = cv2.VideoWriter('output.avi',fourcc)
						#detect_q.put(result_names)
						print("writing true frame")
						status = cv2.imwrite("/home/ubuntu/tida/tida-2/casiabollyself-83_lb_not_done/inputvideo/yesframe"+str(self.c)+".png",frame)
						#out.write(frame)
						
						#checklock(self,result_names)
						if lock_status:
							self.socket_test('UnLock')
						

						# print('Result Indices: ', best_class_indices[0])
						# print(self.HumanNames)
						# for H_i in self.HumanNames:
						#     if self.HumanNames[best_class_indices[0]] == H_i:
						#         result_names = self.HumanNames[best_class_indices[0]]
						#         cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
						#                     2, (0, 0, 255), thickness=2, lineType=2)
					
			else:
				print("writing false frame")
				#status = cv2.imwrite("/home/ubuntu/tida/tida-2/casiabollyself-10teamcctv-deploy/notframe"+str(self.c)+".png",frame)
				#out.write(frame)
				self.socket_test('Lock')
				print('Alignment Failure')
		self.c+=1
		#out.release()




	



	def init_face_recognition(self):
		with tf.Graph().as_default():
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0)
			self.sess = tf.Session(config=tf.ConfigProto(
				gpu_options=gpu_options, log_device_placement=False))
			with self.sess.as_default():
				self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(
					self.sess, npy)

				self.minsize = 20  # minimum size of face
				self.threshold = [0.8, 0.8, 0.8]  # three steps's threshold
				self.factor = 0.709  # scale factor
				self.margin = 32
				self.frame_interval = 1
				self.batch_size = 1000
				self.image_size = 160
				self.input_image_size = 160

				self.HumanNames = os.listdir(train_img)
				self.HumanNames.sort()

				print('Loading Modal')
				facenet.load_model(modeldir)
				self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
				self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
				self. phase_train_placeholder = tf.get_default_graph(
				).get_tensor_by_name("phase_train:0")
				self.embedding_size = self.embeddings.get_shape()[1]

				classifier_filename_exp = os.path.expanduser(
					classifier_filename)
				print('CLASSIFIER', classifier_filename_exp)
				with open(classifier_filename_exp, 'rb') as infile:
					(self.model, self.class_names) = pickle.load(
						infile, encoding='latin1')
					print('classNames',self.class_names)

				print('Start Recognition')
				prevTime = 0

	def create_dict(self):
		df = pd.read_csv('/home/ubuntu/tida/tida-2/dataset/dataset-emp-names.csv', index_col='login_id',usecols=['login_id', 'username'])
		result = df.to_dict(orient='dict')
		self.Name_Dict = result['username']

	def __del__(self):
		cv2.destroyAllWindows()

	# async def socket_test(websocket, path):
	#     greeting = "Hello!"		

	#     await websocket.send(greeting)
	#     print(f"> {greeting}")


	def socket_test(self,send="Lock"):
		
		if self.ws:
			# self.ws.send("UnLock")
			self.ws.send(send)
		# self.ws.close()

	# def init_websocket(self):
	#     websocket.enableTrace(True)
	#     self.ws = websocket.WebSocketApp("ws://192.168.7.96:8084",
	#                                 on_message=on_message,
	#                                 on_error=on_error,
	#                                 on_close=on_close)
	#     self.ws.on_open = on_open
	#     # self.ws.run_forever()
	#     self.ws.send('UnLock')
	#     self.ws.send('Lock')
	#     self.ws.close()

