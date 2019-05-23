#!/usr/bin/env python
import cv2
import numpy as np

import os
import pickle
import time
import sys
import tensorflow as tf
from scipy import misc


from facenet.src import facenet
from facenet.src.align import detect_face
modeldir = './trained-models/20180402-114759/'
classifier_filename = './trained-models/20180402-114759/my_tida_classifier.pkl'
npy = ''
train_img = "./dataset/raw"


class UsbCamera(object):

    """ Init camera """

    def __init__(self):
        self.init_face_recognition()
        # select first video device in system
        self.cam = cv2.VideoCapture("http://118.185.61.234:8000/stream.mjpg")
        # set camera resolution
        self.w = 800
        self.h = 600
        # set crop factor
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        # load cascade file
        self.face_cascade = cv2.CascadeClassifier('face.xml')

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
        print("Read new frame", time.time())
        # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)


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
                print(predictions)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(
                    len(best_class_indices)), best_class_indices]
                # print("predictions")
                print(best_class_indices, ' with accuracy ',
                        best_class_probabilities)

                # boxing face
                cv2.rectangle(
                    frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                # print(best_class_probabilities)
                if best_class_probabilities > 0.52:

                    # plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    result_names = self.HumanNames[best_class_indices[0]]
                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        2, (0, 0, 255), thickness=2, lineType=2)
                    # print('Result Indices: ', best_class_indices[0])
                    # print(self.HumanNames)
                    # for H_i in self.HumanNames:
                    #     if self.HumanNames[best_class_indices[0]] == H_i:
                    #         result_names = self.HumanNames[best_class_indices[0]]
                    #         cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    #                     2, (0, 0, 255), thickness=2, lineType=2)
        else:
            print('Alignment Failure')

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
                with open(classifier_filename_exp, 'rb') as infile:
                    (self.model, class_names) = pickle.load(infile)

                print('Start Recognition')
                prevTime = 0

    def __del__(self):
        cv2.destroyAllWindows()
