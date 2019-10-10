# First import the library
from __future__ import division, print_function
#import pyrealsense2 as rs
#import numpy as np
#from guided import *
#from vehicle import Vehicle
#import defines
from threading import Thread
import numpy as np
#from scipy import ndimage
import time
import cv2
import socketio
import base64
import os
import gc

import time
import sys

PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_DIR = os.path.join(PY_DIR, '_install/lib')
for root, dirs, files in os.walk(LIB_DIR):
  if files:
    sys.path.append(root)


import mynteye_py as mynteye
import subprocess


import mynteye_py as mynteye


# pylint: disable=import-error,wrong-import-position
#import mynteye_py as mynteye
# glog_init = mynteye.glog_init.create(sys.argv)
#sys.path.insert(1,'/tfmini')
#from tfmini import TFmini

#np.set_printoptions(threshold=sys.maxsize)

class Point_distance:

    def __init__(self):
        #self.vehicle = vehicle
        #self.sio = socketio.Client()
        return None

        #self.vehicle = vehicle

    def create_thread(self):
        #Thread(target=self.calculate_min_distance_decimation).start()
        #if(defines.blob_use==True):
        #Thread(target=self.calculate_min_distance_median_blob).start()
        #else:

        #Thread(target=self.calculate_min_distance_decimation).start()
        #Thread(target=self.image_send).start()
        #Thread(target=self.get_tfmini_values).start()
        Thread(target=self.mynt_min_distance).start()
        return self
        #print("helo")
    


    def mynt_min_distance(self):
    	api = mynteye.API.create(sys.argv)
    	if not api:
    		sys.exit(1)  ###see what needs to be done

    	api.set_option_value(mynteye.IR_CONTROL,160)
    	api.set_disparity_computing_method(mynteye.SGBM)
    	api.enable_stream_data(mynteye.DISPARITY_NORMALIZED)
    	api.enable_stream_data(mynteye.Stream.DEPTH)

    	api.enable_motion_datas()

    	api.start(mynteye.ALL)

    	fps = 0
    	while True:
    		t = cv2.getTickCount()
    		api.wait_for_streams()
    		left_data = api.get_stream_data(mynteye.LEFT)
    		right_data = api.get_stream_data(mynteye.RIGHT)
    		left_data_rectified = api.get_stream_data(mynteye.LEFT_RECTIFIED)
    		right_data_rectified = api.get_stream_data(mynteye.RIGHT_RECTIFIED)

    		motion_datas = api.get_motion_datas()

    		left_data_frame = left_data.frame
    		right_data_frame = right_data.frame
    		left_data_rectified_frame = left_data_rectified.frame
    		right_data_rectified_frame = right_data_rectified.frame
    		if motion_datas:
    			imu = motion_datas[0].imu
    		img = np.hstack((left_data_frame, right_data_frame))
    		#img2 = np.hstack((left_data_rectified_frame, right_data_rectified_frame))
    		cv2.imshow('frame',img)
    		
    		print(type(img))

    		disp_data = api.get_stream_data(mynteye.DISPARITY_NORMALIZED)

    		disp_data_frame = disp_data.frame
    		if(disp_data.img.frame_id > 0):
    			print("here")
    			cv2.imshow('disparity',disp_data_frame)

    		depth_data = api.get_stream_data(mynteye.Stream.DEPTH)
    		depth_data_frame = depth_data.frame


    		####filtering########
    		#print("hellomaaa")
    		#array1 = type(left_data_rectified_frame)
    		#print(array1)
    		#binary_mask  = self.main_filter(left_data_rectified_frame)
    		#print("binary mask")
    		#print(binary_mask)
    		#depth_data_frame = np.bitwise_and(depth_data_frame,binary_mask)
    		########
    		
    		
    		if(depth_data.img.frame_id > 0 and left_data_rectified.img.frame_id>0):
    			for i in range(64):
    				left_data_rectified_frame[:,i] = 0
    			cv2.imshow("frame3",left_data_rectified_frame)
    			binary_mask  = self.main_filter(left_data_rectified_frame)
    			binary_colorized = cv2.applyColorMap(cv2.convertScaleAbs(binary_mask, alpha=0.04), cv2.COLORMAP_JET)
    			cv2.imshow("frame4",binary_colorized)
    			result = depth_data_frame.copy()
    			result[binary_mask==0] = 0
	    		print("binary mask")
	    		print("")
    			print("final frontier")
    			print((depth_data_frame.shape))
    			#depth_data_frame = np.bitwise_and(depth_data_frame,binary_mask)
    			threshold = 500
    			super_threshold_indices = depth_data_frame < threshold
    			print(super_threshold_indices)
    			depth_data_frame[super_threshold_indices] = 0
    			out = depth_data_frame[depth_data_frame!=0]
    			print("out")
    			print(out)

    			ignore_zeros_colorized = cv2.applyColorMap(cv2.convertScaleAbs(result, alpha=0.01), cv2.COLORMAP_JET)
    			#binary_colorized = cv2.applyColorMap(cv2.convertScaleAbs(binary_mask, alpha=0.04), cv2.COLORMAP_JET)
    			try:
    				obs_distance = min((out))*.001
    				print(obs_distance)
    			except Exception as e:
    				print(e)
    			cv2.imshow('depth', ignore_zeros_colorized)
    			#cv2.imshow('mask',binary_colorized)
    		t = cv2.getTickCount() - t
    		fps = cv2.getTickFrequency() / t
    		#print("fps:",fps)
    		key = cv2.waitKey(10) & 0xFF
    		if key == 27 or key == ord('q'):
    			break
    	api.stop(mynteye.ALL)
    	cv2.destroyAllWindows()



    def get_mynt_data(self):
        process = subprocess.Popen(['./get_depth_with_region'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
        process.stdin.write(b'0')
        #process.communicate()[0]
        process.stdin.close()

        while True:
            output = process.stdout.readline()
            if output=='' and process.poll() is not None:
                #print(output)
                break
            if output:
                timestamped_obs_distance = [None]*2

                obs_distance = (int(output.strip().decode("utf-8")))*.001
                print(obs_distance)
                timestamped_obs_distance[0] = obs_distance
                timestamped_obs_distance[1] = time.time()
                print("current MYNT obstacle distance is"+str(obs_distance))
                defines.point_distance_decimated.put(timestamped_obs_distance)



    def get_tfmini_values(self):
        #tf = TFmi
        mini=['RF@Right','RF@Back','RF@Left','RF@Down']
        #for num in range(0,500):
        while True :
        	
            print('============') 
            try :  
              tf1= TFmini('/dev/Right', mode=TFmini.STD_MODE)
              d1 = tf1.read()
            except Exception as e:
              d1 = None
              print("for RF@Right"+str(e))
            try :  
              tf2= TFmini('/dev/Back', mode=TFmini.STD_MODE)
              d2 = tf2.read()
            except Exception as e:
              d2 = None
              print("for RF@Back"+str(e))
            try :  
              tf3= TFmini('/dev/Left', mode=TFmini.STD_MODE)
              d3 = tf3.read()
            except Exception as e:
              d3 = None
              print("for RF@Left"+str(e))
            try :  
              tf4= TFmini('/dev/Down', mode=TFmini.STD_MODE)
              d4 = tf4.read()
            except Exception as e:
              d4 = None
              print("for RF@Down"+str(e))
            #d1 = tf.read()
            
            #d2 = tf1.read()
            #d3 = tf2.read()
            #d4 = tf3.read()
            d=[d1,d2,d3,d4]      #d3 and d4 will included
            for index in range(len(d)):
                if (d[index]):
                           print('measured {} : {:5} '.format(mini[index],d[index]))
                           f1 = mini[index]
                           f2 = d[index]
                           sio.emit('Distance',{'max_distance': 11,'RangeFinder': f1,'distance': f2 })
                else:
                  print('No valid response')


    def main_filter(self,frame):
        source = frame
        edge_mask = self.filter_edge(source)
        #print("Edge maskxxxx")
        #print(edge_mask.width)

        harris_mask = self.filter_corner(source)
        print("harriswidth")

        #print(harris_mask.height)
        combined_mask = np.bitwise_or(edge_mask,harris_mask)

        #kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3,3))
        #combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        return combined_mask


    def filter_edge(self,frame):
        source = frame
        #skinCrCbHist = np.zeros((256, 256, 1), dtype = "uint16")
        source = np.int16(source)

        scharr_x = cv2.Scharr(source,-1,1, 0)
        abs_scharr_x = cv2.convertScaleAbs(scharr_x)

        #cv::convertScaleAbs(area->scharr_x, area->abs_scharr_x);
        #cv::Scharr(area->decimated_ir, area->scharr_y, CV_16S, 0, 1);
        scharr_y = cv2.Scharr(source,-1,0, 1)

        abs_scharr_y = cv2.convertScaleAbs(scharr_y)
        print(abs_scharr_y)
        #cv::convertScaleAbs(area->scharr_y, area->abs_scharr_y);
        edge_mask = cv2.addWeighted(abs_scharr_y,0.5,abs_scharr_x,0.5,0)
        #cv::addWeighted(area->abs_scharr_y, 0.5, area->abs_scharr_y, 0.5, 0, area->edge_mask);

        retval,binary = cv2.threshold(edge_mask,192,255,cv2.THRESH_BINARY)

        #if(retval==True):
        return binary
        #cv::threshold(area->edge_mask, area->edge_mask, 192, 255, cv::THRESH_BINARY);


    def filter_corner(self,frame):
        source = frame
        float_ir = np.float32(source)

        corners = cv2.cornerHarris(float_ir,2,3,0.04)

        retval,binary = cv2.threshold(corners,300,255,cv2.THRESH_BINARY)
        #if(retval==True):
        harris_mask = np.uint(binary)

        return harris_mask
        #area->corners.convertTo(area->harris_mask, CV_8U);

    def calculate_min_distance_decimation(self):

        try:
            # Create a context object. This object owns the handles to all connected realsense devices
            #### pipeline settings for realsense depth camera
            
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth,848,480,rs.format.z16,30)
            #config.enable_stream(rs.stream.infrared,1)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            ## decimation filter ##
            decimation = rs.decimation_filter()
            colorizer = rs.colorizer()
            colorizer.set_option(rs.option.color_scheme, 0)# // white to black
            #decimated_depth = np.asanyarray
            pipe_profile = pipeline.start(config)
            depth_sensor = pipe_profile.get_device().first_depth_sensor()
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            print(preset_range)
            for i in range(int(preset_range.max)):
                visul_preset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
                if visul_preset == "High Accuracy":
                    depth_sensor.set_option(rs.option.visual_preset,i)
            new = []
            counter = 0
            img_counter = 0
            path = '/home/drone/drone-application/folder/'
            os.chdir(path)
            run_num = defines.run_number
            os.system("mkdir "+str(run_num))
            obs_distance = 0
            #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            while True:
                path = '/home/drone/drone-application/folder/'
                # This call waits until a new coherent set of frames is available on a device
                # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
                try:
                    frames = pipeline.wait_for_frames()
                except Exception as e:
                    print(e)
                    #send_offset_velocity(0,0,0,1)
                print("INSIDE DECIMATION MIN CALCULATE")
                counter = counter + 1

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                #ir1_frame = frames.get_infrared_frame(1)
                #ir2_frame = frames.get_infrared_frame(2)

                #ir1_image = np.asanyarray(ir1_frame.get_frame())
                #cv2.imwrite(os.path.join(path,'ir1_image'+str(img_counter)+'.jpg'),ir1_image)
                
                if not depth_frame or not color_frame:
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                #depth_data = depth_frame.as_frame().get_data()
                decimation.set_option(rs.option.filter_magnitude, 6)
                decimated_depth = decimation.process(depth_frame)
                
                new_depth = (np.asanyarray(decimated_depth.as_frame().get_data()))

                colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
                
                #frame_width = check_data.shape[0]
                #frame_height =check_data.shape[1]

                ##### height based ignore ######
                if(self.vehicle.location.global_relative_frame.alt<=1.0):
                    new_depth[239:479] = 0
                elif(self.vehicle.location.global_relative_frame.alt>1.0 and self.vehicle.location.global_relative_frame.alt<=2.0):

                    new_depth[359:479] = 0
                elif(self.vehicle.location.global_relative_frame.alt>2.0 and self.vehicle.location.global_relative_frame.alt<=3.0):
                    new_depth[419:479] =0

                #### height based ignore ends#####

                dist = []
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                #out = ndimage.median_filter(new_depth,size=3)
                before_ignore = cv2.applyColorMap(cv2.convertScaleAbs(new_depth, alpha=0.01), cv2.COLORMAP_JET)
                print("wrote frame 1")
                threshold = 500
                #new_depth1 = np.ma.masked_where(new_depth<=0.5,new_depth)
                #new_depth1 = np.where(new_depth<threshold,0,new_depth)
                super_threshold_indices = new_depth < threshold
                new_depth[super_threshold_indices] = 0
                out = new_depth[new_depth !=0]
                #print(type(new_depth1))
                #new_depth2 = np.ma.filled(new_depth1,fill_value=0)
                #choice = numpy.logical_and(np.greater(a, 3), np.less(a, 8))
                #numpy.extract(choice, a)
                #choice = np.greater(new_depth,0.5)#,np.less(a,20))
                #np.extract(choice,new_depth)

                #ignore_zeros = 
                ignore_zeros_colorized = cv2.applyColorMap(cv2.convertScaleAbs(new_depth, alpha=0.01), cv2.COLORMAP_JET)
                print("wrote frame 2")
                #write_image = np.vstack((before_ignore,ignore_zeros_colorized))

                # list of size 2 that contains data to send to waypoint.py
                timestamped_obs_distance = [None]*2

                try:
                    obs_distance = (min(out))*.001
                    strong_filter = []
                    timestamped_obs_distance[0] = obs_distance
                    timestamped_obs_distance[1] = time.time()
                    print("current DECIMATION obstacle distance is"+str(obs_distance))
                    defines.point_distance_decimated.put(timestamped_obs_distance)
                except Exception as e:
                    print(e)

                print("write cv2 frame")
                path = path + str(run_num)
                #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                ##depth_colormap_median = cv2.applyColorMap(cv2.convertScaleAbs(check_new, alpha=0.03), cv2.COLORMAP_JET)
                #depth_colormap_alternate = cv2.applyColorMap(cv2.convertScaleAbs(new, alpha=0.03), cv2.COLORMAP_JET)

                cv2.imwrite(os.path.join(path,str(img_counter)+'color'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),color_image)
                cv2.imwrite(os.path.join(path,str(img_counter)+'before'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),before_ignore)
                cv2.imwrite(os.path.join(path,str(img_counter)+'after'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),ignore_zeros_colorized)
                cv2.imwrite(os.path.join(path,str(img_counter)+'decimated'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),colorized_depth)

                #image_new = np.vstack(before_ignore,ignore_zeros_colorized)
                #cv2.imwrite(os.path.join(path,'before_after'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),image_new)
                #file = open()
                #os.chdir(path)

                #name_string = str(img_counter)+'decimated'+str(obs_distance)+'_'+str(img_counter)+'.txt'
                #with open(os.path.join(path,name_string),"w") as file:
                #    file.write(ignore_zeros) np.savetxt('test.out', x, delimiter=',')
                #    np.savetxt(file, new_depth, fmt="%1.3f")
                name_string = str(img_counter)+'decimated'+str(obs_distance)+'_'+str(img_counter)+'.txt'
                with open(os.path.join(path,name_string),"w") as file:
                #    file.write(ignore_zeros) np.savetxt('test.out', x, delimiter=',')
                    np.savetxt(file, new_depth, fmt="%1.3f")
                    #os.chdir('..')
                #file.close()
                #cv2.imwrite(os.path.join(path,'depthmedian'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),depth_colormap_alternate)
                #im = Image.fromarray(color_image)
                jpg = cv2.imencode('.jpg', color_image)[1].tostring()
                #img_send = toimage(color_image)

                scale_percent = defines.resolution_percentage # percent of original size
                width_c = int(color_image.shape[1] * scale_percent / 100)
                height_c = int(color_image.shape[0] * scale_percent / 100)
                dim = (width_c, height_c)
                # resize image
                timestamped_img_send = [None]*3
                color_resized = cv2.resize(color_image, dim, interpolation = cv2.INTER_AREA)
                retval,img_send = cv2.imencode('.jpg',color_resized)
                data_to_send = base64.b64encode(img_send)#.decode()
                #sio.emit('droneCamera',{'drone_id':100,'data':data_to_send})
                #if(img_counter%2==0):
                #sio.emit('droneCamera',{'drone_id':101,'data':data_to_send})
                timestamped_img_send[0] = time.time()
                timestamped_img_send[1] = img_counter
                timestamped_img_send[2] = data_to_send
                #if(img_counter%1==0):
                    #defines.image_pkts.put(timestamped_img_send)

                defines.alter_image.appendleft(timestamped_img_send)
                #if(img_counter ==2):
                    

                #sio.emit('droneCamera',{'drone_id':100,'data':data_to_send})
                img_counter += 1
                #time.sleep(.1)
            #exit(0)
        except Exception as e:
            print(e)
            pass




    def calculate_min_distance_median_blob(self):

        try:
            # Create a context object. This object owns the handles to all connected realsense devices
            #### pipeline settings for realsense depth camera
            
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth,848,480,rs.format.z16,30)
            config.enable_stream(rs.stream.infrared,1)
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
            decimation = rs.decimation_filter()

            colorizer = rs.colorizer()

            #decimated_depth = np.asanyarray
            pipe_profile = pipeline.start(config)
            depth_sensor = pipe_profile.get_device().first_depth_sensor()
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            print(preset_range)
            for i in range(int(preset_range.max)):
                visul_preset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
                if visul_preset == "High Accuracy":
                    depth_sensor.set_option(rs.option.visual_preset,i)
            new = []
            counter = 0
            img_counter = 0
            path = '/home/drone/drone-application/folder/'
            os.chdir(path)
            run_num = defines.run_number
            os.system("mkdir "+str(run_num))
            #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            while True:
                path = '/home/drone/drone-application/folder/'
                # This call waits until a new coherent set of frames is available on a device
                # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
                try:
                    frames = pipeline.wait_for_frames()
                except Exception as e:
                    print(e)
                    #send_offset_velocity(0,0,0,1)
                print("INSIDE MIN CALCULATE")
                counter = counter + 1
                depth_frame = frames.get_depth_frame()

                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_data = depth_frame.as_frame().get_data()

                check_data = np.asanyarray(depth_data)
                
                frame_width = check_data.shape[0]
                frame_height =check_data.shape[1]

                ###HEIGHT BASED IGNORED points DISTANCES
                if(self.vehicle.location.global_relative_frame.alt<=1.0):
                    check_data[239:479] = 0    
                elif(self.vehicle.location.global_relative_frame.alt>1.0 and self.vehicle.location.global_relative_frame.alt<=2.0):
                    check_data[359:479] = 0    
                elif(self.vehicle.location.global_relative_frame.alt>2.0 and self.vehicle.location.global_relative_frame.alt<=3.0):
                    check_data[419:479] = 0
                ###HEIGHT BASED IGNORED points DISTANCES ends

                dist = []
                width = depth_frame.get_width()
                height = depth_frame.get_height()


                median_out = ndimage.median_filter(check_data,size=3)

                ####use depth map to find blobs if any####
                depth_colormap_median = cv2.applyColorMap(cv2.convertScaleAbs(median_out, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(path,'median_out'+'_'+str(img_counter)+'.jpg'),depth_colormap_median)

                #####blob detection #####
                ### prepare image for blob detection####
                thresh = 127
                im_bw = cv2.threshold(depth_colormap_median,thresh,255,cv2.THRESH_BINARY)[1]
                inverted_image = cv2.bitwise_not(im_bw)
                
                params = cv2.SimpleBlobDetector_Params()

                ### Change threshods settings for blob detection
                params.minThreshold = 10
                params.maxThreshold = 2000

                # Filter by Area.
                params.filterByArea = True
                params.minArea = 1.0 * 1.0 * 1.0
                params.maxArea = 3.14159 * 6.0 * 6.0
                params.minDistBetweenBlobs = 1
                # enabling these can cause us to miss points
                params.filterByCircularity = False
                params.filterByConvexity = False
                params.filterByInertia = False

                # Create a detector with the parameters
                detector = cv2.SimpleBlobDetector_create(params)

                # Detect blobs.
                keypoints = detector.detect(inverted_image)
                print("printing keypoints")
                #print(keypoints)
                #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                # Draw detected blobs as red circles.
                # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
                # the size of the circle corresponds to the size of blob

                #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), #(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                #img2 = im.copy()
                blob_write_check = False
                img2 = depth_colormap_median.copy()
                for marker in keypoints:
                    #print(marker)
                    blob_size = marker.size
                    blob_x = marker.pt[0]
                    blob_y = marker.pt[1]
                    #print("SIZE OF BLOBSSSSSSSS ISSSSSSSSS")
                    #print(blob_size)
                    #print(blob_x)
                    #print(blob_y)
                    #img2 = cv2.drawMarker(img2,tuple(int(i) for i in marker.pt),color=(0,0,0))
                for x in range(0,len(keypoints)):
                    #print("inside fill")
                    ceiling = math.ceil(blob_size/2)
                    left_x = (np.int(keypoints[x].pt[0])-ceiling-4)
                    right_x = (np.int(keypoints[x].pt[0])+ceiling+4)
                    top_y = (np.int(keypoints[x].pt[1])-ceiling-4)
                    bottom_y = (np.int(keypoints[x].pt[1])+ceiling+4)
                    #median_out[left_x:right_x,top_y:bottom_y] = 0
                    median_out[top_y:bottom_y,left_x:right_x] = 0
                    #print(median_out[left_x:right_x,top_y:bottom_y])
                    depth_colormap_median_blob = cv2.applyColorMap(cv2.convertScaleAbs(median_out, alpha=0.03), cv2.COLORMAP_JET)
                    blob_write_check = True
                    #if(blob_size<100.0):
                    img2_blob = cv2.circle(img2, (np.int(keypoints[x].pt[0]),np.int(keypoints[x].pt[1])), radius=np.int(keypoints[x].size), color=(0), thickness=10)
                    both = np.vstack((img2_blob,depth_colormap_median_blob))
                    #all_frame = np.vstack((both,color_image))
                #####blob ends#####
                
                ignore_zeros = median_out[median_out !=0 ]
                timestamped_obs_distance = [None]*2  # list of size 2


                try:
                    obs_distance = (min(ignore_zeros))*.001
                    
                    
                    strong_filter = []

                    timestamped_obs_distance[0] = obs_distance

                    timestamped_obs_distance[1] = time.time()
                    print("current obstacle distance is"+str(obs_distance))

                    defines.point_distance_median_blob.put(timestamped_obs_distance)
                except Exception as e:
                    print(e)
                #for y in range(0, int(height), 2):
                #    for x in range(0, int(width), 2):
                        #dist = depth.get_distance(x, y)
                #        check = depth.get_distance(x, y)
                #        if(check > 1):
                #            dist.append(check)
                        # dist.append(check)
                #print(min(dist))
                #defines.point_distance.put(min(dist))
                #images = np.hstack((color_image,depth_image))
                #data_to_send = base64.encode(color_image)
                print("write cv2 frame")
                path = path + str(run_num)
                if(blob_write_check==True):
                    cv2.imwrite(os.path.join(path,'blob_filled_depthmedian'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),both)
                

                #cv2.imwrite(os.path.join(path,'color'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),color_image)
                #cv2.imwrite(os.path.join(path,'depth'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),depth_colormap)
                #cv2.imwrite(os.path.join(path,'blob_filled_depthmedian'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),img2)
                #cv2.imwrite(os.path.join(path,'decimated'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),colorized_depth)
                #cv2.imwrite(os.path.join(path,'depthmedian'+str(obs_distance)+'_'+str(img_counter)+'.jpg'),depth_colormap_alternate)
                #im = Image.fromarray(color_image)
                #jpg = cv2.imencode('.jpg', color_image)[1].tostring()
                #img_send = toimage(color_image)
                #retval,img_send = cv2.imencode('.jpg',color_image)
                #data_to_send = base64.b64encode(img_send).decode()
                #defines.image_pkts.put(data_to_send)
                #sio.emit('droneCamera',{'drone_id':100,'data':data_to_send})
                img_counter += 1
                #time.sleep(.1)
            #exit(0)
        except Exception as e:
            print(e)
            pass


    def set_distance_sensor(self, vehicle, distance):
        pwm_value_int = int(pwm_value)
        msg = vehicle.message_factory.command_long_encode(
            0, 0,  # target_system, target_component
            mavutil.mavlink.DISTANCE_SENSOR,  # command
            0,  # Unused parameter
            # param 2,Minimum distance the sensor can measure in cm (realsense can measure min .3m)
            30,
            600,  # param 3, Maximum distance sensor can measure 6m for realsense
            distance,  # current distance reading
            0,  # type of distance sensor
            0,  # Onboard ID of the sensor
            ROTATION_NONE,  # Onboard ID of the sensor
            255  # covariance 255 if unknown
        )
        vehicle.send_mavlink(msg)

    def set_distance_sensor2(self):
        vehicle = self.vehicle
        while True:
            current_distance = defines.point_distance.get()
            print("current distance "+str(current_distance))
            #print(current_distance)
            #pwm_value_int = int(pwm_value)
            msg = self.vehicle.message_factory.command_long_encode(
                0, 0,  # target_system, target_component
                132,#mavutil.mavlink.MAV_CMD_DISTANCE_SENSOR,  # command
                0,  # Unused parameter
                # param 2,Minimum distance the sensor can measure in cm (realsense can measure min .3m)
                30,
                600,  # param 3, Maximum distance sensor can measure 6m for realsense
                current_distance,  # current distance reading
                0,  # type of distance sensor
                0,  # Onboard ID of the sensor
                0,  # MAV_SENSOR_ROTATION_NONE,  #
                255  # covariance 255 if unknown
            )
            #vehicle.send_mavlink(msg)
            if((current_distance < 1) and (vehicle.location.global_relative_frame.alt>5)):
                send_offset_velocity(vehicle, 0, 0, 0, 1)

    def image_send(self):
        counter = 1
        while True:
            #print("sending imaaagesss")
            #if(counter%1==0):
            #print("time image receive is")
            #time_image_recv = time.time()
            #print(time_image_recv)
            #image_received = defines.image_pkts.get()
            
            if defines.alter_image:
                alter_image_received = defines.alter_image.pop()
                time_of_push = alter_image_received[0]
                print(time_of_push)
                print("time of push"+str(time_of_push))
                pkt_number = alter_image_received[1]
                print("pkt number"+str(pkt_number))
                image_to_send = alter_image_received[2]
                #print(image_to_send)
                current_time = time.time()
                print("current_time"+str(current_time))
                sio.emit('droneCamera',{'drone_id':100,'data':image_to_send,'packet_num':pkt_number,'date_time':time_of_push})
            #print(counter)
            counter+=1


pobj = Point_distance()
pobj.create_thread()
