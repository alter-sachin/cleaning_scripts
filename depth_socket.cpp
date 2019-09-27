#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mynteye/api/api.h"
#include <ctime>
#include<queue>
#include <functional>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <string>
#include <chrono>
#include <thread>
#include <stdio.h>
#include <tchar.h>

using namespace ssio;
//std::queue<byte> Q;
std::queue<std::vector<uchar> > p;

MYNTEYE_USE_NAMESPACE

int main(int argc, char *argv[]) {
  auto &&api = API::Create(0, argv);
  if (!api) return 1;

  bool ok;
  auto &&request = api->SelectStreamRequest(&ok);
  if (!ok) return 1;
  api->ConfigStreamRequest(request);

  api->SetOptionValue(Option::IR_CONTROL, 80);
  //api->SetDisparityComputingMethodType(DisparityComputingMethod::SGBM);
  api->EnableStreamData(Stream::DISPARITY_NORMALIZED);
  api->EnableStreamData(Stream::DEPTH);

  api->Start(Source::VIDEO_STREAMING);


  // connect with socket using query
  std::map<std::string, std::string> query;
  query["access_token"] = "1";
  query["device_type"] = "1";
  h.connect("wss://gps-tracker-7022.jungleworks.com/",query);


  std::thread t1(send_image);

  //cv::namedWind
  int count = 0;
  while (true) {
    api->WaitForStreams();

    auto &&left_data = api->GetStreamData(Stream::LEFT);
    auto &&right_data = api->GetStreamData(Stream::RIGHT);

    cv::Mat img;
    cv::hconcat(left_data.frame, right_data.frame, img);
    //cv::imshow("frame", img);
    count++;
    auto &&disp_data = api->GetStreamData(Stream::DISPARITY_NORMALIZED);
    auto &&depth_data = api->GetStreamData(Stream::DEPTH);
    if (!disp_data.frame.empty() && !depth_data.frame.empty()) {
      // Show disparity instead of depth, but show depth values in region.
      auto &&depth_frame = disp_data.frame;

#ifdef WITH_OPENCV3
      // ColormapTypes
      //   http://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html#ga9a805d8262bcbe273f16be9ea2055a65
      cv::applyColorMap(depth_frame, depth_frame, cv::COLORMAP_JET);
      cv::Mat img_depth_only;
      img_depth_only = depth_data.frame;

      img_depth_only.convertTo(img_depth_only,CV_8UC1);
      cv::applyColorMap(img_depth_only, img_depth_only, cv::COLORMAP_JET);
#endif

      //cv::setMouseCallback("depth", OnDepthMouseCallback, &depth_region);
      // Note: DrawRect will change some depth values to show the rect.
      //depth_region.DrawRect(depth_frame);
      //DepthRegion dada;
      std::time_t current_time;
      time(&current_time);
      int width = depth_frame.cols;
      int height = depth_frame.rows;
      int _stride = depth_frame.step;
      int counter = 0;
      int wichi = 0;
      int wichj = 0;
      int low = 10000;//depth_frame.at<unsigned short>(0,0);
      for(int i=0;i<height;i++)
      {
        for(int j=0;j<width;j++){
            //val = depth_frame[i,j]
            unsigned short val = depth_data.frame.at<unsigned short>(i,j);
            //cout<<val;
            if(val<10000 && val>500){
              if(val<low){
                low = val;
                wichj = j;
                wichi = i;
              }
              //std::cout<<"value is"<<low<<" i is"<<i<<" j is"<<j<<std::endl;
          }  
        }
      }
      //std::cout<<"value is"<<low<<" i is"<<wichi<<" j is"<<wichj<<std::endl;
      std::cout<<low<<std::endl;
      cv::Mat combined;
      cv::hconcat(depth_frame,img_depth_only, combined);
      //cv::imwrite("folder/mynt/depth_normal_"+std::to_string(count)+"_"+std::to_string(current_time)+"_"+std::to_string(low)+".jpg", depth_frame);
      cv::imwrite("folder/mynt/combineddepth_"+std::to_string(count)+"_"+std::to_string(current_time)+"_"+std::to_string(low)+"_i_"+std::to_string(wichi)+"_j_"+std::to_string(wichj)+".jpg", combined);
      cv::imwrite("folder/mynt/frame"+std::to_string(count)+"_"+std::to_string(current_time)+"_"+std::to_string(low)+".jpg",img);
      //cv::Mat frame;
      //frame = (combined.reshape(0,1));
      //int  imgSize = frame.total()*frame.elemSize();
      //byte * bytes =  new byte[imgSize];
      //std::memcpy(bytes,frame.data,imgSize*sizeof((byte)));
      vector<uchar> buf;
      cv::imencode(".jpg",combined,buf);

      current_socket = h.socket();
      
      sio::message::ptr joinObj = sio::object_message::create();
      joinObj->get_map()["droneId"] = sio::string_message::create("101");
      joinObj->get_map()["data"]= sio::make_shared<std::string>(buf,buf.size());
      
      h.socket()->emit("droneCamera",joinObj);
      _lock.lock();
      EM("\t\t\t"<<line<<":"<<"You");
      _lock.unlock();


      p.push(buf);
      //Q.push(bytes);
      // Send data here
	  //bytes = send(clientSock, frame.data, imgSize, 0);
    }

  }

  //api->Stop(Source::VIDEO_STREAMING);
  return 0;
}



void send_image(){
  vector<uchar> buf;
  cv::Mat received;
  while(true):
    buf = p.pop();
    //received = cv::imdecode(".jpg",combined,buf);
    sio::message::ptr joinObj = sio::object_message::create();
    joinObj->get_map()["droneId"] = sio::string_message::create("101");
    joinObj->get_map()["data"]= sio::make_shared<std::string>(buf,buf.size());
    h.socket()->emit("droneCamera",joinObj);
}

void main_filter(const cv::Mat &depth){
    cv::Mat edge_mask = edge_mask(depth);
    cv::Mat harris_mask = harris_mask(depth);
    cv::Mat combined_mask = cv2::bitwise_or(edge_mask,harris_mask);
}


void edge_mask(const cv::Mat &decimated_ir){
  cv::Scharr(decimated_ir, scharr_x, CV_16S, 1, 0);
  cv::convertScaleAbs(scharr_x, abs_scharr_x);
  cv::Scharr(decimated_ir, scharr_y, CV_16S, 0, 1);
  cv::convertScaleAbs(scharr_y, abs_scharr_y);
  cv::addWeighted(abs_scharr_y, 0.5, abs_scharr_y, 0.5, 0, edge_mask);
  cv::threshold(edge_mask, edge_mask, 192, 255, cv::THRESH_BINARY);
  return edge_mask;
}


void harris_mask(const cv::Mat &decimated_ir){
  decimated_ir.convertTo(float_ir, CV_32F);
  cv::cornerHarris(float_ir, corners, 2, 3, 0.04);
  cv::threshold(corners, corners, 300, 255, cv::THRESH_BINARY);
  corners.convertTo(harris_mask, CV_8U);
  return harris_mask;
}
