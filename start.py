import time

import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.process
import tornado.template
import video
import gen
import os

from VideoGet import VideoGet
import video
from imutils.video import FPS
import cv2
from threading import Thread

cam = None
video_getter = None
html_page_path = dir_path = os.path.dirname(
    os.path.realpath(__file__)) + '/www/'

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('test.mp4',fourcc, 25.0, (640,480))
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('test.avi', fourcc, 10.0, (640, 480))


class HtmlPageHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self, file_name='index.html'):
        # Check if page exists
        index_page = os.path.join(html_page_path, file_name)
        if os.path.exists(index_page):
            # Render it
            self.render('www/' + file_name)
        else:
            # Page not found, generate template
            err_tmpl = tornado.template.Template(
                "<html> Err 404, Page {{ name }} not found</html>")
            err_html = err_tmpl.generate(name=file_name)
            # Send response
            self.finish(err_html)


class SetParamsHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def post(self):
        # print self.request.body
        # get args from POST request
        # width = int(self.get_argument('width'))
        # height = int(self.get_argument('height'))
        # try to change resolution
        try:
            # cam.set_resolution(width, height)
            self.write({'resp': 'ok'})
        except:
            self.write({'resp': 'bad'})
            self.flush()
            self.finish()


class StreamHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        ioloop = tornado.ioloop.IOLoop.current()

        self.set_header(
            'Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
        self.set_header('Pragma', 'no-cache')
        self.set_header(
            'Content-Type', 'multipart/x-mixed-replace;boundary=--jpgboundary')
        self.set_header('Connection', 'close')

        self.served_image_timestamp = time.time()
        my_boundary = "--jpgboundary"
        while True:
            # Generating images for mjpeg stream and wraps them into http resp
            # if self.get_argument('fd') == "true":
            #     img = cam.get_frame(True)
            # else:
            #     img = cam.get_frame(False)
            # img = None
            # if video_getter.frame_lock:
            #     img = video_getter.frame
            # else:
            #time.sleep(1 / 25)
            img = video_getter.frame_to_send
            ret, jpeg = cv2.imencode('.jpg', img)
            img = jpeg.tobytes()
            interval = 0.25
            if self.served_image_timestamp + interval < time.time():
                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(img)
                self.served_image_timestamp = time.time()
                yield tornado.gen.Task(self.flush)
            else:
                yield tornado.gen.Task(ioloop.add_timeout, ioloop.time() + interval)


def make_app():
    # add handlers
    return tornado.web.Application([
        (r'/', HtmlPageHandler),
        (r'/video_feed', StreamHandler),
        (r'/setparams', SetParamsHandler),
        (r'/(?P<file_name>[^\/]+htm[l]?)+', HtmlPageHandler),
        (r'/(?:image)/(.*)', tornado.web.StaticFileHandler,
         {'path': './image'}),
        (r'/(?:css)/(.*)', tornado.web.StaticFileHandler,
         {'path': './css'}),
        (r'/(?:js)/(.*)', tornado.web.StaticFileHandler, {'path': './js'})
    ],
    )


def threadVideoGet(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('outputframe.mp4', fourcc, 60, (1280, 720))

    fps = FPS().start()
    try:
        while True:
            if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
                video_getter.stop()
                break
            # video_getter.frame_lock = True
            frame = video_getter.frame
            cam.detect_face(frame)
            video_getter.frame_to_send = frame
            # video_getter.frame_lock = False
            #   cv2.imshow("Video", frame)
            # write the flipped frame
#			out.write(frame)
            fps.update()
            #time.sleep(1/20)
            # out.write(frame)
    except KeyboardInterrupt:
        print("Goodbye")
    finally:
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        out.release()


if __name__ == "__main__":
    # creates camera
    print("Here")
    cam = video.UsbCamera()
    # video.UsbCamera()
    # bind server on 8080 port
    #app = make_app()
    #app.listen(8000)
    URL = 'http://118.185.61.234:8000/stream.mjpg'
    #URL = "/home/ubuntu/tida/tida-2/input_videos/outputfinal.mp4"
    video_getter = VideoGet(URL).start()
    Thread(target=threadVideoGet, args=()).start()
    # threadVideoGet()
    tornado.ioloop.IOLoop.current().start()
