# import the necessary packages
# from imutils.video import FPS
import sys
import numpy as np
import argparse
import imutils
import cv2
import queue

from timeloop import Timeloop
from datetime import timedelta
import time

import threading
from collections import namedtuple
from tqdm import tqdm
from itertools import zip_longest


DetectionResult = namedtuple("DetectionResult", ["frame", "faces"])

class FPS:
    def __init__(self):
        self.start()

    def start(self):
        self._start = time.time()
        self.frames = 0

    def stop(self):
        pass

    def update(self):
        self.frames += 1

    def fps(self):
        _end   = time.time()
        return  self.frames # / (_end - self._start)

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

class Detector:
    def __init__(self, args, in_iter, out_q, terminating, _id):
        self.in_iter = in_iter
        self._id = _id
        self.out_q = out_q
        self.args = args
        self.process_thd = threading.Thread(target=self.process_entry, name='detector_thd_'+str(_id))
        self.terminating = terminating
        self.net = None

    def start_thread(self):
        self.process_thd.start()
        return self.process_thd

    def load_model(self):
        # load our serialized model from disk
        net = cv2.dnn.readNetFromCaffe(self.args.prototxt, self.args.model)
        # check if we are going to use GPU
        if self.args.use_gpu:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return net

    def detect_faces(self, detection_model, frame, conf):
        # Grab frame dimention and convert to blob
        (h,w) =  frame.shape[:2]
        if (h,w) != (300,300):
            frame = cv2.resize(frame, (300, 300))
        # Preprocess input image: mean subtraction, normalization
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        # Set read image as input to model
        detection_model.setInput(blob)

        # Run forward pass on model. Receive output of shape (1,1,no_of_predictions, 7)
        predictions = detection_model.forward()
        coord_list = []
        count = 0
        for i in range(0, predictions.shape[2]):
            confidence = predictions[0,0,i,2]
            if confidence > conf:
                # Find box coordinates rescaled to original image
                box_coord = predictions[0,0,i,3:7] * np.array([w,h,w,h])
                # conf_text = '{:.2f}'.format(confidence)
                # Find output coordinates
                xmin, ymin, xmax, ymax = box_coord.astype('int')
                coord_list.append([xmin, ymin, (xmax-xmin), (ymax-ymin)])
                
            # print('Coordinate list:', coord_list)

        return coord_list

    def process_entry(self):
        print(f"detector_thd_{self._id} started")
        net = self.load_model()

        while not self.terminating.is_set():
            frame = next(self.in_iter)

            faces = self.detect_faces(net, frame, self.args.confidence)

            ret = DetectionResult(frame, faces)
            # return [DetectionResult(fr, fc) for fr,fc in zip(frame, faces)]
            self.out_q.put(ret)
        print(f"detector_thd_{self._id} terminated")

class Input:
    def __init__(self, args, terminating):
        self.args = args
        self.terminating = terminating
        self.process_thd = threading.Thread(target=self.input_thd_etnry, name='input_thd')
        self.in_q = queue.Queue(16)
        self.en = self.video_en() if self.args.fixed_size is None else None # self.fixed_en()  

    def video_en(self):
        print("[INFO] accessing video stream...")
        vs = cv2.VideoCapture(self.args.input if self.args.input else 0)
        while not self.terminating.is_set():
            (grabbed, frame) = vs.read()
            if not grabbed:
                print("stream terminated")
                break
            yield frame

    def fixed_en(self):
        w,h = self.args.fixed_size.split('x')
        frame = np.zeros((int(h),int(w),3), dtype=np.uint8)
        while not self.terminating.is_set():
            yield frame

    # called from worker threads
    def worker_en(self):
        last_frame = None
        while not self.terminating.is_set():
            try:
                frame = self.in_q.get_nowait()
                last_frame = frame.copy()
                # print('from queue')
                yield frame
                continue
            except queue.Empty:
                pass

            if last_frame is not None:
                frame = last_frame.copy()
                # print('from last_frame')
                yield frame
                continue

            time.sleep(1)
                          

    def get_worker_iter(self):
        if self.args.fixed_size is not None:
            return iter(self.fixed_en())
        return iter(self.worker_en())

    def input_thd_etnry(self):
        print("input thread started")
        if self.en is None:
            print("no input source set or fixed size source - exit")
            return
        _iter = iter(self.en)
        while not self.terminating.is_set():
            frame = next(_iter)
            try:
                self.in_q.put_nowait(frame)
            except queue.Full:
                time.sleep(0.001)
            # if not self.in_q.full():
                # print('put!')
            # else:
            #     print('full!')
            # sleep(1)
        print("input thread terminated")

    def start_thread(self):
        self.process_thd.start()
        return self.process_thd

class FPSCounter:
    def __init__(self, in_q, out_q, terminating, avg_len=5):
        self.in_q  = in_q
        self.out_q = out_q
        self.avg_len = avg_len
        self.terminating = terminating
        self.process_thd = threading.Thread(target=self.thd_etnry, name='fps_thd')
        self.fps = FPS()
        self.fps.start()
        self.fps.stop()
        self.last_fps = []
        self.tl = Timeloop()
        _deco = self.tl.job(interval=timedelta(seconds=1))
        _deco(self.fps_job)

    def start_thread(self):
        self.tl.start(block=False)
        self.process_thd.start()
        return self.process_thd
    
    def thd_etnry(self):
        print("fps thread started")
        while not self.terminating.is_set():
            ret = self.in_q.get()
            self.fps.update()
            try:
                self.out_q.put_nowait(ret)
            except queue.Full:
                pass

        self.tl.stop()
        print("fps thread terminated")

    def get_last_fps(self):
        return sum(self.last_fps)/len(self.last_fps)

    def fps_job(self):
        self.fps.stop()
        self.last_fps.append(self.fps.fps())
        self.fps.start()
        if len(self.last_fps) > self.avg_len:
            self.last_fps.pop(0)
        print(f"fps: curr={self.last_fps[-1]:3.2f}, min={min(self.last_fps):3.2f}, avg={self.get_last_fps():3.2f}, max={max(self.last_fps):3.2f}")


class Display:
    def draw(self, en, fps):
        for ret in en:
            frame, faces = ret.frame, ret.faces

            crop = None
            face_offsets = (30, 40)
            for face_coordinates in faces:
                x1, x2, y1, y2 = self.apply_offsets(face_coordinates, face_offsets)
                _x1, _x2 = np.clip([x1,x2], 0, frame.shape[1])
                _y1, _y2 = np.clip([y1,y2], 0, frame.shape[0])
                # crop = frame[_y1:_y2, _x1:_x2].copy()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            cv2.putText(frame, f"{fps.get_last_fps():.2f}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            # show the output frame
            yield ret

    def show(self, en, fps):
        for ret in en:
            cv2.imshow("Frame", frame)
            # if crop is not None and np.prod(crop.shape) > 0:
            #     cv2.imshow("crop", crop)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            yield ret

    def write(self, en):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.output, fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)

        for ret in en:
            frame, faces = ret.frame, ret.faces
            writer.write(frame)
            yield ret

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

class Main:

    def __init__(self, args):
        self.args = args
        self.terminating = threading.Event()


    def apply_offsets(self, face_coordinates, offsets):
        x, y, width, height = face_coordinates
        x_off, y_off = offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


    def entry(self):
        print("main thread started")
        try:
            _input = Input(self.args, self.terminating)
            _input.start_thread()
            _in_iter = _input.get_worker_iter()
            _in_iter = threadsafe_iter(_in_iter)
            out_q = queue.Queue(16)

            for i in range(self.args.workers):
                det = Detector(self.args, _in_iter, out_q, self.terminating, i)
                det.start_thread()

            disp_q = queue.Queue(16)
            fps = FPSCounter(out_q, disp_q, self.terminating)
            fps.start_thread()

            if args.display > 0 or args.output != "":

                disp = Display()
                en = iter(disp_q.get, None)
                en = disp.draw(en, fps)

                if args.display > 0:
                    en = disp.show(en)
                if args.output != "":
                    en = disp.write(en)

                _iter_en = iter(en)
                while True:
                    next(_iter_en)
            else:
                while True:
                    time.sleep(1)
        finally:
            self.terminating.set()
            print("main thread terminated")


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str, default="",
        help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="",
        help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1,
        help="whether or not output frame should be displayed")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
    ap.add_argument("-u", "--use-gpu", type=bool, default=False,
        help="boolean indicating if CUDA GPU should be used")
    ap.add_argument("--workers", type=int, default=1, help="num of workers")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--fixed_size", type=str, help='WxH, use fixed image size (not read from input)')

    args = ap.parse_args()

    main = Main(args)
    main.entry()
