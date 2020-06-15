# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import queue

from timeloop import Timeloop
from datetime import timedelta

import threading

tl = Timeloop()

@tl.job(interval=timedelta(seconds=1))
def fps_job():
    main_instance.fps_job()


class Main:
    def __init__(self):
        self.last_fps = 0
        self.fps = FPS()
        self.fps.start()
        self.fps.stop()

        self.input_q = queue.Queue(16)
        self.output_q = queue.Queue(16)
        self.process_thd = threading.Thread(target=self.process_entry)
        self.terminating = False

    def fps_job(self):
        self.fps.stop()
        self.last_fps = self.fps.fps()
        print(f"fps={self.last_fps:3.2f}, inp_shape={None if self.last_frame is None else self.last_frame.shape}")

    def detect_faces(self, detection_model, gray_image_array, conf):
        frame = gray_image_array
        # Grab frame dimention and convert to blob
        (h,w) =  frame.shape[:2]
        self.last_frame = frame
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
                conf_text = '{:.2f}'.format(confidence)
                # Find output coordinates
                xmin, ymin, xmax, ymax = box_coord.astype('int')
                coord_list.append([xmin, ymin, (xmax-xmin), (ymax-ymin)])
                
            # print('Coordinate list:', coord_list)

        return coord_list

    def apply_offsets(self, face_coordinates, offsets):
        x, y, width, height = face_coordinates
        x_off, y_off = offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


    def process_entry(self):
        last_frame = None
        while not self.terminating:
            if not self.input_q.empty():
                frame = self.input_q.get()
                last_frame = frame.copy()
            elif last_frame is not None:
                frame = last_frame.copy()
            else:
                continue
            # print('.', end='',flush=True)
            faces = self.detect_faces(self.net, frame, self.args["confidence"])

            face_offsets = (30, 40)
            # print("len(faces)", len(faces))

            # update the FPS counter
            self.fps.update()
            crop = None
            for face_coordinates in faces:
                x1, x2, y1, y2 = self.apply_offsets(face_coordinates, face_offsets)
                _x1, _x2 = np.clip([x1,x2], 0, frame.shape[1])
                _y1, _y2 = np.clip([y1,y2], 0, frame.shape[0])
                crop = frame[_y1:_y2, _x1:_x2].copy()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                # print(crop.shape, x1, x2, y1, y2)
                # if crop.shape[0] > 0 and crop.shape[1] > 0:
            if not self.output_q.full():
                self.output_q.put((frame, crop))

    def entry(self):

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
        self.args = args = vars(ap.parse_args())

        # load our serialized model from disk
        self.net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        # check if we are going to use GPU
        if args["use_gpu"]:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


        # initialize the video stream and pointer to output video file, then
        # start the FPS timer
        print("[INFO] accessing video stream...")
        vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
        writer = None
        self.process_thd.start()
        tl.start(block=False)

        # loop over the frames from the video stream
        while True:
            # read the next frame from the file
            (grabbed, frame) = vs.read()
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                print("stream terminated")
                break
            # resize the frame, grab the frame dimensions, and convert it to
            # a blob
            self.input_q.put(frame)
            # process as fast as we can
            frame, crop = self.output_q.get()
            if crop is not None and np.prod(crop.shape) > 0:
                print("crop.shape: {crop.shape}")
            # check to see if the output frame should be displayed to our
            # screen
            if args["display"] > 0:

                cv2.putText(frame, f"{self.last_fps:.2f}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                # show the output frame
                cv2.imshow("Frame", frame)
                if crop is not None and np.prod(crop.shape) > 0:
                    cv2.imshow("crop", crop)
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break
            # if an output video file path has been supplied and the video
            # writer has not been initialized, do so now
            if args["output"] != "" and writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
                    (frame.shape[1], frame.shape[0]), True)
            # if the video writer is not None, write the frame to the output
            # video file
            if writer is not None:
                writer.write(frame)
        # stop the timer and display FPS information
        self.fps.stop()
        self.terminating = True
        self.process_thd.join()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))


if __name__ == '__main__':
    main_instance = Main()
    main_instance.entry()
