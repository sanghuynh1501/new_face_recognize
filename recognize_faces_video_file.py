# USAGE
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4 --output output/lunch_scene_output.avi --display 0

# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import numpy as np

import RPi.GPIO as GPIO
import time

servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50)
p.start(2.5)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", default=0,
	help="path to input video")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load model from disk
print("[INFO] loading from model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# initialize the pointer to the video file and the video writer
print("[INFO] processing video...")
stream = cv2.VideoCapture(args["input"])
writer = None
dem = 0

try:
  # loop over frames from the video file stream
  while True:
    # grab the next frame
    (grabbed, frame) = stream.read()
    if frame is not None:
      # convert the input frame from BGR to RGB then resize it to have
      # a width of 750px (to speedup processing)
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      rgb = imutils.resize(frame, width=750)
      frame = imutils.resize(frame, width=750)
      r = frame.shape[1] / float(rgb.shape[1])
     
      (h, w) = frame.shape[:2]
      blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
      # pass the blob through the network and obtain the detections and
      # predictions
      print("[INFO] computing object detections...")
      net.setInput(blob)
      detections = net.forward()
      for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
          # compute the (x, y)-coordinates of the bounding box for the
          # object
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")
          # startY1 = int(startY * 2.4)
          # startX1 = int(startX * 1.4)
          # endX1 = endX
          # endY1 = startY1 + (startX1 - endX1)
          boxes = [(endY - (endX - startX), endX, endY, startX)]
          # cropped = frame[endY-(endX - startX):endY, startX:endX]
          # cv2.imshow('image ', cropped)
          # cv2.waitKey(0)
          # # detect the (x, y)-coordinates of the bounding boxes
          # # corresponding to each face in the input frame, then compute
          # # the facial embeddings for each face
          # top, right, bottom, left = face_recognition.face_locations(rgb, model=args["detection_method"])[0]
          # cropped1 = frame[top:bottom, left:right]
          # cv2.imshow('image 1', cropped1)
          # cv2.waitKey(0)

          encodings = face_recognition.face_encodings(rgb, boxes)
          names = []

          # loop over the facial embeddings
          for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
              encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
              # find the indexes of all matched faces then initialize a
              # dictionary to count the total number of times each face
              # was matched
              matchedIdxs = [i for (i, b) in enumerate(matches) if b]
              counts = {}

              # loop over the matched indexes and maintain a count for
              # each recognized face face
              for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            
              # determine the recognized face with the largest number
              # of votes (note: in the event of an unlikely tie Python
              # will select first entry in the dictionary)
              name = max(counts, key=counts.get)
            # update the list of name
            names.append(name)
          print("name ", names[0])
          if names[0] == "sang_huynh":
            p.ChangeDutyCycle(5)
            time.sleep(0.5)
            p.ChangeDutyCycle(7.5)
            time.sleep(0.5)
            p.ChangeDutyCycle(10)
            time.sleep(0.5)
            p.ChangeDutyCycle(12.5)
            time.sleep(0.5)
            p.ChangeDutyCycle(10)
            time.sleep(0.5)
            p.ChangeDutyCycle(7.5)
            time.sleep(0.5)
            p.ChangeDutyCycle(5)
            time.sleep(0.5)
            p.ChangeDutyCycle(2.5)
            time.sleep(0.5)
except KeyboardInterrupt:
  p.stop()
  GPIO.cleanup()
