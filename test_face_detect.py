from PIL import Image
import face_recognition
import cv2
import numpy as np

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("test_1.jpg")

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(image)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

for face_location in face_locations:

    # Print the location of each face in this image
    yStart, xEnd, yEnd, xStart = face_location
    # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    print('yEnd -  yStart = ' + str(yEnd - yStart))
    print('xEnd -  xStart = ' + str(xEnd - xStart))
    face_image = image[yStart:yEnd, xStart:xEnd]
    pil_image = Image.fromarray(face_image)
    pil_image.show()


image = cv2.imread("test_1.jpg")
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
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

    print('endX -  startX = ' + str(endX - startX))
    print('endY -  startY = ' + str(endY - startY))

    # draw the bounding box of the face along with the associated
    # probability
    text = "{:.2f}%".format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cropped = image[endY-(endX - startX):endY, startX:endX]
cv2.imshow('image ', cropped)
cv2.waitKey(0)