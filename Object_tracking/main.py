import cv2
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway_Trim.mp4")
# object detection from stable camera

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=400)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # extract region of interest
    roi = frame[370: 720, 90: 590]

    border = cv2.line(frame, (525, 630), (20, 630), (0, 0, 255), 1)
# object detection

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)  # for remove shadow but doesn't work my video
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        # calculate area and remove small area
        area = cv2.contourArea(cnt)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
         #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)

# object tracking

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)



    cv2.imshow("roi", roi)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    key = cv2.waitKey(100)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

