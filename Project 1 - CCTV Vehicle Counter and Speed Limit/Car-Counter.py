import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png") # the mask is here to only detect cars from a specific region on the image

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCountCar = []
totalCountMoto = []
totalCountTruck = []
totalCountBus = []


tracker_class_association = {}  # A new dictionary to hold tracker ID - class associations

frame_rate = cap.get(cv2.CAP_PROP_FPS)  # get frame rate
scale = 0.05

tracker_positions = {}  # dictionary to keep track of positions of objects
tracker_start_frames = {}  # dictionary to keep track of start frame of objects



while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED) # import in the loop for better visual
    # imgGraphics = cv2.resize(imgGraphics, (img.shape[1], img.shape[0]))
    # img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" or currentClass == "person" or currentClass == "bicycle" or \
                    currentClass == "traffic light" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

                # added

                # Store the class associated with this box's ID in the dictionary
                tracker_class_association[id] = currentClass  # 'id' is the tracker ID associated with this box

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        # cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
        #                    scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        if id not in tracker_positions:
            tracker_positions[id] = (cx, cy)  # store the initial position
            tracker_start_frames[id] = cap.get(cv2.CAP_PROP_POS_FRAMES)  # store the initial frame
        else:
            initial_position = tracker_positions[id]
            initial_frame = tracker_start_frames[id]
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            distance_pixels = math.sqrt((cx - initial_position[0]) ** 2 + (cy - initial_position[1]) ** 2)
            distance_meters = distance_pixels * scale  # apply scale
            time_seconds = (current_frame - initial_frame) / frame_rate
            speed_m_per_s = distance_meters / time_seconds
            speed_km_per_h = speed_m_per_s * 3.6  # convert to km/h
            print(f'Speed of object {id}: {speed_km_per_h} km/h')

            # Show the class, id and speed of the object on the frame
            cvzone.putTextRect(img,
                               f'{tracker_class_association.get(int(id), "Unknown")} {int(id)} Speed: {speed_km_per_h:.2f} km/h',
                               (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # modify
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            # currentClass = tracker_class_association.get(int(id), "Unknown")
            if currentClass == "car":
                if totalCountCar.count(id) == 0:
                    totalCountCar.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) # when the region crosses the line it turns green

            if currentClass == "bus":
                if totalCountBus.count(id) == 0:
                    totalCountBus.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) # when the region crosses the line it turns green

            if currentClass == "truck":
                if totalCountTruck.count(id) == 0:
                    totalCountTruck.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) # when the region crosses the line it turns green

            if currentClass == "motorbike":
                if totalCountMoto.count(id) == 0:
                    totalCountMoto.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) # when the region crosses the line it turns green

        # # og code
        # if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
        #     if totalCount.count(id) == 0:
        #         totalCount.append(id)
        #         cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) # when the region crosses the line it turns green
    base_y = 500  # base y position for the text
    offset_y = 70  # how much to move down for each subsequent line

    cvzone.putTextRect(img, f'Car Count: {len(totalCountCar)}', (780, base_y))
    cvzone.putTextRect(img, f'Bus Count: {len(totalCountBus)}', (780, base_y + offset_y))
    cvzone.putTextRect(img, f'Truck Count: {len(totalCountTruck)}', (780, base_y + 2 * offset_y))
    cvzone.putTextRect(img, f'Motorbike Count: {len(totalCountMoto)}', (780, base_y + 3 * offset_y))

    # cvzone.putTextRect(img, f' Car Count: {len(totalCountCar)}', (50, 50))
    # cvzone.putTextRect(img, f' Car Count: {len(totalCountBus)}', (40, 50))
    # cvzone.putTextRect(img, f' Car Count: {len(totalCountTruck)}', (30, 50))
    # cvzone.putTextRect(img, f' Car Count: {len(totalCountMoto)}', (20, 50))
    # cv2.putText(img,str(len(totalCountCar)),(1,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    # cv2.putText(img, str(len(totalCountBus)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    # cv2.putText(img, str(len(totalCountTruck)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    # cv2.putText(img, str(len(totalCountMoto)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # # og code
    # cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)