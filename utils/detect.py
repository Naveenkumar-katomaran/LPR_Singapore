import cv2
import numpy as np


from ultralytics import YOLO

import torch

def resolve_device(device_str):
    device_str = device_str.lower()
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not found, falling back to CPU")
        return "cpu"
    return device_str

def get_ln(config_file, weight_file, device="auto"):
    resolved_device = resolve_device(device)
    
    # If weight_file is a .pt model, we use Ultralytics
    if weight_file.endswith('.pt'):
        model = YOLO(weight_file)
        model.to(resolved_device)
        return model, None
    
    # Fallback to Darknet/OpenCV for .weights files
    net = cv2.dnn.readNetFromDarknet(config_file, weight_file)
    
    # Apply device settings to OpenCV DNN
    if resolved_device.startswith("cuda"):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    ln = net.getLayerNames()
    unconnected_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_layers, np.ndarray):
        unconnected_layers = unconnected_layers.flatten()
    return net, [ln[i - 1] for i in unconnected_layers]


def get_bbox(image, net, ln, threshold):
    # Check if we are using Ultralytics YOLO (net will be the YOLO object)
    if isinstance(net, YOLO):
        results = net(image, conf=threshold, verbose=False)[0]
        boxes = []
        confidences = []
        classIDs = []
        
        for result in results.boxes:
            # result.xywh is [center_x, center_y, width, height]
            # result.xyxy is [xmin, ymin, xmax, ymax]
            b = result.xywh[0].cpu().numpy()
            centerX, centerY, width, height = b
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(result.conf[0]))
            classIDs.append(int(result.cls[0]))
        
        return boxes, confidences, classIDs

    # Original OpenCV DNN fallback
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                            0.3)

    ret_boxes = []
    ret_confidences = []
    ret_classids = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            ret_boxes.append(boxes[i])
            ret_confidences.append(confidences[i])
            ret_classids.append(classIDs[i])
        return ret_boxes, ret_confidences, ret_classids
    return None, None, None


def get_key_x(item):
    return item[1][0]


def sort_rect(recog_info):
    info = sorted(recog_info, key=get_key_x)

    avgH = 0.0
    for item in info:
        avgH += float(item[1][3])
    avgH /= len(info)  # average height

    max_dy = 0.0
    for i in range(0, len(info) - 1):
        if abs(float(info[i][1][1]) - float(info[i + 1][1][1])) > max_dy:
            max_dy = abs(float(info[i][1][1]) - float(info[i + 1][1][1]))

    tY = avgH / 2
    if max_dy > tY:  # 2 Lines
        line1 = []
        line2 = []
        b_first = True
        line1.append(info[0])
        for i in range(1, len(info)):
            d_y = float(info[i - 1][1][1]) - float(info[i][1][1])
            if abs(d_y) > tY:
                if b_first:
                    line2.append(info[i])
                else:
                    line1.append(info[i])
                b_first = not b_first
            else:
                if b_first:
                    line1.append(info[i])
                else:
                    line2.append(info[i])

        if line1[0][1][1] > line2[0][1][1]:
            info = line2 + line1
        else:
            info = line1 + line2

    return info
