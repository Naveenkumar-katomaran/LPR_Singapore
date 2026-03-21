import cv2
import numpy as np


from ultralytics import YOLO
import torch
import logging as log

logging = log.getLogger(__name__)

def resolve_device(device_str):
    device_str = device_str.lower()
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device_str == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not found, falling back to CPU")
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
        
        if len(boxes) > 0:
            return boxes, confidences, classIDs
        return None, None, None

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


def get_bbox_batch(images, net, ln, threshold):
    """
    Performs batch inference on a list of images.
    Returns: list of (boxes, confidences, classIDs) tuples.
    """
    if isinstance(net, YOLO):
        results_list = net(images, conf=threshold, verbose=False)
        batch_results = []
        for results in results_list:
            boxes = []
            confidences = []
            classIDs = []
            for result in results.boxes:
                b = result.xywh[0].cpu().numpy()
                centerX, centerY, width, height = b
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(result.conf[0]))
                classIDs.append(int(result.cls[0]))
            
            if len(boxes) > 0:
                batch_results.append((boxes, confidences, classIDs))
            else:
                batch_results.append((None, None, None))
        return batch_results
    
    # Fallback to single calling for Darknet
    return [get_bbox(img, net, ln, threshold) for img in images]


def get_key_x(item):
    return item[1][0]


def sort_rect(recog_info):
    """
    Groups character bounding boxes into text lines using Y-interval overlap.
    Adapted for Indian LPR (1 or 2 lines).
    
    recog_info format: list of tuples (label, [x, y, w, h], confidence)
    """
    if not recog_info:
        return []

    # 1. Filter out extremely low confidence detections to reduce noise
    recog_info = [item for item in recog_info if float(item[2]) >= 0.35]
    if not recog_info:
        return []

    # 2. Pre-sort by vertical centre to make grouping deterministic
    # item[1] is [x, y, w, h]
    info = sorted(
        recog_info,
        key=lambda item: float(item[1][1]) + float(item[1][3]) / 2.0
    )

    groups = []   # list of lists of boxes
    overlap_threshold = 0.4

    for item in info:
        b_y1 = float(item[1][1])
        b_y2 = b_y1 + float(item[1][3])
        b_h  = max(float(item[1][3]), 1.0)

        placed = False
        for group in groups:
            # Current Y span of the group
            g_y1 = min(float(b[1][1]) for b in group)
            g_y2 = max(float(b[1][1]) + float(b[1][3]) for b in group)

            overlap  = min(b_y2, g_y2) - max(b_y1, g_y1)
            min_span = min(b_h, max(g_y2 - g_y1, 1.0))
            ratio    = overlap / min_span

            if ratio >= overlap_threshold:
                group.append(item)
                placed = True
                break

        if not placed:
            groups.append([item])   # start a new line group

    # 3. Guard: keep only the 2 largest groups (discard phantom lines/glare)
    if len(groups) > 2:
        groups = sorted(groups, key=len, reverse=True)[:2]

    # 4. Sort groups top-to-bottom by mean Y
    groups.sort(key=lambda g: sum(
        float(b[1][1]) + float(b[1][3]) / 2.0 for b in g
    ) / len(g))

    # 5. Within each group sort left-to-right by X centre
    sorted_info = []
    for g in groups:
        g.sort(key=lambda b: float(b[1][0]) + float(b[1][2]) / 2.0)
        sorted_info.extend(g)

    return sorted_info
