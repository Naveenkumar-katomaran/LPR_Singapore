# from openvino.inference_engine import IENetwork
# from openvino.inference_engine import IECore
# import numpy as np
# from PIL import Image
# import math

# ie = IECore()
# net = IENetwork(model="./model/ocr/yolo_v3_tiny_openvino/frozen_darknet_yolov4_tiny_ocr.xml", 
#                 weights="./model/ocr/yolo_v3_tiny_openvino/frozen_darknet_yolov4_tiny_ocr.bin")
# net1 = IENetwork(model="./model/lp_detection/yolo_v3_tiny_lp_det_openvino/frozen_darknet_yolov3_tiny_lp_detection_model.xml", 
#                 weights="./model/lp_detection/yolo_v3_tiny_lp_det_openvino/frozen_darknet_yolov3_tiny_lp_detection_model.bin")

# m_input_size = 416

# yolo_scale_13 = 13
# yolo_scale_26 = 26
# yolo_scale_52 = 52

# classes = 34
# coords = 4
# num = 3
# anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]

# label_text_color = (255, 255, 255)
# label_background_color = (125, 175, 75)
# box_color = (255, 128, 0)
# box_thickness = 1

# def EntryIndex(side, lcoords, lclasses, location, entry):
#     n = int(location / (side * side))
#     loc = location % (side * side)
#     return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)

# def IntersectionOverUnion(box_1, box_2):
#     width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
#     height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
#     area_of_overlap = 0.0
#     if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
#         area_of_overlap = 0.0
#     else:
#         area_of_overlap = width_of_overlap_area * height_of_overlap_area
#     box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
#     box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
#     area_of_union = box_1_area + box_2_area - area_of_overlap
#     retval = 0.0
#     if area_of_union <= 0.0:
#         retval = 0.0
#     else:
#         retval = (area_of_overlap / area_of_union)
#     return retval


# class DetectionObject():
#     xmin = 0
#     ymin = 0
#     xmax = 0
#     ymax = 0
#     class_id = 0
#     confidence = 0.0

#     def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
#         self.xmin = int((x - w / 2) * w_scale)
#         self.ymin = int((y - h / 2) * h_scale)
#         self.xmax = int(self.xmin + w * w_scale)
#         self.ymax = int(self.ymin + h * h_scale)
#         self.class_id = class_id
#         self.confidence = confidence

# def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):

#     out_blob_h = blob.shape[2]
#     out_blob_w = blob.shape[3]

#     side = out_blob_h
#     anchor_offset = 0

#     if len(anchors) == 18:   ## YoloV3
#         if side == yolo_scale_13:
#             anchor_offset = 2 * 6
#         elif side == yolo_scale_26:
#             anchor_offset = 2 * 3
#         elif side == yolo_scale_52:
#             anchor_offset = 2 * 0

#     elif len(anchors) == 12: ## tiny-YoloV3
#         if side == yolo_scale_13:
#             anchor_offset = 2 * 3
#         elif side == yolo_scale_26:
#             anchor_offset = 2 * 0

#     else:                    ## ???
#         if side == yolo_scale_13:
#             anchor_offset = 2 * 6
#         elif side == yolo_scale_26:
#             anchor_offset = 2 * 3
#         elif side == yolo_scale_52:
#             anchor_offset = 2 * 0

#     side_square = side * side
#     output_blob = blob.flatten()

#     for i in range(side_square):
#         row = int(i / side)
#         col = int(i % side)
#         for n in range(num):
#             obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords)
#             box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
#             scale = output_blob[obj_index]
#             if (scale < threshold):
#                 continue
#             x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
#             y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
#             height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
#             width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
#             for j in range(classes):
#                 class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
#                 prob = scale * output_blob[class_index]
#                 if prob < threshold:
#                     continue
#                 obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
#                 objects.append(obj)
#     return objects

# a = ie.load_network(net, 'CPU',None)
# a1 = ie.load_network(net1, 'CPU',None)
# def get_bbox_openvino(image, threshold=0.5,model="ocr"):
#     if model=="lp":
#         aa = a1
#     else:
#         aa = a
#     h,w= image.shape[:2]
#     image= Image.fromarray(image)
#     # print(w,h)
#     image_res = image.resize((416, 416),2)
#     rgb_img_expanded = np.stack([image_res], axis=0)
#     rgb_img_expanded=np.array(rgb_img_expanded)

#     f = np.transpose(rgb_img_expanded, (0,3,1,2))
#     new_w = int(w * 416/w)
#     new_h = int(h * 416/h)
    
#     res=aa.infer(inputs={'inputs': f})
    
#     objects = []

#     for output in res.values():
#         objects = ParseYOLOV3Output(output, new_h, new_w, h, w, 0.4, objects)
#     objlen = len(objects)
#     for i in range(objlen):
#         if (objects[i].confidence == 0.0):
#             continue
#         for j in range(i + 1, objlen):
#             if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4):
#                 if objects[i].confidence < objects[j].confidence:
#                     objects[i], objects[j] = objects[j], objects[i]
#                 objects[j].confidence = 0.0
    
#     rg_boxes = []
#     rg_confidences = []
#     rg_classids = []
#     for obj in objects:
#         if obj.confidence < threshold:
#             continue
#         label = obj.class_id
#         confidence = obj.confidence
#         rg_boxes.append([obj.xmin, obj.ymin, obj.xmax - obj.xmin, obj.ymax - obj.ymin])
#         rg_classids.append(label)
#         rg_confidences.append(confidence)
#     return rg_boxes, rg_confidences, rg_classids



# verified, need to add nms
from numpy.lib.type_check import imag
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore
import numpy as np
from PIL import Image
import math
import cv2
import time

#image = Image.open('000029002.jpg')
#draw_image = np.array(image)
# w, h = image.size
#print(w,h)
#image_res = image.resize((416, 416),2)
#rgb_img_expanded = np.stack([image_res], axis=0)
#rgb_img_expanded=np.array(rgb_img_expanded)

#f = np.transpose(rgb_img_expanded, (0,3,1,2))
ie = IECore()
# net = IENetwork(model="./frozen_darknet_yolov4_tiny_ocr.xml", weights="./frozen_darknet_yolov4_tiny_ocr.bin")
net = IENetwork(model="/home/katomaran/Desktop/Ganesh/detection/Models/checkpoint_ssd_resnet1/yolo_v3_tiny_ocr_jan_10/openvino/frozen_darknet_yolov3_model.xml", 
                 weights="/home/katomaran/Desktop/Ganesh/detection/Models/checkpoint_ssd_resnet1/yolo_v3_tiny_ocr_jan_10/openvino/frozen_darknet_yolov3_model.bin")
# net1 = IENetwork(model="./model/lp_detection/yolo_v3_tiny_lp_det_openvino/frozen_darknet_yolov3_tiny_lp_detection_model.xml", 
#                  weights="./model/lp_detection/yolo_v3_tiny_lp_det_openvino/frozen_darknet_yolov3_tiny_lp_detection_model.bin")


#aaa = 0

m_input_size = 416

yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52

classes = 34
coords = 4
num = 3
anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]

LABELS = ('0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z')

label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1
def EntryIndex(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)

def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
    area_of_overlap = 0.0
    if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    retval = 0.0
    if area_of_union <= 0.0:
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    return retval


class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence

def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):

    out_blob_h = blob.shape[2]
    out_blob_w = blob.shape[3]

    side = out_blob_h
    anchor_offset = 0

    if len(anchors) == 18:   ## YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    elif len(anchors) == 12: ## tiny-YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 3
        elif side == yolo_scale_26:
            anchor_offset = 2 * 0

    else:                    ## ???
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    side_square = side * side
    output_blob = blob.flatten()

    for i in range(side_square):
        row = int(i / side)
        col = int(i % side)
        for n in range(num):
            obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords)
            box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
            scale = output_blob[obj_index]
            if (scale < threshold):
                continue
            x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
            y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
            height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
            width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
            for j in range(classes):
                class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
                prob = scale * output_blob[class_index]
                if prob < threshold:
                    continue
                obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                objects.append(obj)
    return objects



'''
{'detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion': <openvino.inference_engine.ie_api.DataPtr object at 0x7fd3ef280768>,
 'detector/yolo-v3-tiny/Conv_9/BiasAdd/YoloRegion': <openvino.inference_engine.ie_api.DataPtr object at 0x7fd3ef280880>}

'''

# new_w = int(w * 416/w)
# new_h = int(h * 416/h)
#aaa = 0
a = ie.load_network(net, 'CPU',None)
print('got it')
# a1 = ie.load_network(net1, 'CPU',None)

def get_bbox_openvino(image, threshold=0.5,model="ocr"):
    if model=="lp":
        aa = "a1"
    else:
        aa = a
    h,w= image.shape[:2]
    image= Image.fromarray(image)
    # print(w,h)
    image_res = image.resize((416, 416),2)
    rgb_img_expanded = np.stack([image_res], axis=0)
    rgb_img_expanded=np.array(rgb_img_expanded)

    f = np.transpose(rgb_img_expanded, (0,3,1,2))
    new_w = int(w * 416/w)
    new_h = int(h * 416/h)
    res=a.infer(inputs={'inputs': f})
    # res = res['detector/yolo-v4-tiny/Conv_17/BiasAdd/YoloRegion']
    # print(type(res))
    # res1 = {}
    # res1['detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion'] = res['detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion']
    # aaa += time.time() - t1
    # print('duration : ', time.time() - t1)
    objects = []
    

    for output in res.values():
        objects = ParseYOLOV3Output(output, new_h, new_w, h, w, 0.4, objects)
    objlen = len(objects)
    for i in range(objlen):
        if (objects[i].confidence == 0.0):
            continue
        for j in range(i + 1, objlen):
            if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4):
                if objects[i].confidence < objects[j].confidence:
                    objects[i], objects[j] = objects[j], objects[i]
                objects[j].confidence = 0.0
        
    rg_boxes = []
    rg_confidences = []
    rg_classids = []

    # Drawing boxes
    for obj in objects:
        if obj.confidence < threshold:
            continue
        label = obj.class_id
        confidence = obj.confidence

        rg_boxes.append([obj.xmin, obj.ymin, obj.xmax - obj.xmin, obj.ymax - obj.ymin])
        rg_classids.append(int(label))
        rg_confidences.append(float(confidence))
    return rg_boxes, rg_confidences, rg_classids
