import cv2
import os
import logging as log

logging = log.getLogger('Rotating Log')


def raw2yolo(annotation_list, width, height, dst):
    labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # print("annotation_list  :  ", annotation_list)
    txtfile = open(dst, "w")
    for objects in annotation_list:
        label = labels.find(objects[0])
        raw_x = objects[1][0]
        raw_y = objects[1][1]
        raw_width = objects[1][2]
        raw_height = objects[1][3]

        w = raw_width / width
        h = raw_height / height
        x = (raw_x + (raw_x + raw_width)) / 2 / width
        y = (raw_y + (raw_y + raw_height)) / 2 / height
        str_r = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(label, x, y, w, h)
        txtfile.write(str_r)

    txtfile.close()


def write_fun(image, file_name, folder_name, parent_folder_name, annotation_list):
    if not os.path.exists(parent_folder_name + '/' + folder_name):
        os.system("mkdir {}/{}".format(parent_folder_name, folder_name))
    cv2.imwrite("{}/{}/{}.jpg".format(parent_folder_name, folder_name, file_name), image)
    if annotation_list is not None:
        (h, w) = image.shape[:2]
        raw2yolo(annotation_list, w, h, "{}/{}/{}.txt".format(parent_folder_name, folder_name, file_name))


def write_list(image_list, folder_name, parent_folder_name, annotation_list=None):
    if annotation_list is None:
        logging.info(
            "Car Images Folder Details  : \nParent_folder_name  :  {}, Folder_name  :  {} ".format(parent_folder_name,
                                                                                                   folder_name))
    else:
        logging.info(
            "Car Images Folder Details  : \nParent_folder_name  :  {}, Folder_name  :  {} ".format(parent_folder_name,
                                                                                                   folder_name))

    for index, image in enumerate(image_list):
        if annotation_list is not None:
            write_fun(image, str(index), folder_name, parent_folder_name, annotation_list[index])
        else:
            write_fun(image, str(index), folder_name, parent_folder_name, annotation_list)
