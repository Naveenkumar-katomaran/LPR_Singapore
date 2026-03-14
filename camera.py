# ------------------------------
# Notice
# ------------------------------

# Copyright 2019 Katomaran Technology and Bussiness solution

# ------------------------------
# Imports
# ------------------------------


from utils.detect import *
from utils.bbox_asumption import *
from utils.ocr import *
#from utils.ocr_bench import *
from utils.db import *

import logging.handlers as lh
import logging as log
from datetime import datetime
from pytz import timezone, utc
import os
import json
import argparse
import subprocess
import select
import numpy as np
from threading import Thread
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from time import strftime, sleep
import schedule
import threading
from queue import Queue


#  ------------------------------------------
#   Command line interface is required for passing camera name and configuration file for the specified residency
#  ------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--camera_name', required=True, help='Camera name is needed')
parser.add_argument('--config_file', required=True, help='Configuration file is needed')
args = parser.parse_args()


#  ------------------------------------------
#   RTSP reader via ffmpeg (when OpenCV has no FFMPEG support)
#  ------------------------------------------

class FFmpegRTSPReader:
    """Read RTSP stream via ffmpeg subprocess; provides .read() and .get(1) like cv2.VideoCapture."""
    def __init__(self, url, width=1920, height=1080):
        self.url = url
        self.width = int(width)
        self.height = int(height)
        self.proc = None
        self._frame_index = 0
        self._start()

    def _start(self):
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                pass
            self.proc = None
        cmd = [
            "ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", self.url,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "{}x{}".format(self.width, self.height), "-an", "-"
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found; install ffmpeg in the container for RTSP (e.g. build image from install.Dockerfile).")

    def read(self, timeout_sec=10):
        if not self.proc or self.proc.poll() is not None:
            return False, None
        size = self.width * self.height * 3
        try:
            ready, _, _ = select.select([self.proc.stdout], [], [], timeout_sec)
            if not ready:
                return False, None
            buf = self.proc.stdout.read(size)
        except Exception:
            return False, None
        if len(buf) != size:
            return False, None
        self._frame_index += 1
        frame = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame

    def get(self, prop):
        if prop == 1:  # CAP_PROP_POS_FRAMES
            return self._frame_index
        return 0

    def release(self):
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                pass
            self.proc = None


def _open_camera_stream(config, camera_name):
    """Open video stream: use ffmpeg for RTSP (avoids OpenCV CAP_IMAGES misdetection), else OpenCV."""
    url = config["camera_url"][camera_name]
    if url.strip().lower().startswith("rtsp://"):
        w = config.get("stream_width", 1920)
        h = config.get("stream_height", 1080)
        return FFmpegRTSPReader(url, w, h)
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


#  ------------------------------------------
#   Custom time (Indian Standard Time - Asia/Kolkata) for logging
#  ------------------------------------------

def custom_time(*args):
    utc_dt = datetime.now(utc)
    my_tz = timezone("Asia/Kolkata")
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()


#  ------------------------------------------
#   Loading the configuration file
#  ------------------------------------------

with open(args.config_file) as file:
    config = json.load(file)

#  ------------------------------------------
#   If image directory is not found, create it
#  ------------------------------------------

if not os.path.exists(config["camera_numberplate_path"]["parent_folder"]):
    os.system(f"mkdir {config['camera_numberplate_path']['parent_folder']}")

#  ------------------------------------------
#   For testing purpose
#  ------------------------------------------

if config["testing"]["status"]:
    from utils.testing_usecase import write_list

    if not os.path.exists(config["testing"]["parent_folder"]):
        os.system(f"mkdir {config['testing']['parent_folder']}")

    if not os.path.exists(config["testing"]["parent_folder"] + "/" + args.camera_name):
        os.system(f"mkdir {config['testing']['parent_folder'] + '/' + args.camera_name}")
    if not os.path.exists(
            config["testing"]["parent_folder"] + "/" + args.camera_name + "/" + config["testing"]["car_image_folder"]):
        os.system(
            f"mkdir {config['testing']['parent_folder'] + '/' + args.camera_name + '/' + config['testing']['car_image_folder']}")
    if not os.path.exists(
            config["testing"]["parent_folder"] + "/" + args.camera_name + "/" + config["testing"]["lp_images_folder"]):
        os.system(
            f"mkdir {config['testing']['parent_folder'] + '/' + args.camera_name + '/' + config['testing']['lp_images_folder']}")
    config["testing"]["car_image_folder"] = config["testing"]["parent_folder"] + "/" + args.camera_name + "/" + \
                                            config["testing"][
                                                "car_image_folder"]
    config["testing"]["lp_images_folder"] = config["testing"]["parent_folder"] + "/" + args.camera_name + "/" + \
                                            config["testing"][
                                                "lp_images_folder"]

#  ------------------------------------------
#   Merging the number plate path with the parent folder for easy use
#  ------------------------------------------

config["camera_numberplate_path"][args.camera_name] = config["camera_numberplate_path"]["parent_folder"] + '/' + \
                                                      config["camera_numberplate_path"][args.camera_name]

if not os.path.exists(config["camera_numberplate_path"][args.camera_name]):
    os.system(f"mkdir {config['camera_numberplate_path'][args.camera_name]}")
if not os.path.exists(config['log_path']):
    os.system(f"mkdir {config['log_path']}")

#  ------------------------------------------
#   Customized logging with file handler and custom time
#  ------------------------------------------
log_level = log.INFO
if "verbose" in config.keys():
    if config["verbose"] is True:
        log_level = log.DEBUG

if not "number_plate_threshold" in config["models"].keys():
    config["models"]["number_plate_threshold"] = 0.8
if not "ocr_threshold" in config["models"].keys():
    config["models"]["ocr_threshold"] = 0.5

checksum_exclude = ['W', 'B', 'M', 'N', 'J', 'CC']
if "checksum_exclude" in config.keys():
    checksum_exclude = config['checksum_exclude']

log.basicConfig(level=log_level, datefmt='%d-%b-%y %H:%M:%S',
                format=f'%(asctime)s - {config["db"]["camera_id"][args.camera_name]} - %(message)s')
log.getLogger('matplotlib.font_manager').disabled = True
logging = log.getLogger('Rotating Log')
formatter = log.Formatter(f'%(asctime)s - {config["db"]["camera_id"][args.camera_name]} -  %(message)s')
log.Formatter.converter = custom_time
handler = lh.TimedRotatingFileHandler(config['log_path'] + '/' + args.camera_name, 'midnight', 1, False)
handler.suffix = "%Y-%m-%d %H:%M:%S"
handler.setFormatter(formatter)
logging.addHandler(handler)

#  ------------------------------------------
#   Loading the region of interest
#  ------------------------------------------

if config.get("regions", {}).get(args.camera_name):
    if config.get("car_in") and args.camera_name in config["car_in"]:
        polygon_pts = Polygon(eval(config["car_in"][args.camera_name]))
        logging.debug(f"Polygon points - {polygon_pts}")
    elif config.get("car_in_relative") and args.camera_name in config["car_in_relative"]:
        logging.info("Dynamic relative bounds will be calculated per frame.")


#  ------------------------------------------
#   MQTT BMS API INTEGRATION
#  ------------------------------------------
MQTT = {'host':"","port":"","username":"","password":"","topic":"","serial_id":""}

if config["mqtt"]["status"] is True and config['application_type'][args.camera_name] == 'normal':
    
    try:
        mqtt_channel = requests.get(url=config["mqtt"]["topic_endpoint"],
                                    headers={"Authorization": config["db"]["Authorization"]},timeout=10)
        logging.info("mqtt channel status code {}".format(mqtt_channel.status_code))
        
        if mqtt_channel.status_code == 200 :

            mqtt_channel = mqtt_channel.json()
            mqtt_channel = mqtt_channel["data"]["values"]
            #logging.info("mqtt_channel info ----{}".format(mqtt_channel))
            MQTT["topic"] = mqtt_channel['lnpr_channel']
        else:    
            MQTT['topic'] = config['mqtt']['topic']
            
    except Exception as e:
        logging.info("MQTT Channel error {}".format(e))
        MQTT['topic'] = config['mqtt']['topic']
        logging.info(f"exception mqtt channel data {MQTT['topic']}")

    try:
        mqtt_config = requests.get(url=config["mqtt"]["mqtt_endpoint"],
                        headers={"Authorization": config["db"]["Authorization"]},timeout=10)

        logging.info("mqtt config status code {}".format(mqtt_config.status_code))

        if mqtt_config.status_code == 200:

            logging.info(mqtt_config)
            mqtt_config = mqtt_config.json()
            mqtt_config = mqtt_config["data"]["values"]
            MQTT["host"] = mqtt_config['host']
            MQTT["port"] = int(mqtt_config['port'])
            MQTT["username"] = mqtt_config['username']
            MQTT["password"] = mqtt_config['password']
            
           
        else:
            MQTT["host"] = config['mqtt']['host']
            MQTT["port"] = int(config['mqtt']['port'])
            MQTT["username"] = config['mqtt']['username']
            MQTT["password"] = config['mqtt']['password']
        if args.camera_name in config['mqtt']['serial_ids'].keys():
            MQTT["serial_id"] = config['mqtt']['serial_ids'][args.camera_name]
        else:
            logging.info("DCS not found {}".format(args.camera_name))
            
    except Exception as e:
        
        logging.info("Internet is diconnected or there is no mqtt config datas fetch {}".format(e))
        MQTT["host"] = config['mqtt']['host']
        MQTT["port"] = int(config['mqtt']['port'])
        MQTT["username"] = config['mqtt']['username']
        MQTT["password"] = config['mqtt']['password']
        if args.camera_name in config['mqtt']['serial_ids'].keys():
            MQTT["serial_id"] = config['mqtt']['serial_ids'][args.camera_name]
        else:
            logging.info("DCS not found {}".format(args.camera_name))
            MQTT = {}

        logging.info(f"exception data mqtt config {MQTT}")

elif config['application_type'][args.camera_name] == 'resident':
    MQTT = {}
    MQTT['topic'] = config['resident_mqtt']['topic']
    MQTT["host"] = config['resident_mqtt']['host']
    MQTT["port"] = int(config['resident_mqtt']['port'])
    MQTT["username"] = config['resident_mqtt']['username']
    MQTT["password"] = config['resident_mqtt']['password']
    if args.camera_name in config['mqtt']['serial_ids'].keys():
        MQTT["serial_id"] = config['mqtt']['serial_ids'][args.camera_name]
    else:
        logging.info("DCS not found {}".format(args.camera_name))
        MQTT = {}
    

logging.info("MQTT DETAILS FETCH DATAS {}".format(MQTT))


#  ------------------------------------------
#   Creating an object to process the detected results
#  ------------------------------------------



TextProcess_var = TextProcess(config, args.camera_name,MQTT)

if config["testing"]["status"]:
    tester = 1
    car_thread_list = []
    lp_thread_list = []


def lp_detection(image, dt_net, dt_ln):
    #  ------------------------------------------
    #   Licence plate are detected here
    #  ------------------------------------------

    # Build the active polygon for this frame
    # We dynamically scale `car_in_relative` to the actual image shape,
    # adapting effortlessly if the RTSP resolution changes.
    img_h, img_w = image.shape[:2]
    active_polygon = None
    
    if config.get("regions", {}).get(args.camera_name):
        if config.get("car_in_relative") and args.camera_name in config["car_in_relative"]:
            rel_pts = config["car_in_relative"][args.camera_name]
            abs_pts = [(int(p[0] * img_w), int(p[1] * img_h)) for p in rel_pts]
            active_polygon = Polygon(abs_pts)
        else:
            # Fallback to the static legacy `car_in`
            if "car_in" in config and args.camera_name in config["car_in"]:
                active_polygon = Polygon(eval(config["car_in"][args.camera_name]))

    # Determine cropping area based on active ROI
    crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, img_w, img_h
    if active_polygon is not None:
        min_x, min_y, max_x, max_y = active_polygon.bounds
        crop_x1 = max(0, int(min_x))
        crop_y1 = max(0, int(min_y))
        crop_x2 = min(img_w, int(max_x))
        crop_y2 = min(img_h, int(max_y))
        # Ensure valid crop dimensions
        if crop_x2 > crop_x1 and crop_y2 > crop_y1:
            crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            crop_img = image # fallback to full image if polygon is invalid
    else:
        crop_img = image

    dt_boxes, dt_confidences, _ = get_bbox(crop_img, dt_net, dt_ln, config["models"]["number_plate_threshold"])
    dt_list = []
    dt_conf = []
    if dt_boxes is not None:
        for index, b in enumerate(dt_boxes):
            (l, t, w, h) = b[:4]
            # Adjust coordinates back to the original full-frame reference
            l += crop_x1
            t += crop_y1

            if l < 0:
                w = w + l
                l = 0
            if t < 0:
                h = h + t
                t = 0

            if active_polygon is not None:
                # Check if the license plate center falls within the polygon
                # BUG FIX: dynamically tracking actual bounding box's center
                cx = l + (w / 2.0)
                cy = t + (h / 2.0)
                
                if active_polygon.contains(Point(cx, cy)):
                    logging.info("Car and Number Plate is detected in this camera")
                    dt_list.append([(l, t), (l + w, t + h)])
                    dt_conf.append(dt_confidences[index])
                else:
                    logging.debug("Car and Number Plate is detected in this camera outside the region")
            else:
                logging.info("Car and Number Plate is detected in this camera")
                dt_list.append([(l, t), (l + w, t + h)])
                dt_conf.append(dt_confidences[index])
        return dt_list, dt_conf
    return None, None

###--------------------Ping api status is alive ...--------------------------##

# def ping_request():    
#     ping_id = config["ping_id"][args.camera_name]["uuid"]
#     ping_api = f"https://cronbeat.katomaran.tech/ping/{ping_id}" 
#     if config["ping_id"][args.camera_name]["status"] is True:
#         try:            
#             ping_response = requests.post(url=ping_api , timeout=10)
#             logging.info("PING API RESPONSE {}".format(ping_response.status_code))
#         except requests.RequestException as e:
#         # Log ping failure here...
#             logging.debug("PING API FAILED: {}".format(e),exc_info=1)

# ping_request()

# scheduler = BackgroundScheduler(timezone="Singapore")
# scheduler.add_job(ping_request,'interval', seconds=300)
# scheduler.start()

schedule.every().day.at("23:59:59").do(create_plot,config["camera_numberplate_path"][args.camera_name])
logging.info("Camera setup confgiuration is dones")
if 'application_type' in config.keys():
    if config['application_type'][args.camera_name] == 'resident':
        schedule.every().hour.at(":00").do(owner_data,config=config)
        owner_data(config)
        


def box_draw(image_list, bbox_list, rg_net, rg_ln, labels, dt_net, dt_ln, lp_confss):
    #  ------------------------------------------
    #   Initialising variable for every occurences
    #  ------------------------------------------
    image_len = -1
    plate_results = list()
    infos = list()
    full_image = None

    #  ------------------------------------------
    #   If only one detection is occured, we need to perform detection for other images.
    #  ------------------------------------------

    if len(bbox_list) == 1:
        for i in range(len(image_list) - 1, 0, -1):
            lp_detection_value, lp_conf = lp_detection(image_list[i], dt_net, dt_ln)
            if lp_detection_value is not None and len(lp_detection_value) > 0:
                bbox_list += lp_detection_value
                lp_confss += lp_conf
                bbox_list = rect_points(bbox_list, i + 1)
                break

    #  ------------------------------------------
    #   For testing purpose
    #  ------------------------------------------

    if config["testing"]["status"]:
        global tester, lp_thread_list
        if len(lp_thread_list) > 0:
            if lp_thread_list[0].isAlive() is False:
                del (lp_thread_list[0])
        plate_image_list = []
        annotation_list = []
    logging.info(f"Bbox_list  :  {bbox_list}")

    #  ------------------------------------------
    #   OCR operation is done here
    #  ------------------------------------------

    for image, box in zip(image_list, bbox_list):
        number_plate_image = image[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        rg_boxes, rg_confidences, rg_classids = get_bbox(number_plate_image, rg_net, rg_ln, config["models"]["ocr_threshold"])
        # rg_boxes, rg_confidences, rg_classids = get_bbox_openvino(number_plate_image, 0.5, "ocr")

        #   If rg_boxes is none, there is no detected results

        if rg_boxes is not None:

            #  ------------------------------------------
            #   This will store the number plate for the database
            #  ------------------------------------------

            varrr = len(rg_boxes)
            if varrr >= image_len:
                image_len = varrr
                lp_image = number_plate_image
                full_image = image
            info = []
            for i in range(len(rg_classids)):
                info.append((labels[rg_classids[i]], rg_boxes[i], rg_confidences[i]))
            if config["testing"]["status"]:
                plate_image_list.append(number_plate_image)
                annotation_list.append(info)
            if len(info) > 0:
                logging.debug(f" Sort rect - {info}")
                info = sort_rect(info)
                infos.append(info)
            plate_number = ''
            for item in info:
                plate_number += item[0]
            plate_results.append(plate_number)

    if config["testing"]["status"]:
        l_t = Thread(target=write_list,
                     args=(plate_image_list, str(tester), config["testing"]["lp_images_folder"], annotation_list))
        lp_thread_list.append(l_t)
        l_t.start()

    #  ------------------------------------------
    #   This will consolidate the results occured in all images
    #  ------------------------------------------

    if len(plate_results) > 0:
        plate_results = [x for x in plate_results if x.strip() != ""]
        if not plate_results:
             logging.info("All plate results are empty")
             return
        logging.info(f"Plate_results  :  {plate_results}")
        logging.debug(f"Infos - {infos}")
        consolidated, checksum_valid = consolidate_ocr_results(plate_results, checksum_exclude)
        consolidated = consolidated[0]
        if not consolidated is None:
            if len(consolidated) > 0 and consolidated[0] in ['W', 'B', 'M', 'N', 'J']:
                checksum_valid = 2
            text_info = []
            plate_results_without_cs = [k[:-1] for k in plate_results]
            try:
                for plate in plate_results:
                    logging.debug(f"consolidated - {consolidated}, plate - {plate}")
                    if plate[:-1] == consolidated[:-1]:
                        logging.debug(f"consolidated - {consolidated[:-1]}, plate_results_without_cs - {plate_results_without_cs}, checksum_valid - {checksum_valid}")
                        text_info = infos[plate_results_without_cs.index(consolidated[:-1])] + [checksum_valid]
                        break
            except Exception as ve:
                logging.info(f"Info Error  : {ve}", exc_info=1)
            logging.info(f"Consolidated  :  {consolidated} {text_info}")

            n = Thread(target=TextProcess_var.text_process, args=(lp_image, consolidated, text_info, lp_confss, full_image))
            n.start()
        else:
            logging.info("Check with Plate results")
            logging.info("\n\n\n\n\n")
    else:
        logging.info("Text not detected")

#  ------------------------------------------
#   Asynchronous OCR Worker
#  ------------------------------------------

ocr_queue = Queue(maxsize=config.get("OCR_QUEUE_MAXSIZE", 3))

def ocr_worker(rg_net, rg_ln, labels, dt_net, dt_ln):
    while True:
        try:
            # Get detection data from queue
            data = ocr_queue.get()
            if data is None: # Shutdown signal
                break
                
            img_list, bbox_list, lp_confss_copy = data
            
            # Execute batch OCR
            box_draw(img_list, bbox_list, rg_net, rg_ln, labels, dt_net, dt_ln, lp_confss_copy)
            
            # Explicitly clear objects to free memory
            del img_list
            del bbox_list
            del lp_confss_copy
            
            # Signal task completion
            ocr_queue.task_done()
            
        except Exception as e:
            logging.error(f"Error in ocr_worker: {e}", exc_info=1)

def camera_main():
    #  ------------------------------------------
    #    For the detection and recognition, the models are loaded here.
    #  ------------------------------------------
    twilio_counter = 0
    if 'twilio' in config.keys() and 'application_type' in config.keys():
        if config['twilio']['status'] is True and config['application_type'][args.camera_name] == 'resident' or (config['twilio']['status'] is True and config["sms_status"][args.camera_name] is True and config['application_type'][args.camera_name] == 'normal'):
            from twilio.rest import Client
            from twilio.http.http_client import TwilioHttpClient
            account_sid = config['twilio']['account_sid']
            auth_token  = config['twilio']['auth_token']
            phone_numbers = config['twilio']['phone_numbers']
            twilio_count = config['twilio']['count_to_send']
            
    device_setting = config["models"].get("device", "auto")
    logging.info(f"Loading YOLO models with device: {device_setting}")
    dt_net, dt_ln = get_ln(None, config["models"]["number_plate_model"], device=device_setting)
    rg_net, rg_ln = get_ln(None, config["models"]["ocr_model"], device=device_setting)
    labels = config.get("labels", "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
    logging.info("Models loaded. Opening camera stream...")

    # Start the background OCR worker
    worker_t = Thread(target=ocr_worker, args=(rg_net, rg_ln, labels, dt_net, dt_ln), daemon=True)
    worker_t.start()

    #  ------------------------------------------
    #    Capturing the live feed (RTSP via ffmpeg when needed)
    #  ------------------------------------------

    cap = _open_camera_stream(config, args.camera_name)
    logging.info("Stream opened. Entering main loop.")

    #  ------------------------------------------
    #   Default variables for a single output for a single car
    #  ------------------------------------------

    car_detected = 0 # used for car tracking 
    car_crossed = 0 # used for car tracking 
    number_plate_images = list() # used for storing frames from camera
    rectangle_list = list() # used for storing detected lp rectangles
    lp_confs = [] # used for storing the lp confideces
    first_time = 0 # used for car tracking 

    time_track = {'try': {'start': None, 'state': None, 'loop': False},
                  'excep': {'start': None, 'state': None, 'loop': False}, 'present_state': None, 'message': None}
    last_no_frame_log = 0

    #  ------------------------------------------
    #   For testing purpose
    #  ------------------------------------------

    if config["testing"]["status"]:
        global tester, car_thread_list

    while True:
        
        try:
            
            schedule.run_pending()
            ret, image = cap.read()
            if not ret or image is None:
                if time.time() - last_no_frame_log >= 30:
                    logging.warning("No frame from camera (check camera_url and network). Skipping.")
                    last_no_frame_log = time.time()
                sleep(1)
                continue
            (h, w) = image.shape[:2]
            twilio_counter = 0
            if config["feed_cuts_notification"] is True:
                if time_track['try']['state'] is None and time_track['try']['loop'] is False and time_track['try'][
                    'start'] is None and time_track['present_state'] is None:
                    time_track['try']['state'] = True
                    time_track['present_state'] = False
                    time_track['excep']['loop'] = True
                    time_track['message'] = 'Initially host is detected and the ip camera is connected successfully'
                elif time_track['try']['loop'] is True:
                    if time_track['excep']['start'] is None:
                        time_track['excep']['start'] = None
                    time_track['try']['start'] = time.time()
                    time_track['try']['loop'] = False
                    time_track['excep']['loop'] = True
                    time_track['excep']['start'] = None
                elif time_track['try']['start'] is not None:
                    if (time.time() - time_track['try']['start']) // 60 == config['mail_time_delay']:
                        time_track['try']['state'] = True
                        time_track['try']['start'] = None
                        time_track['excep']['loop'] = True
                        time_track[
                            'message'] = f"The camera is continuously watched for {config['mail_time_delay'] // 60}hr and it is connected successfully"
                #  ------------------------------------------
                #   After this only, we know the feed is available for the detection purpose
                #  ------------------------------------------
                if config['outbound'] is True:
                    if time_track['try']['state'] is True and time_track['present_state'] is False:
                        logging.info("Successfully requested the ip camera")
                        try:
                            # if config["mail_status"] is True:
                            #     mail_res = requests.post(config['mail_server'],
                            #                             data={"camera_type": args.camera_name,
                            #                                 'site_name': config['site_name'],
                            #                                 'message': time_track['message'],
                            #                                 'application_name': "LNPR"})

                            #     logging.info(
                            #         f"Mail server hit for feed connectivity is done successfully with the response code of {mail_res.status_code}")
                            res = requests.put(config["db"]["state_change"].format(config["db"]["camera_id"][args.camera_name]),headers={"Authorization":config["db"]["state_auth_token"]},
                                            params={"status": "working"})
                            logging.info(f"Camera state  :  {res.json()['data']['state']}")

                            ## camera monitoring
                            payload = {}
                            payload['condo_name'] = config['site_name']
                            payload['app_name'] = "lnpr"
                            payload['cam_name'] = args.camera_name
                            payload['app_live'] = True
                            payload['cam_live'] = True 
                            payload['updated_time'] = strftime("%d-%m-%Y %H:%M:%S", custom_time())
                            logging.info(f"\nData to hit Camera Monitoring {payload}")
                            headers = {
                            'Content-Type': 'application/json'
                            }
                            payload = json.dumps(payload)
                            response = requests.post("http://va-healthchecks.katomaran.tech/monitor/update", headers=headers, data=payload)
                            if response.status_code == 200:
                                logging.info(f"\nCamera Monitoring hitted with {response.status_code} code and output from db is {response.json()}")
                            else:
                                logging.info(f"\n Camera Monitoring Failed with {response.status_code} code")
                        except Exception as e:
                            logging.info(f"Alert error  : {e}")


                        time_track['try']['state'] = False
                        time_track['present_state'] = True
                    elif time_track['try']['state'] is True and time_track['present_state'] is True:
                        time_track['try']['state'] = False

            #  ------------------------------------------
            #   We need to perform detection once in a specified camera fps, so for that we need a following condition.
            #  ------------------------------------------

            if cap.get(1) % config["camera_fps"] == 0:
                car_detected = 0
                lp_detection_value, lp_conf = lp_detection(image, dt_net, dt_ln)
                if lp_detection_value is not None and len(lp_detection_value) > 0:
                    car_detected = 1
                    rectangle_list += lp_detection_value
                    lp_confs += lp_conf

            #  ------------------------------------------
            #   This will continuously collect the images once the licence plate is detected
            #  ------------------------------------------

            if car_detected == 1 and first_time == 0:
                number_plate_images.append(image)
                max_batch = config.get("max_plate_batch_size", config["camera_fps"] * config["batch_size"])
                if len(number_plate_images) >= max_batch:
                    if config["testing"]["status"]:
                        if len(car_thread_list) > 0:
                            # print("Thread 0 Status  : ", thread_list[0].isAlive())
                            if car_thread_list[0].isAlive() is False:
                                del (car_thread_list[0])
                        c_t = Thread(target=write_list,
                                     args=(
                                         number_plate_images.copy(), str(tester),
                                         config["testing"]["car_image_folder"]))
                        car_thread_list.append(c_t)
                        c_t.start()
                    logging.info(f"Length of image list  {len(number_plate_images)}")
                    box = rect_points(rectangle_list, config["camera_fps"])
                    # box_draw(number_plate_images, box, rg_net, rg_ln, labels, dt_net, dt_ln, lp_confs.copy())
                    
                    # Offload to background worker
                    if not ocr_queue.full():
                        ocr_queue.put((number_plate_images.copy(), box, lp_confs.copy()))
                    else:
                        logging.warning("OCR Queue is full! Skipping this detection to prevent memory overflow.")
                        
                    # Immediately clear main thread lists to free RAM
                    number_plate_images.clear()
                    rectangle_list.clear()
                    lp_confs.clear()
                    
                    first_time = 1
                car_crossed = 1

            #  ------------------------------------------
            #   This will create an bbox from the collected bbox by using a mathematical formula to reduce the cpu
            #   usage and it fed to an OCR process and delete the images that we have collected earlier.
            #  ------------------------------------------

            elif car_detected == 0 and car_crossed == 1:
                car_crossed = 0
                if first_time == 0:
                    if config["testing"]["status"]:
                        w_t = Thread(target=write_list,
                                     args=(
                                         number_plate_images.copy(), str(tester),
                                         config["testing"]["car_image_folder"]))
                        w_t.start()
                    logging.info(f"Length of image list  {len(number_plate_images)}")
                    box = rect_points(rectangle_list, config["camera_fps"])
                    # box_draw(number_plate_images, box, rg_net, rg_ln, labels, dt_net, dt_ln, lp_confs.copy())
                    
                    # Offload to background worker
                    if not ocr_queue.full():
                        ocr_queue.put((number_plate_images.copy(), box, lp_confs.copy()))
                    else:
                        logging.warning("OCR Queue is full! Skipping this detection to prevent memory overflow.")
                        
                first_time = 0
                logging.info(f"Total image list  {len(number_plate_images)}")
                if config["testing"]["status"]:
                    tester += 1
                del (number_plate_images[:])
                del (rectangle_list[:])
                del (lp_confs[:])

            #  ------------------------------------------
            #   When show video is True, it gives the video output
            #  ------------------------------------------

            if config["show_video"] is True:
                cv2.namedWindow("Visitor Camera", cv2.WINDOW_NORMAL)
                cv2.imshow("Visitor Camera", image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break

        except Exception as e:
            '''
                Exception is raised when live feed cuts
            '''
            if config["feed_cuts_notification"] is True:
                if time_track['excep']['state'] is None and time_track['excep']['loop'] is False and time_track['excep'][
                    'start'] is None and time_track['present_state'] is None:
                    time_track['excep']['state'] = True
                    time_track['try']['loop'] = True
                    time_track['present_state'] = True
                    time_track['message'] = 'Initially the camera is unreachable'
                elif time_track['excep']['loop'] is True:
                    if time_track['try']['start'] is not None:
                        time_track['try']['start'] = None
                    time_track['excep']['start'] = time.time()
                    time_track['excep']['loop'] = False
                    time_track['try']['loop'] = True
                    time_track['try']['start'] = None
                elif time_track['excep']['start'] is not None:
                    if (time.time() - time_track['excep']['start']) // 60 == config['mail_time_delay']:
                        time_track['excep']['state'] = True
                        time_track['excep']['start'] = None
                        time_track['try']['loop'] = True
                        time_track[
                            'message'] = f"The camera is continuously watched for {config['mail_time_delay'] // 60}hr and there is no feed from the host"

                if config['outbound'] is True:
                    if time_track['excep']['state'] is True and time_track['present_state'] is True:
                        #    This is for database side to show the status of the live feed
                        try:
                            # if config["mail_status"] is True:
                            #     mail_res = requests.post(config['mail_server'],
                            #                             data={"camera_type": args.camera_name,
                            #                                 'site_name': config['site_name'],
                            #                                 'message': time_track['message'],
                            #                                 'application_name': "LNPR"})
                            #     logging.info(
                            #         f"Mail server hit for feed lost is done successfully with the response code of {mail_res.status_code}")
                            res = requests.put(
                                config["db"]["state_change"].format(config["db"]["camera_id"][args.camera_name]),headers={"Authorization":config["db"]["state_auth_token"]},
                                params={"status": "lost_connectivity"})
                            logging.info(f"Camera state  :  {res.json()['data']['state']}")

                            ## camera monitoring
                            payload = {}
                            payload['condo_name'] = config['site_name']
                            payload['app_name'] = "lnpr"
                            payload['cam_name'] = args.camera_name
                            payload['app_live'] = True
                            payload['cam_live'] = False
                            payload['updated_time'] = strftime("%d-%m-%Y %H:%M:%S", custom_time()) 
                            logging.info(f"\nData to hit Camera Monitoring {payload}")
                            headers = {
                            'Content-Type': 'application/json'
                            }
                            payload = json.dumps(payload)
                            response = requests.post("http://va-healthchecks.katomaran.tech/monitor/update", headers=headers, data=payload)
                            if response.status_code == 200:
                                logging.info(f"\nCamera Monitoring hitted with {response.status_code} code and output from db is {response.json()}")
                            else:
                                logging.info(f"\n Camera Monitoring Failed with {response.status_code} code")


                            time_track['excep']['state'] = False
                            time_track['present_state'] = False
                        except Exception as al:
                            logging.info(f"Alert error  : {al}", exc_info=1)
                    elif time_track['excep']['state'] is True and time_track['present_state'] is False:
                        time_track['excep']['state'] = False
            #   This will log the error to a file and recapture the camera to be in live

            logging.info(f"Error  : {e}", exc_info=1)
            if 'twilio' in config.keys() and 'application_type' in config.keys():
                if config['twilio']['status'] is True and config['application_type'][args.camera_name] == 'resident' or (config['twilio']['status'] is True and config["sms_status"][args.camera_name] is True and config['application_type'][args.camera_name] == 'normal'):
                    twilio_counter += 1
                    if twilio_counter % twilio_count == 0:
                        twilio_http_client = TwilioHttpClient(timeout=10)
                        client = Client(account_sid, auth_token,http_client=twilio_http_client)

                        for phone_number in phone_numbers:
                            message = client.messages.create(
                            body="{} - {} - lost connection".format(config['site_name'], args.camera_name),
                            to=phone_number,
                            from_=config['twilio']['from_number'])
                            logging.info(f"Message ID - {message.sid}")
            cap.release()
            sleep(2)
            cap = _open_camera_stream(config, args.camera_name)


if __name__ == '__main__':
    camera_main()
