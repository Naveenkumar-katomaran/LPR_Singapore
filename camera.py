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
from utils.tracker import PlateTracker

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

# Global state for cumulative validation feedback
ocr_queue = Queue()
validated_ids = set()
validated_lock = threading.Lock()


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
#   Perfect Logging System Initialization
#  ------------------------------------------

class CameraIdFilter(log.Filter):
    """Filter to inject camera_id into log records."""
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
    def filter(self, record):
        record.camera_id = self.camera_id
        return True

class CustomFormatter(log.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;21m"
    reset = "\x1b[0m"
    format_str = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(camera_id)s] - %(message)s"

    FORMATS = {
        log.DEBUG: grey + format_str + reset,
        log.INFO: green + format_str + reset,
        log.WARNING: yellow + format_str + reset,
        log.ERROR: red + format_str + reset,
        log.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        # Ensure camera_id exists on record even if filter missed it
        if not hasattr(record, 'camera_id'):
            record.camera_id = "SYSTEM"
        formatter = log.Formatter(log_fmt, datefmt='%d-%b-%y %H:%M:%S')
        formatter.converter = custom_time
        return formatter.format(record)

def setup_perfect_logging(config, camera_name):
    camera_id = config["db"]["camera_id"][camera_name]
    log_level = log.DEBUG if config.get("verbose") else log.INFO
    
    # Root logger config
    root_logger = log.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to prevent duplicate logs
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    
    # root_logger.addFilter(CameraIdFilter(camera_id)) # Not needed on root if on handlers
    
    # 1. Console Handler (Colorized)
    console_handler = log.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    console_handler.addFilter(CameraIdFilter(camera_id))
    root_logger.addHandler(console_handler)

    # 2. File Handler (Rotating)
    log_dir = config.get('log_path', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, camera_name)
    file_handler = lh.TimedRotatingFileHandler(file_path, 'midnight', 1, backupCount=7)
    file_handler.suffix = "%Y-%m-%d"
    file_handler.addFilter(CameraIdFilter(camera_id))
    
    file_formatter = log.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] [%(camera_id)s] - %(message)s", datefmt='%d-%b-%y %H:%M:%S')
    file_formatter.converter = custom_time
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Disable noisy loggers
    log.getLogger('matplotlib').setLevel(log.WARNING)
    log.getLogger('ultralytics').setLevel(log.WARNING)

    return log.getLogger(args.camera_name)

# Initialize the perfect logging
logging = setup_perfect_logging(config, args.camera_name)

#  ------------------------------------------
#   Global System Configurations
#  ------------------------------------------
if not "number_plate_threshold" in config["models"].keys():
    config["models"]["number_plate_threshold"] = 0.8
if not "ocr_threshold" in config["models"].keys():
    config["models"]["ocr_threshold"] = 0.5

checksum_exclude = ['W', 'B', 'M', 'N', 'J', 'CC']
if "checksum_exclude" in config.keys():
    checksum_exclude = config['checksum_exclude']

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
        # Add 100px leeway buffer to ensure the full plate is seen even if 
        # its center is just crossing the ROI line.
        crop_x1 = max(0, int(min_x) - 100)
        crop_y1 = max(0, int(min_y) - 100)
        crop_x2 = min(img_w, int(max_x) + 100)
        crop_y2 = min(img_h, int(max_y) + 100)
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
                # logging.info("Car and Number Plate is detected in this camera")
                    dt_list.append([(l, t), (l + w, t + h)])
                    dt_conf.append(dt_confidences[index])
                else:
                    logging.debug("Car and Number Plate is detected in this camera outside the region")
            else:
                # logging.info("Car and Number Plate is detected in this camera")
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
        


def box_draw(image_list, bbox_list, rg_net, rg_ln, labels, dt_net, dt_ln, lp_confss, obj_id=None, best_global_frame=None):
    #  ------------------------------------------
    #   Initialising variable for every occurences
    #  ------------------------------------------
    image_len = -1
    plate_results = list()
    infos = list()
    full_image = None

    #  ------------------------------------------
    #   Dense crop augmentation — works for ALL N detection points.
    #   Replaces the old YOLO re-inference fallback (which only handled
    #   the 1-detection edge-case and was extremely CPU-heavy).
    #
    #   Strategy:
    #     1. Interpolate ghost bboxes between every consecutive pair of
    #        known detections using interpolate_bboxes() — pure numpy,
    #        near-zero CPU cost, no model calls.
    #     2. Re-crop each ghost bbox directly from best_global_frame
    #        (already in memory — free).
    #     3. Prepend these augmented crops so the OCR loop below sees
    #        both originals AND the new ghost crops in one unified pass.
    #  ------------------------------------------

    augmented_crops = []
    if best_global_frame is not None and len(bbox_list) >= 1:
        dense_bboxes = interpolate_bboxes(bbox_list, num_intermediate=2)
        fh, fw = best_global_frame.shape[:2]
        logging.debug(
            f"[OCR] ID:{obj_id} BBox densification: {len(bbox_list)} detections "
            f"→ {len(dense_bboxes)} dense boxes (including {len(dense_bboxes) - len(bbox_list)} ghost crops)"
        )
        for bbox in dense_bboxes:
            x1 = max(0,  int(bbox[0][0]) - 15)
            y1 = max(0,  int(bbox[0][1]) - 15)
            x2 = min(fw, int(bbox[1][0]) + 15)
            y2 = min(fh, int(bbox[1][1]) + 15)
            if x2 > x1 and y2 > y1:
                augmented_crops.append(best_global_frame[y1:y2, x1:x2].copy())

    # Merge: originals first (highest priority for quality scoring), then ghost crops
    combined_images = list(image_list) + augmented_crops

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
    logging.info(f"[OCR] ID:{obj_id} Processing {len(combined_images)} crops for OCR "
                 f"({len(image_list)} original + {len(augmented_crops)} ghost)")

    # ------------------------------------------
    #  OCR operation is done here (Batch Inference)
    # ------------------------------------------
    best_quality_score = -1.0
    plate_data = [] # List of (text, confidence)
    
    # Selection Fallback: Initialize with the first crop
    if len(combined_images) > 0:
        lp_image = combined_images[0]
        full_image = best_global_frame if best_global_frame is not None else combined_images[0]
    else:
        lp_image = None
        full_image = None

    # Perform Batch Inference for massive GPU speedup
    batch_results = get_bbox_batch(combined_images, rg_net, rg_ln, config["models"]["ocr_threshold"])

    for i, (rg_boxes, rg_confidences, rg_classids) in enumerate(batch_results):
        number_plate_image = combined_images[i]
        
        if rg_boxes is not None:
            char_count = len(rg_boxes)
            avg_conf = sum(rg_confidences) / char_count if char_count > 0 else 0
            
            # Quality Score: CharCount is weight 10, AvgConf is weight 1
            quality_score = (char_count * 10) + avg_conf
            
            if quality_score > best_quality_score:
                best_quality_score = quality_score
                lp_image = number_plate_image
                full_image = best_global_frame if best_global_frame is not None else number_plate_image
                
            info = []
            for j in range(len(rg_classids)):
                info.append((labels[rg_classids[j]], rg_boxes[j], rg_confidences[j]))
            
            if config["testing"]["status"]:
                plate_image_list.append(number_plate_image)
                annotation_list.append(info)
                
            if len(info) > 0:
                info = sort_rect(info)
                infos.append(info)
                plate_number = "".join([item[0] for item in info])
                plate_results.append(plate_number)
                plate_data.append((plate_number, avg_conf))

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
        
        # Read validation override from config
        v_indian = config["models"].get("validate_indian_plate", True)
        consolidated_all, checksum_valid = consolidate_ocr_results(plate_data, checksum_exclude, validate_indian_plate=v_indian)
        
        consolidated = consolidated_all[0]
        group_summary = consolidated_all[1]

        if consolidated is not None:
            # Mark ID as validated globally to skip further intermediate batches
            if obj_id is not None:
                with validated_lock:
                    validated_ids.add(obj_id)
            
            if len(consolidated) > 0 and consolidated[0] in ['W', 'B', 'M', 'N', 'J','T']:
                checksum_valid = 2
            text_info = []
            plate_results_without_cs = [k[:-1] for k in plate_results]
            try:
                for plate in plate_results:
                    if plate[:-1] == consolidated[:-1]:
                        text_info = infos[plate_results_without_cs.index(consolidated[:-1])] + [checksum_valid]
                        break
            except Exception as ve:
                logging.debug(f"Info extraction error: {ve}")

            logging.info(f"[OCR] ID:{obj_id} Final Result: {consolidated}")

            # Send to API (Pass group_summary here for detailed JSON logging)
            n = Thread(target=TextProcess_var.text_process, 
                       args=(lp_image, consolidated, group_summary, lp_confss, full_image, obj_id))
            n.start()
        else:
            logging.info(f"[OCR] ID:{obj_id} Batch Rejected - No valid Indian plate found. Continuing collection...")
    else:
        logging.info("Text not detected")

#  ------------------------------------------
#   Asynchronous OCR Worker
#  ------------------------------------------

ocr_queue = Queue(maxsize=config.get("OCR_QUEUE_MAXSIZE", 3))

def ocr_worker(rg_net, rg_ln, labels, dt_net, dt_ln):
    logging.info("OCR Worker started successfully.")
    while True:
        try:
            # Get detection data from queue
            data = ocr_queue.get()
            if data is None: # Shutdown signal
                logging.info("OCR Worker received shutdown signal.")
                break
                
            obj_id, img_list, bbox_list, lp_confss_copy, best_global_frame, is_final = data
            
            # EARLY EXIT: If already validated and it's not the final check, skip GPU work
            with validated_lock:
                if obj_id in validated_ids and not is_final:
                    logging.debug(f"[OCR] ID:{obj_id} Skipping intermediate batch (Already validated)")
                    ocr_queue.task_done()
                    continue

            # Execute batch OCR
            try:
                box_draw(img_list, bbox_list, rg_net, rg_ln, labels, dt_net, dt_ln, lp_confss_copy, obj_id=obj_id, best_global_frame=best_global_frame)
            except Exception as e:
                logging.error(f"Error during box_draw processing: {e}", exc_info=1)
            
            # Cleanup on Exit
            if is_final:
                with validated_lock:
                    if obj_id in validated_ids: validated_ids.remove(obj_id)
            
            # Explicitly clear objects to free memory
            del img_list
            del bbox_list
            del lp_confss_copy
            
            # Signal task completion
            ocr_queue.task_done()
            
        except Exception as e:
            logging.error(f"Critical error in ocr_worker loop: {e}", exc_info=1)
            sleep(1) # Prevent rapid fire looping on failure
            
def validate_config(config, camera_name):
    """Ensure all required keys are present in config."""
    required_keys = ["models", "camera_fps", "batch_size", "outbound", "db"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    if camera_name not in config["db"]["camera_id"]:
        raise ValueError(f"Camera name '{camera_name}' not found in config[db][camera_id]")
    
    logging.info("Configuration validated successfully.")

def handle_camera_monitoring(config, camera_name, custom_time_func):
    """Handle periodic camera monitoring heartbeats."""
    try:
        payload = {
            'condo_name': config['site_name'],
            'app_name': "lnpr",
            'cam_name': camera_name,
            'app_live': True,
            'cam_live': True,
            'updated_time': strftime("%d-%m-%Y %H:%M:%S", custom_time_func())
        }
        logging.info(f"\nSending Camera Monitoring Heartbeat: {payload}")
        headers = {'Content-Type': 'application/json'}
        response = requests.post("http://va-healthchecks.katomaran.tech/monitor/update", 
                                headers=headers, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            logging.info(f"Heartbeat success: {response.json()}")
        else:
            logging.warning(f"Heartbeat failed with code {response.status_code}")
    except Exception as e:
        logging.error(f"Monitoring heartbeat error: {e}")

def offload_tracks_to_queue(completed_tracks, ocr_queue, camera_name, config):
    """Offload finalized tracks after zone-to-zone sequence validation."""
    dir_cfg = config.get("direction_config", {}).get(camera_name, {})
    
    # If direction filtering is disabled or not using zone_transition, process all
    if not dir_cfg.get("enabled", True) or dir_cfg.get("method") != "zone_transition":
        for track_data in completed_tracks:
            # track_data: obj_id, images, bboxes, confs, zone_history, is_final, best_full, displacement_vector
            if not ocr_queue.full():
                # Pass is_final (6th element) to the worker
                ocr_queue.put(track_data[:4] + (track_data[6], track_data[5]))
        return

    expected_mode = dir_cfg.get("mode", "entry") # "entry" expects B -> A

    for track_data in completed_tracks:
        # track_data: obj_id, images, bboxes, confs, zone_history, is_final, best_full, displacement_vector
        obj_id, images, bboxes, confs, zone_history, is_final, best_full_frame, disp_vec = track_data

        is_valid = False
        
    # 1. Master Direction Vector (from Zone B to Zone A)
    # We calculate this once per batch to avoid redundant math
    master_vec = (0, 0)
    if "zone_a" in dir_cfg and "zone_b" in dir_cfg:
        w = config.get("stream_width", 1920)
        h = config.get("stream_height", 1080)
        def get_poly_centroid(pts):
            pts = np.array(pts)
            return np.mean(pts, axis=0)
        cA = get_poly_centroid(dir_cfg["zone_a"])
        cB = get_poly_centroid(dir_cfg["zone_b"])
        # Convert relative to absolute pixels for accurate dot product mapping
        cA = (cA[0] * w, cA[1] * h)
        cB = (cB[0] * w, cB[1] * h)
        # Entry vector: B -> A
        master_vec = (cA[0] - cB[0], cA[1] - cB[1])
        logging.debug(f"[Direction] Master Vector (B->A): {master_vec}")

    for track_data in completed_tracks:
        # track_data: obj_id, images, bboxes, confs, zone_history, is_final, best_full, displacement_vector
        obj_id, images, bboxes, confs, zone_history, is_final, best_full_frame, disp_vec = track_data

        is_valid = False
        
        # 2. Angular Variance Enforcement
        target_vec = master_vec if expected_mode == "entry" else (-master_vec[0], -master_vec[1])
        dot = disp_vec[0] * target_vec[0] + disp_vec[1] * target_vec[1]
        
        import math
        # Calculate total pixel displacement from first detection to last
        disp_mag = math.hypot(disp_vec[0], disp_vec[1])
        target_mag = math.hypot(target_vec[0], target_vec[1])
        
        if disp_mag > 0 and target_mag > 0:
            cos_theta = dot / (disp_mag * target_mag)
            cos_theta = max(min(cos_theta, 1.0), -1.0) # Clamp against precision errors
            angle = math.degrees(math.acos(cos_theta))
        else:
            angle = 180.0 # Strict failure for completely stationary detections
            
        max_angle = dir_cfg.get("max_angle_deviation", 85.0)
        
        # A valid journey MUST fall within the configurable acceptance arc
        is_moving_correctly = (angle <= max_angle)

        # Remove duplicates while preserving history order (e.g., ['A', 'A', 'B'] -> ['A', 'B'])
        clean_seq = [z for z in zone_history if z in ['A', 'B']]
        
        if is_moving_correctly and len(clean_seq) > 0:
            if expected_mode == "entry":
                # ENTRY (B -> A): If both zones were hit, it MUST end in A.
                if 'A' in clean_seq and 'B' in clean_seq:
                    if clean_seq[-1] == 'A':
                        is_valid = True
                else: 
                    # Only one zone hit, rely purely on the strong projection vector check
                    is_valid = True
            else: 
                # EXIT (A -> B): If both zones were hit, it MUST end in B.
                if 'A' in clean_seq and 'B' in clean_seq:
                    if clean_seq[-1] == 'B':
                        is_valid = True
                else:
                    is_valid = True
            
        if is_valid:
            if not ocr_queue.full():
                # Format: obj_id, images, bboxes, confs, best_full_frame, is_final
                ocr_queue.put((obj_id, images, bboxes, confs, best_full_frame, is_final))
            
            if is_final:
                logging.info(f"[Direction] ID:{obj_id} Validated | Seq:{clean_seq} | Angle:{angle:.1f}° (Mag:{disp_mag:.1f}px)")
        else:
            if is_final:
                logging.info(f"[Direction] ID:{obj_id} Discarded | Seq:{clean_seq} | Moving Correctly:{is_moving_correctly} (Angle:{angle:.1f}°)")

def camera_main():
    # Validation
    try:
        validate_config(config, args.camera_name)
    except Exception as e:
        logging.critical(f"Config Validation Failed: {e}")
        return

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
    logging.info(f"=== {args.camera_name.upper()} STARTUP COMPLETE ===")
    logging.info(f"FPS: {config['camera_fps']} | Tracker Max Age: {config['TRACKER_MAX_AGE']}")
    #  ------------------------------------------
    #   Initialising Plate Tracker for multi-object handling
    #  ------------------------------------------
    tracker = PlateTracker(
        iou_threshold=config.get("IOU_THRESHOLD", 0.3),
        max_age=config.get("TRACKER_MAX_AGE", 10),
        max_batch_size=config.get("max_plate_batch_size", config["camera_fps"] * config["batch_size"]),
        distance_threshold=config.get("DISTANCE_THRESHOLD", 800),
        distance_scale_factor=config.get("DISTANCE_SCALE_FACTOR", 2.5)
    )

    zone_a_poly, zone_b_poly = None, None
    zone_initialized = False

    time_track = {'try': {'start': None, 'state': None, 'loop': False},
                  'excep': {'start': None, 'state': None, 'loop': False}, 'present_state': None, 'message': None}


    #  ------------------------------------------
    #   For testing purpose
    #  ------------------------------------------

    if config["testing"]["status"]:
        global tester, car_thread_list

    camera_failure_count = 0
    while True:
        
        try:
            
            schedule.run_pending()
            ret, frame = cap.read()
            if not ret or frame is None:
                camera_failure_count += 1
                logging.warning(f"No frame from camera (failure count: {camera_failure_count}). Skipping.")
                
                if camera_failure_count >= 5:
                    logging.warning(f"Consecutive failures ({camera_failure_count}) reached threshold. Attempting reconnection...")
                    cap.release()
                    sleep(2)
                    cap = _open_camera_stream(config, args.camera_name)
                    # Reset counter after reconnection attempt to avoid rapid-fire reconnects
                    camera_failure_count = 0
                
                sleep(1)
                continue
            
            # Reset failure count on successful frame read
            camera_failure_count = 0
            
            # CRITICAL FIX: We MUST copy the frame. 
            # Camera buffers are reused; without a copy, the 'Exit' frame overwrites 
            # the 'Center' frames in the tracker, causing 'half-plate' crops.
            image = frame.copy()
            (h, w) = image.shape[:2]

            # Lazy-initialize Zones once dimensions are known
            if not zone_initialized:
                dir_cfg = config.get("direction_config", {}).get(args.camera_name, {})
                if dir_cfg.get("method") == "zone_transition":
                    try:
                        if "zone_a" in dir_cfg and "zone_b" in dir_cfg:
                            abs_a = [(p[0]*w, p[1]*h) for p in dir_cfg["zone_a"]]
                            abs_b = [(p[0]*w, p[1]*h) for p in dir_cfg["zone_b"]]
                            zone_a_poly = Polygon(abs_a)
                            zone_b_poly = Polygon(abs_b)
                            logging.info(f"[Zones] Initialized A & B for {args.camera_name}")
                    except Exception as ze:
                        logging.error(f"[Zones] Initialization failed: {ze}")
                zone_initialized = True

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
                #   State and Health Check APIs have been disabled as per user request
                #  ------------------------------------------

            #  ------------------------------------------
            #   We need to perform detection once in a specified camera fps, so for that we need a following condition.
            #  ------------------------------------------

            if cap.get(1) % config["camera_fps"] == 0:
                dt_list, dt_conf = lp_detection(image, dt_net, dt_ln)
                
                # Update tracker with current detections (association)
                if dt_list is None: dt_list = []
                if dt_conf is None: dt_conf = []
                
                completed_tracks = tracker.update(dt_list, dt_conf, image)
            else:
                # Continuous collection: Add the frame to all active tracks
                completed_tracks = tracker.add_frame_to_all(image)
            
            # Update Zone History for active tracks
            if zone_a_poly or zone_b_poly:
                for obj in tracker.objects:
                    if not obj.has_ended:
                        bbox = obj.bboxes[-1]
                        centroid = Point((bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2)
                        
                        if zone_a_poly and zone_a_poly.contains(centroid):
                            obj.update_zone_history('A')
                        elif zone_b_poly and zone_b_poly.contains(centroid):
                            obj.update_zone_history('B')
                
            # Offload completed tracks to the OCR queue
            offload_tracks_to_queue(completed_tracks, ocr_queue, args.camera_name, config)

            #  ------------------------------------------
            #   Inference Video Overlay & Display
            #  ------------------------------------------
            if config.get("draw_inference", False):
                # Draw Zones
                if zone_a_poly or zone_b_poly:
                    overlay = image.copy()
                    if zone_a_poly:
                        pts = np.array(zone_a_poly.exterior.coords, np.int32)
                        cv2.fillPoly(overlay, [pts], (255, 0, 0)) # Blue
                        cv2.putText(image, "GATE (A)", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if zone_b_poly:
                        pts = np.array(zone_b_poly.exterior.coords, np.int32)
                        cv2.fillPoly(overlay, [pts], (0, 0, 255)) # Red
                        cv2.putText(image, "APPROACH (B)", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

                # Draw active tracks from the tracker
                # This will "burn-in" the boxes to the frames stored in the tracker
                for obj in tracker.objects:
                    if not obj.has_ended and obj.missing_frames < 2:
                        # Get last bbox
                        bbox = obj.bboxes[-1] # format: [(x1, y1), (x2, y2)]
                        p1 = (int(bbox[0][0]), int(bbox[0][1]))
                        p2 = (int(bbox[1][0]), int(bbox[1][1]))
                        
                        # Draw yellow box for tracking
                        cv2.rectangle(image, p1, p2, (0, 255, 255), 3)

                        # Draw ID label
                        label = f"ID: {obj.obj_id}"
                        cv2.putText(image, label, (p1[0], p1[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                        
                        # Show current Zone status if in one
                        if obj.zone_history:
                            z_text = f"Zone: {obj.zone_history[-1]}"
                            cv2.putText(image, z_text, (p1[0], p1[1] - 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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

                # Exception state and health check APIs disabled as per user request.
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
