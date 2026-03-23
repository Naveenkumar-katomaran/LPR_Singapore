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
from utils.db import *
from utils.tracker import PlateTracker
import torch

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
import threading
import time
from threading import Thread
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from time import strftime, sleep
import schedule
import threading
from queue import Queue
from types import SimpleNamespace
import time

# Global state for display and validation
ocr_queue = Queue(maxsize=10)
validated_ids = set()
validated_lock = threading.Lock()
text_processors = {}

display_frames = {}
display_lock = threading.Lock()

#  ------------------------------------------
#   Command line interface settings
#  ------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_name', required=False, help='Optional: Specific camera name to run. If omitted, runs all.')
    parser.add_argument('--config_file', required=True, help='Configuration file is needed')
    return parser.parse_args()

#  ------------------------------------------
#   RTSP reader via ffmpeg
#  ------------------------------------------

class FFmpegRTSPReader:
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
            except Exception: pass
            self.proc = None
        # cmd = [
        #     "ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", self.url,
        #     "-f", "rawvideo", "-pix_fmt", "bgr24",
        #     "-s", "{}x{}".format(self.width, self.height), "-an", "-"
        # ]
        cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",          # ✅ same as ffplay
        "-fflags", "nobuffer",             # ✅ reduce latency
        "-flags", "low_delay",             # ✅ better for RTSP
        "-strict", "experimental",
        "-analyzeduration", "1000000",     # ✅ faster stream start
        "-probesize", "1000000",

        "-i", self.url,

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={self.width}:{self.height}",  # ⚠️ better than -s
        "-an",
        "-"
    ]

        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL
            )
        except Exception as e:
            log.error(f"FFmpeg launch failed: {e}")

    def read(self, timeout_sec=5):
        if not self.proc or self.proc.poll() is not None:
            return False, None
        size = self.width * self.height * 3
        try:
            ready, _, _ = select.select([self.proc.stdout], [], [], timeout_sec)
            if not ready: return False, None
            buf = self.proc.stdout.read(size)
        except Exception: return False, None
        if len(buf) != size: return False, None
        self._frame_index += 1
        frame = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame

    def get(self, prop):
        return self._frame_index if prop == 1 else 0

    def release(self):
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1)
            except Exception: pass
            self.proc = None

    def restart(self):
        log.info(f"[{self.url}] Restarting FFmpeg reader...")
        self.release()
        self._start()

def _open_camera_stream(config, camera_name):
    url = config["camera_url"][camera_name]
    if url.strip().lower().startswith("rtsp://"):
        w = config.get("stream_width", 1920)
        h = config.get("stream_height", 1080)
        return FFmpegRTSPReader(url, w, h)
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

def custom_time(*args):
    utc_dt = datetime.now(utc)
    my_tz = timezone("Asia/Kolkata")
#  ------------------------------------------
#   Logging Setup
#  ------------------------------------------

# Thread-local storage for camera context
log_context = threading.local()

class CameraIdFilter(log.Filter):
    """
    Filter that injects camera_id into log records and routes them
    to the correct file handler in multi-threaded mode.
    """
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        
    def filter(self, record):
        # Inject camera_id (used by formatter)
        record.camera_id = self.camera_id
        
        # Routing logic:
        # Get active camera for this thread (defaults to 'System' for main/shared threads)
        current_cam = getattr(log_context, 'camera_name', 'System')
        
        # 1. System log handler accepts everything from 'System' context
        if self.camera_id == 'System':
            return current_cam == 'System' or not hasattr(log_context, 'camera_name')
            
        # 2. Camera-specific log handler ONLY accepts logs when its camera is active in this thread
        return current_cam == self.camera_id

def custom_time(*args):
    """India Standard Time (IST) timezone adjustment"""
    return datetime.now(timezone('Asia/Kolkata')).timetuple()

class CustomFormatter(log.Formatter):
    def __init__(self):
        super().__init__()
        self.fmt = "[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] [%(camera_id)s] - %(message)s"
        self.FORMATS = {
            log.DEBUG: "\033[38;20m" + self.fmt + "\033[0m",
            log.INFO: "\033[32;20m" + self.fmt + "\033[0m",
            log.WARNING: "\033[33;20m" + self.fmt + "\033[0m",
            log.ERROR: "\033[31;20m" + self.fmt + "\033[0m",
            log.CRITICAL: "\033[31;1m" + self.fmt + "\033[0m"
        }
    def format(self, record):
        if not hasattr(record, 'camera_id'):
            record.camera_id = 'System'
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = log.Formatter(log_fmt, datefmt='%d-%b-%y %H:%M:%S')
        formatter.converter = custom_time
        return formatter.format(record)

def setup_perfect_logging(config, camera_name, to_root=False):
    root_logger = log.getLogger()
    # Add unique logger for each camera
    logger = log.getLogger(camera_name)
    logger.propagate = True
    
    camera_id = config["db"]["camera_id"].get(camera_name, camera_name)
    root_logger.setLevel(log.DEBUG if config.get("verbose") else log.INFO)
    
    if not root_logger.handlers:
        console_handler = log.StreamHandler()
        console_handler.setFormatter(CustomFormatter())
        # Console shows everything
        root_logger.addHandler(console_handler)

    log_dir = config.get('log_path', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, f"{camera_name}.log")
    
    # Check if this handler already exists to avoid duplicates
    for h in root_logger.handlers:
        if isinstance(h, lh.TimedRotatingFileHandler) and h.baseFilename.endswith(f"{camera_name}.log"):
            return logger

    file_handler = lh.TimedRotatingFileHandler(file_path, 'midnight', 1, backupCount=7)
    file_handler.suffix = "%Y-%m-%d"
    
    # IMPORTANT: The filter is what isolates the logs in the file
    file_handler.addFilter(CameraIdFilter(camera_name))
    
    file_formatter = log.Formatter("[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] [%(camera_id)s] - %(message)s", datefmt='%d-%b-%y %H:%M:%S')
    file_formatter.converter = custom_time
    file_handler.setFormatter(file_formatter)
    
    # Always add to root_logger. The filter will handle the steering.
    root_logger.addHandler(file_handler)
    
    return logger

#  ------------------------------------------
#   Detection & OCR Logic
#  ------------------------------------------

def lp_detection(image, dt_net, dt_ln, args, config):
    img_h, img_w = image.shape[:2]
    active_polygon = None
    if config.get("car_in_relative") and args.camera_name in config["car_in_relative"]:
        rel_pts = config["car_in_relative"][args.camera_name]
        abs_pts = [(int(p[0] * img_w), int(p[1] * img_h)) for p in rel_pts]
        active_polygon = Polygon(abs_pts)
    elif "car_in" in config and args.camera_name in config["car_in"]:
        active_polygon = Polygon(eval(config["car_in"][args.camera_name]))

    crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, img_w, img_h
    if active_polygon is not None:
        min_x, min_y, max_x, max_y = active_polygon.bounds
        crop_x1 = max(0, int(min_x) - 100); crop_y1 = max(0, int(min_y) - 100)
        crop_x2 = min(img_w, int(max_x) + 100); crop_y2 = min(img_h, int(max_y) + 100)
    
    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2] if crop_x2 > crop_x1 and crop_y2 > crop_y1 else image
    dt_boxes, dt_confidences, _ = get_bbox(crop_img, dt_net, dt_ln, config["models"]["number_plate_threshold"])
    
    dt_list, dt_conf = [], []
    if dt_boxes is not None:
        for index, b in enumerate(dt_boxes):
            (l, t, w, h) = b[:4]
            # Format: [(x1, y1), (x2, y2)] for the tracker
            dt_list.append([(int(l + crop_x1), int(t + crop_y1)), (int(l + crop_x1 + w), int(t + crop_y1 + h))])
            dt_conf.append(dt_confidences[index])
    return dt_list, dt_conf

def box_draw(image_list, bbox_list, rg_net, rg_ln, labels, dt_net, dt_ln, lp_confss, 
             obj_id=None, best_global_frame=None, camera_name=None, config=None):
    if not image_list: return
    
    # 1. Augmentation (Ghost Crops)
    augmented_crops = []
    if best_global_frame is not None and len(bbox_list) >= 1:
        dense_bboxes = interpolate_bboxes(bbox_list, num_intermediate=2)
        fh, fw = best_global_frame.shape[:2]
        for bbox in dense_bboxes:
            x1, y1 = max(0, int(bbox[0][0]) - 15), max(0, int(bbox[0][1]) - 15)
            x2, y2 = min(fw, int(bbox[1][0]) + 15), min(fh, int(bbox[1][1]) + 15)
            if x2 > x1 and y2 > y1: augmented_crops.append(best_global_frame[y1:y2, x1:x2].copy())

    combined_images = list(image_list) + augmented_crops
    
    # 2. Batch OCR (Reduced chunk size for 7.5GB GPU)
    batch_results = get_bbox_batch(combined_images, rg_net, rg_ln, config["models"].get("ocr_threshold", 0.5), chunk_size=config.get("ocr_chunk_size", 8))
    
    plate_data, plate_results, infos = [], [], []
    lp_image, best_quality = combined_images[0], -1.0

    for i, (rg_boxes, rg_confidences, rg_classids) in enumerate(batch_results):
        if rg_boxes is not None:
            char_count = len(rg_boxes)
            avg_conf = sum(rg_confidences) / char_count if char_count > 0 else 0
            quality = char_count * 10 + avg_conf
            if quality > best_quality:
                best_quality = quality
                lp_image = combined_images[i]
            
            info = sort_rect([(labels[rg_classids[j]], rg_boxes[j], rg_confidences[j]) for j in range(len(rg_classids))])
            if info:
                infos.append(info)
                plate_number = "".join([item[0] for item in info])
                plate_results.append(plate_number)
                plate_data.append((plate_number, avg_conf))

    if not plate_data: return

    # 3. Consolidation
    checksum_exclude = config.get("checksum_exclude", ['W', 'B', 'M', 'N', 'J', 'CC'])
    v_indian = config["models"].get("validate_indian_plate", True)
    consolidated_all, _ = consolidate_ocr_results(plate_data, checksum_exclude, validate_indian_plate=v_indian)
    
    consolidated = consolidated_all[0]
    group_summary = consolidated_all[1]

    if consolidated:
        with validated_lock: validated_ids.add((camera_name, obj_id))
        log.info(f"[OCR] [{camera_name}] ID:{obj_id} Final: {consolidated}")
        
        processor = text_processors.get(camera_name)
        if processor:
            full_frame = best_global_frame if best_global_frame is not None else lp_image
            def text_process_context_wrapper(*args):
                log_context.camera_name = camera_name
                processor.text_process(*args)
                
            t = Thread(target=text_process_context_wrapper, args=(lp_image, consolidated, group_summary, lp_confss, full_frame, obj_id))
            t.start()
    
    # Final memory cleanup for the vehicle batch
    del combined_images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ocr_worker(rg_net, rg_ln, labels, dt_net, dt_ln, config):
    log.info("Shared OCR Worker started.")
    while True:
        try:
            data = ocr_queue.get()
            if data is None: break
            obj_id, img_list, bbox_list, lp_confss, best_frame, is_final, cam_name = data
            log_context.camera_name = cam_name # Set thread context for logging
            
            with validated_lock:
                if (cam_name, obj_id) in validated_ids and not is_final:
                    ocr_queue.task_done(); continue
            
            box_draw(img_list, bbox_list, rg_net, rg_ln, labels, dt_net, dt_ln, lp_confss, 
                     obj_id, best_frame, cam_name, config)
            
            if is_final:
                with validated_lock:
                    if (cam_name, obj_id) in validated_ids: validated_ids.remove((cam_name, obj_id))
            ocr_queue.task_done()
            
            # Proactive memory release
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log.error(f"OCR Worker error: {e}", exc_info=1)
            sleep(1)

def offload_tracks_to_queue(completed_tracks, camera_name, config):
    dir_cfg = config.get("direction_config", {}).get(camera_name, {"enabled": False})
    for track in completed_tracks:
        obj_id, images, bboxes, confs, zone_history, is_final, best_full, disp_vec = track
        
        is_valid = not dir_cfg.get("enabled", False)
        if dir_cfg.get("enabled"):
            clean_seq = [z for z in zone_history if z in ['A', 'B']]
            if clean_seq:
                mode = dir_cfg.get("mode", "entry")
                if mode == "entry": is_valid = ('B' in clean_seq and clean_seq[-1] == 'A') or len(clean_seq) == 1
                else: is_valid = ('A' in clean_seq and clean_seq[-1] == 'B') or len(clean_seq) == 1

        if is_valid and not ocr_queue.full():
            ocr_queue.put((obj_id, images, bboxes, confs, best_full, is_final, camera_name))

def _draw_live_info(image, objects, zone_polygons, config, camera_name):
    """Enhanced live frame visualizer"""
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # 1. Draw Zones with transparency
    for name, poly in zone_polygons.items():
        pts = np.array(poly.exterior.coords, np.int32).reshape((-1, 1, 2))
        color = (0, 255, 0) if name == "A" else (0, 0, 255)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(image, [pts], True, color, 3)
        cv2.putText(image, f"ZONE {name}", tuple(pts[0][0]), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

    # 2. Blend zones
    cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

    # 3. Draw ROI (car_in)
    roi_rel = config.get("car_in_relative", {}).get(camera_name, [])
    if roi_rel:
        roi_pts = np.array([(int(p[0]*w), int(p[1]*h)) for p in roi_rel], np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [roi_pts], True, (255, 255, 255), 2)

    # 4. Draw Active Vehicles
    for obj in [o for o in objects if not o.has_ended]:
        if not obj.bboxes: continue
        bbox = obj.bboxes[-1]
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[1][0]), int(bbox[1][1])
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        lbl = f"TRACKER ID: {obj.obj_id}"
        if obj.zone_history:
            lbl += f" [{' -> '.join(obj.zone_history[-2:])}]"
        
        # Label background
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 255), -1)
        cv2.putText(image, lbl, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def process_camera(camera_name, config, det_net, labels, ocr_net, cap):
    args = SimpleNamespace(camera_name=camera_name)
    logger = setup_perfect_logging(config, camera_name)
    log_context.camera_name = camera_name # Set thread context for logging

    # FPS Limiter setup
    target_fps = config.get("camera_fps_limit", 25)
    frame_duration = 1.0 / target_fps
    last_frame_time = time.time()

    logger.info(f"Starting camera thread: {camera_name}")
    
    # Init MQTT & TextProcessor
    MQTT = {'host':"", "port":"", "username":"", "password":"", "topic":"", "serial_id":""}
    # (Abbreviated MQTT fetch logic for brevity - assuming standard config for now)
    app_type = config['application_type'].get(camera_name, 'normal')
    if app_type == 'normal' and config['mqtt']['status']:
        MQTT.update(config['mqtt'])
        MQTT['serial_id'] = config['mqtt']['serial_ids'].get(camera_name, "")
    elif app_type == 'resident':
        MQTT.update(config['resident_mqtt'])
        MQTT['serial_id'] = config['mqtt']['serial_ids'].get(camera_name, "")

    text_processors[camera_name] = TextProcess(config, camera_name, MQTT)
    tracker = PlateTracker(
        iou_threshold=config.get("IOU_THRESHOLD", 0.3),
        max_age=config.get("TRACKER_MAX_AGE", 10),
        max_batch_size=config.get("max_plate_batch_size", 40),
        distance_threshold=config.get("DISTANCE_THRESHOLD", 1000),
        distance_scale_factor=config.get("DISTANCE_SCALE_FACTOR", 2.5)
    )
    
    # Scaling optimization (Default to 1280x720 for load reduction)
    w_scale = config.get("stream_width", 1280)
    h_scale = config.get("stream_height", 720)
    
    # Init Frame Skip (Target ~5-8 FPS AI processing)
    frame_count = 0
    skip_interval = config.get("camera_fps", 3) # Process every Nth frame

    resolution_detected = False
    zone_polygons = {}

    fail_count = 0
    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(f"No frame from camera {camera_name}. Retrying ({fail_count+1}/5)...")
                sleep(1)
                fail_count += 1
                if fail_count >= 5: # Faster recovery
                    logger.warning(f"Reconnecting {camera_name} due to inactivity...")
                    if hasattr(cap, 'restart'):
                        cap.restart()
                    else:
                        cap.release()
                        cap = _open_camera_stream(config, camera_name)
                    fail_count = 0
                continue
            
            # FPS Limiter logic (Wall clock sync)
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)
            last_frame_time = time.time()
            
            fail_count = 0
            frame_count += 1
            image = frame.copy() # Essential for FFmpegRTSPReader and multiple threads
            h, w = image.shape[:2]

            # Lazy Initialize Polygons with actual resolution
            if not resolution_detected:
                dir_cfg = config.get("direction_config", {}).get(camera_name, {})
                if dir_cfg.get("method") == "zone_transition":
                    if "zone_a" in dir_cfg:
                        zone_polygons["A"] = Polygon([(int(p[0]*w), int(p[1]*h)) for p in dir_cfg["zone_a"]])
                    if "zone_b" in dir_cfg:
                        zone_polygons["B"] = Polygon([(int(p[0]*w), int(p[1]*h)) for p in dir_cfg["zone_b"]])
                resolution_detected = True
                log.info(f"[{camera_name}] Thread running at {w}x{h}")

            # --- PROCESS EVERY N-TH FRAME FOR LOAD REDUCTION ---
            if frame_count % (skip_interval + 1) == 0:
                with torch.no_grad():
                    dt_list, dt_conf = lp_detection(image, det_net, None, args, config)
                
                # Tracking with Zones
                completed_tracks = tracker.update(dt_list, dt_conf, image, zone_polygons=zone_polygons)
                
                if completed_tracks:
                    offload_tracks_to_queue(completed_tracks, camera_name, config)

            if config.get("draw_inference"):
                _draw_live_info(image, tracker.objects, zone_polygons, config, camera_name)
            
            # --- DISPLAY UPDATE (Thread Safe) ---
            if config.get("show_video"):
                with display_lock:
                    display_frames[camera_name] = image  # Push annotated frame to main thread
                    
        except Exception as e:
            logger.error(f"Thread {camera_name} error: {e}", exc_info=1)
            sleep(2)

def camera_main():
    args = parse_args()
    with open(args.config_file) as f: config = json.load(f)
    
    # Global logger init for main process
    # If running a specific camera, make it the root logger for this process
    main_log_name = args.camera_name if args.camera_name else "System"
    setup_perfect_logging(config, main_log_name)
    log_context.camera_name = main_log_name
    
    # Singleton Model Loading (VRAM Optimized)
    container = ModelContainer()
    det_model, ocr_model = container.load_models(
        config["models"]["number_plate_model"],
        config["models"]["ocr_model"],
        device=config["models"].get("device", "auto"),
        use_fp16=config["models"].get("use_fp16", True)
    )
    labels = config.get("labels", "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
    
    # Start Shared Worker
    Thread(target=ocr_worker, args=(ocr_model, None, labels, det_model, None, config), daemon=True).start()
    
    # Launch Camera Threads
    if args.camera_name:
        cameras = [args.camera_name]
    else:
        enabled = config.get("enabled_cameras")
        if enabled:
            cameras = [c for c in enabled if c in config["camera_url"]]
        else:
            cameras = list(config["camera_url"].keys())
    # Pre-open Streams in Main Thread (Resolves QObject/Timer threading errors)
    threads = []
    for cam in cameras:
        log.info(f"Opening stream for {cam}...")
        cap = _open_camera_stream(config, cam)
        t = Thread(target=process_camera, args=(cam, config, det_model, labels, ocr_model, cap), daemon=True)
        t.start()
        threads.append(t)
    
    # --- MAIN THREAD DISPLAY LOOP ---
    initialized_windows = set()
    try:
        if config.get("show_video"):
            log.info("Starting Main Display Loop...")
            while True:
                current_frames = {}
                with display_lock:
                    current_frames = {k: v for k, v in display_frames.items()}
                
                for cam_name, frame_to_show in current_frames.items():
                    if cam_name not in initialized_windows:
                        cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(cam_name, 1280, 720)
                        initialized_windows.add(cam_name)
                    
                    cv2.imshow(cam_name, frame_to_show)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    log.info("'q' pressed. Shutting down...")
                    break
                sleep(0.01)
        else:
            while True: sleep(1)
    except KeyboardInterrupt:
        log.info("Shutdown requested.")
    finally:
        cv2.destroyAllWindows()
        os._exit(0)

if __name__ == '__main__':
    camera_main()
