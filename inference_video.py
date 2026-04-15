import cv2
import numpy as np
import json
import argparse
import time
import os
import torch
from types import SimpleNamespace
from datetime import datetime
from collections import deque
from threading import Thread, Lock
from queue import Queue
from difflib import SequenceMatcher

from utils.detect import ModelContainer, get_bbox, get_bbox_batch, get_vehicle_type, sort_rect
from utils.tracker import PlateTracker
from utils.bbox_asumption import interpolate_bboxes
from utils.ocr import consolidate_ocr_results

# Color Palette (Aesthetics)
CLR_BG = (20, 20, 20)      # Dark background
CLR_ACCENT = (0, 190, 255) # Cyan/Blue accent
CLR_TEXT = (255, 255, 255) # White text
CLR_GREEN = (0, 255, 127)  # Success green
CLR_YELLOW = (0, 255, 255) # Warning yellow
CLR_BORDER = (50, 50, 50)  # Border color

class Dashboard:
    def __init__(self, target_res=(1920, 1080), sidebar_width=450):
        self.target_res = target_res
        self.sidebar_width = sidebar_width
        self.main_res = (target_res[0] - sidebar_width, target_res[1])
        self.config = None # Will be set in process_video
        
        # History for sidebar (Legend - for display)
        self.history = deque(maxlen=6)
        # Detailed history for deduplication {plate: (timestamp, confidence)}
        self.seen_history = {} 
        self.lock = Lock()
        
        # UI State
        self.start_time = time.time()
        self.fps = 0
        self.frame_count = 0

    def add_detection(self, plate_num, confidence, plate_img, veh_type, timestamp):
        with self.lock:
            # Deduplication Logic
            dedupe_cfg = self.config.get("deduplication", {"enabled": False})
            if dedupe_cfg.get("enabled"):
                cooldown = dedupe_cfg.get("cooldown_seconds", 300)
                threshold = dedupe_cfg.get("similarity_threshold", 0.85)
                current_time = time.time()
                
                # Cleanup old history
                self.seen_history = {p: v for p, v in self.seen_history.items() if current_time - v[0] < cooldown}
                
                # Check Similarity
                is_duplicate = False
                for seen_plate, (last_time, last_conf) in self.seen_history.items():
                    sim = SequenceMatcher(None, plate_num, seen_plate).ratio()
                    if sim >= threshold:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    return False # Skip adding

            # Add to detailed history for future deduplication
            self.seen_history[plate_num] = (time.time(), confidence)
            
            # Add to display deque
            self.history.appendleft({
                "plate": plate_num,
                "confidence": confidence,
                "image": plate_img,
                "type": veh_type,
                "time": timestamp
            })
            return True

    def draw_rounded_rect(self, img, pt1, pt2, color, thickness=1, radius=10):
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw lines
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw corners
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    def compose(self, main_frame, active_objects):
        # 1. Resize main frame to fit dashboard
        main_h, main_w = main_frame.shape[:2]
        scale = min(self.main_res[0] / main_w, self.main_res[1] / main_h)
        nw, nh = int(main_w * scale), int(main_h * scale)
        main_resized = cv2.resize(main_frame, (nw, nh))
        
        # Create canvas
        canvas = np.full((self.target_res[1], self.target_res[0], 3), CLR_BG, dtype=np.uint8)
        
        # Center main frame vertically if needed
        y_off = (self.target_res[1] - nh) // 2
        canvas[y_off:y_off+nh, 0:nw] = main_resized
        
        # 2. Draw Sidebar
        self.draw_sidebar(canvas, nw)
        
        # 3. Draw Header
        self.draw_header(canvas)
        
        return canvas

    def draw_header(self, canvas):
        # Top bar
        cv2.rectangle(canvas, (0, 0), (self.target_res[0], 60), (30, 30, 30), -1)
        cv2.line(canvas, (0, 60), (self.target_res[0], 60), CLR_ACCENT, 2)
        
        # Title
        cv2.putText(canvas, "ANPR | AUTOMATIC NUMBER PLATE RECOGNITION", (20, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, CLR_TEXT, 1, cv2.LINE_AA)
        
        # Live Stats
        elapsed = time.time() - self.start_time
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(canvas, f"STREAM: ACTIVE  |  {time_str}", (self.target_res[0] - 400, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, CLR_ACCENT, 1, cv2.LINE_AA)

    def draw_sidebar(self, canvas, sidebar_x):
        # Sidebar BG
        overlay = canvas.copy()
        cv2.rectangle(overlay, (sidebar_x, 60), (self.target_res[0], self.target_res[1]), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
        
        cv2.line(canvas, (sidebar_x, 60), (sidebar_x, self.target_res[1]), CLR_BORDER, 1)
        
        # Label
        cv2.putText(canvas, "DETECTION LEGEND", (sidebar_x + 20, 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, CLR_ACCENT, 1, cv2.LINE_AA)
        cv2.line(canvas, (sidebar_x + 20, 115), (sidebar_x + 200, 115), CLR_ACCENT, 1)
        
        # Draw History Items
        y_start = 140
        card_h = 130
        gap = 20
        
        with self.lock:
            for i, item in enumerate(list(self.history)):
                curr_y = y_start + i * (card_h + gap)
                if curr_y + card_h > self.target_res[1]: break
                
                # Card Background
                cv2.rectangle(canvas, (sidebar_x + 15, curr_y), (self.target_res[0] - 15, curr_y + card_h), (40, 40, 40), -1)
                cv2.rectangle(canvas, (sidebar_x + 15, curr_y), (self.target_res[0] - 15, curr_y + card_h), (60, 60, 60), 1)
                
                # Plate Image
                if item["image"] is not None:
                    p_img = item["image"]
                    p_h, p_w = p_img.shape[:2]
                    # Resize to fit height roughly
                    target_ph = 60
                    target_pw = int(p_w * (target_ph / p_h))
                    if target_pw > 180: # Limit width
                        target_pw = 180
                        target_ph = int(p_h * (target_pw / p_w))
                    
                    p_resized = cv2.resize(p_img, (target_pw, target_ph))
                    canvas[curr_y + 15 : curr_y + 15 + target_ph, sidebar_x + 30 : sidebar_x + 30 + target_pw] = p_resized
                    cv2.rectangle(canvas, (sidebar_x + 30, curr_y + 15), (sidebar_x + 30 + target_pw, curr_y + 15 + target_ph), CLR_ACCENT, 1)
                
                # OCR Text
                cv2.putText(canvas, item["plate"], (sidebar_x + 230, curr_y + 40), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, CLR_GREEN if item["confidence"] > 0.8 else CLR_YELLOW, 2, cv2.LINE_AA)
                
                # Metadata
                cv2.putText(canvas, f"TYPE: {item['type'].upper()}", (sidebar_x + 230, curr_y + 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)
                
                # Confidence Bar
                conf = item["confidence"]
                bar_w = 180
                cv2.rectangle(canvas, (sidebar_x + 230, curr_y + 90), (sidebar_x + 230 + bar_w, curr_y + 105), (20, 20, 20), -1)
                cv2.rectangle(canvas, (sidebar_x + 230, curr_y + 90), (sidebar_x + 230 + int(bar_w * conf), curr_y + 105), 
                              CLR_GREEN if conf > 0.8 else CLR_YELLOW, -1)
                cv2.putText(canvas, f"{int(conf*100)}%", (sidebar_x + 230 + bar_w + 10, curr_y + 102), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)
                
                # Time
                time_ago = item["time"].strftime("%H:%M:%S")
                cv2.putText(canvas, time_ago, (sidebar_x + 30, curr_y + 115), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

def lp_detection(image, det_net, config):
    """Simplified version of camera.py logic"""
    img_h, img_w = image.shape[:2]
    dt_boxes, dt_confidences, _ = get_bbox(image, det_net, None, config["models"]["number_plate_threshold"])
    
    dt_list, dt_conf = [], []
    if dt_boxes is not None:
        for index, b in enumerate(dt_boxes):
            (l, t, w, h) = b[:4]
            dt_list.append([(int(l), int(t)), (int(l+w), int(t+h))])
            dt_conf.append(dt_confidences[index])
    return dt_list, dt_conf

def process_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config.json')
    parser.add_argument('--camera_name', required=True)
    parser.add_argument('--input', help='Video file path. If omitted, uses RTSP from config.')
    parser.add_argument('--output', default='videos/output.mp4')
    parser.add_argument('--limit_frames', type=int, default=0, help='Stop after N frames')
    args = parser.parse_args()

    with open(args.config_file) as f: config = json.load(f)

    # 1. Setup Models
    container = ModelContainer()
    det_model, ocr_model, veh_model = container.load_models(
        config["models"]["number_plate_model"],
        config["models"]["ocr_model"],
        vehicle_weight=config["models"].get("vehicle_model"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fp16=config["models"].get("use_fp16", False)
    )
    labels = config.get("labels", "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")

    # 2. Setup Input
    video_url = args.input if args.input else config["camera_url"][args.camera_name]
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(f"Error: Could not open {video_url}")
        return

    # 3. Setup Video Writer
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    dash_res = (1920, 1080)
    out_path = args.output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, dash_res)

    # 4. Initialize Components
    dash = Dashboard(target_res=dash_res)
    dash.config = config # Pass config for dedupe settings
    
    # Sync tracker with config
    max_batch = config.get("max_plate_batch_size", 50)
    tracker = PlateTracker(
        iou_threshold=config.get("IOU_THRESHOLD", 0.3),
        max_age=config.get("TRACKER_MAX_AGE", 15),
        max_batch_size=max_batch
    )
    
    print(f"Processing video: {video_url} -> {out_path}")
    print("Press 'q' in playback window to stop early (if visible).")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break
            
            frame_idx += 1
            if args.limit_frames > 0 and frame_idx > args.limit_frames: break

            # Inference - Every frame for high accuracy
            dt_list, dt_conf = lp_detection(frame, det_model, config)
            completed_tracks = tracker.update(dt_list, dt_conf, frame)
            
            # Handle OCR for active objects
            for obj in tracker.objects:
                # Process if it has enough frames and NOT yet processed
                # Increased window to the last 20 frames for better voting
                if not obj.has_ended and len(obj.images) >= 5:
                    if getattr(obj, 'processed_ocr', False): continue
                    
                    # Use a wider batch for better consolidation
                    sample_size = min(len(obj.images), 24)
                    ocr_batch_results = get_bbox_batch(obj.images[-sample_size:], ocr_model, None, 0.4)
                    plate_data = []
                    
                    for res_boxes, res_confs, res_cls in ocr_batch_results:
                        if res_boxes:
                            info = sort_rect([(labels[res_cls[j]], res_boxes[j], res_confs[j]) for j in range(len(res_cls))])
                            if info:
                                plate_data.append(("".join([it[0] for it in info]), sum(res_confs)/len(res_confs)))
                    
                    if plate_data:
                        consolidated_all, _ = consolidate_ocr_results(plate_data, config.get("checksum_exclude", []))
                        consolidated = consolidated_all[0]
                        if consolidated:
                            veh_type, _ = get_vehicle_type(frame, obj.bboxes[-1], veh_model)
                            dash.add_detection(consolidated, plate_data[0][1], obj.images[-1], veh_type, datetime.now())
                            obj.processed_ocr = True # Mark as processed

            # Draw Annotations on raw frame before composing
            annotated = frame.copy()
            for obj in [o for o in tracker.objects if not o.has_ended]:
                if not obj.bboxes: continue
                x1, y1 = obj.bboxes[-1][0]
                x2, y2 = obj.bboxes[-1][1]
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), CLR_ACCENT, 2)
                cv2.putText(annotated, f"ID: {obj.obj_id}", (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_ACCENT, 1)

            # Compose Dashboard
            final_frame = dash.compose(annotated, tracker.objects)
            
            # Write to video
            writer.write(final_frame)
            
            # Optional: Show progress
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames...")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cap.release()
        writer.release()
        print(f"Finished. Video saved to {out_path}")

if __name__ == '__main__':
    process_video()
