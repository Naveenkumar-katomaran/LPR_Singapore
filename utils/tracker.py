import numpy as np
import time
import logging as log
from shapely.geometry import Point

logging = log.getLogger(__name__)


class TrackedObject:
    def __init__(self, obj_id, frame, bbox, conf, max_batch_size):
        self.obj_id = obj_id
        self.images = [] # Store CROPS only
        self.bboxes = []
        self.confs = []
        self.best_full_frame = None # Single high-quality overview shot
        self.best_conf = -1.0
        
        self.last_seen = time.time()
        self.max_batch_size = max_batch_size
        self.is_validated = False
        self.needs_flush = False
        self.has_ended = False
        self.matched_this_cycle = False
        self.detection_updates = 0
        self.zone_history = []
        self.velocity = (0, 0) # (vx, vy) pixels per frame
        self.journey_ready = False 
        self.missing_frames = 0
        self.start_centroid = get_centroid(bbox)

        # Add the first detection
        self.add_detection(frame, bbox, conf)

    def predict(self):
        """Predict the bounding box in the next frame based on constant velocity."""
        last_bbox = self.bboxes[-1]
        (x1, y1), (x2, y2) = last_bbox
        vx, vy = self.velocity
        
        # Shift the box by velocity * (missing_frames + 1)
        # This keeps the ghost box moving during occlusions
        multiplier = self.missing_frames + 1
        p_bbox = [
            (x1 + vx * multiplier, y1 + vy * multiplier),
            (x2 + vx * multiplier, y2 + vy * multiplier)
        ]
        return p_bbox

    def get_direction_vector(self):
        """Calculate the total displacement vector from first seen to last seen."""
        if not self.bboxes: return (0, 0)
        curr_centroid = get_centroid(self.bboxes[-1])
        return (curr_centroid[0] - self.start_centroid[0], 
                curr_centroid[1] - self.start_centroid[1])

    def update_zone_history(self, zone_name):
        """Record a zone visit only if it's different from the last zone recorded."""
        if not self.zone_history or self.zone_history[-1] != zone_name:
            self.zone_history.append(zone_name)
            logging.debug(f"[Tracker] ID:{self.obj_id} entered Zone {zone_name}")

    def add_detection(self, frame, bbox, conf):
        # Update velocity if we have at least one previous box
        # Velocity Bootstrapping:
        # 1. On the very first detection, velocity stays (0,0)
        # 2. On the second detection, we initialize velocity to the first actual delta
        # 3. Subsequent detections use EMA smoothing to handle noise
        if len(self.bboxes) == 1:
            def get_c(b): return ((b[0][0]+b[1][0])/2, (b[0][1]+b[1][1])/2)
            c_prev = get_c(self.bboxes[-1])
            c_now = get_c(bbox)
            self.velocity = (c_now[0] - c_prev[0], c_now[1] - c_prev[1])
        elif len(self.bboxes) > 1:
            def get_c(b): return ((b[0][0]+b[1][0])/2, (b[0][1]+b[1][1])/2)
            c_prev = get_c(self.bboxes[-1])
            c_now = get_c(bbox)
            # instantaneous velocity
            vx_new = c_now[0] - c_prev[0]
            vy_new = c_now[1] - c_prev[1]
            # Smooth velocity (70% new, 30% old)
            self.velocity = (
                0.7 * vx_new + 0.3 * self.velocity[0],
                0.7 * vy_new + 0.3 * self.velocity[1]
            )

        # CROPPING LOGIC: Store only the plate area to save RAM
        try:
            (fh, fw) = frame.shape[:2]
            # bbox format: [(x1, y1), (x2, y2)]
            # Apply 15px padding for robustness (matches legacy box_draw behavior)
            x1, y1 = max(0, int(bbox[0][0])-15), max(0, int(bbox[0][1])-15)
            x2, y2 = min(fw, int(bbox[1][0])+15), min(fh, int(bbox[1][1])+15)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2].copy()
                self.images.append(crop)
                self.bboxes.append(bbox)
                self.confs.append(conf)
                
                # Update Best Full Frame (Overview Shot)
                if conf > self.best_conf:
                    self.best_conf = conf
                    self.best_full_frame = frame.copy() 
        except Exception as e:
            logging.error(f"[Tracker] Cropping error: {e}")

        self.detection_updates += 1
        
        # Batching logic:
        if len(self.images) >= self.max_batch_size:
            self.needs_flush = True

        self.last_seen = time.time()
        self.missing_frames = 0

    def add_frame(self, frame):
        # Optimization: In non-detection frames, we DON'T add images.
        # This saves massive CPU (encoding/copying) and RAM.
        pass


def get_iou(boxA, boxB):
    # box format: [(x1, y1), (x2, y2)]
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[1][0] - boxA[0][0]) * (boxA[1][1] - boxA[0][1])
    boxBArea = (boxB[1][0] - boxB[0][0]) * (boxB[1][1] - boxB[0][1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def get_centroid(box):
    return ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)

class PlateTracker:
    def __init__(self, iou_threshold=0.5, max_age=5, max_batch_size=30, distance_threshold=300, distance_scale_factor=1.5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age # Max frames to miss before finishing
        self.max_batch_size = max_batch_size
        self.objects = []
        self.next_id = 0
        self.frame_count = 0
        self.distance_threshold = distance_threshold
        self.distance_scale_factor = distance_scale_factor

    def update(self, detections, confidences, frame, zone_polygons=None):
        self.frame_count += 1
        
        active_objects = [o for o in self.objects if not o.has_ended]
        if not active_objects:
            # Shortcut: All detections are new
            for i, det in enumerate(detections):
                new_obj = TrackedObject(self.next_id, frame, det, confidences[i], self.max_batch_size)
                self.objects.append(new_obj)
                logging.info(f"[Tracker] New Vehicle Assigned | ID: {self.next_id}")
                self.next_id += 1
            return []

        # 1. Pre-calculate detection properties to avoid inner-loop overhead
        det_centroids = [get_centroid(d) for d in detections]
        
        costs = []
        for obj_idx, obj in enumerate(active_objects):
            predicted_bbox = obj.predict()
            p_centroid = get_centroid(predicted_bbox)
            
            # Pre-calculate object-specific thresholds
            last_bbox = obj.bboxes[-1]
            obj_w = last_bbox[1][0] - last_bbox[0][0]
            
            if obj.detection_updates < 2:
                dynamic_dist_limit = self.distance_threshold * 0.8
            else:
                dynamic_dist_limit = obj_w * self.distance_scale_factor
                dynamic_dist_limit = min(dynamic_dist_limit, self.distance_threshold)
            
            # Square the limit to avoid sqrt in inner loop
            dynamic_dist_limit_sq = dynamic_dist_limit ** 2
            
            for det_idx, det in enumerate(detections):
                # IOU is only for objects that overlap (spatial pruning)
                iou_val = get_iou(predicted_bbox, det)
                
                det_centroid = det_centroids[det_idx]
                dist_sq = (p_centroid[0] - det_centroid[0])**2 + (p_centroid[1] - det_centroid[1])**2
                
                if iou_val > self.iou_threshold:
                    cost = 1.0 - iou_val
                    costs.append((cost, obj_idx, det_idx, "IOU"))
                elif dist_sq < dynamic_dist_limit_sq:
                    cost = 1.1 + (dist_sq / dynamic_dist_limit_sq)
                    costs.append((cost, obj_idx, det_idx, "Dist"))

        # 2. Global Min-Cost Assignment (Greedy sorted)
        costs.sort(key=lambda x: x[0])
        
        matched_objs = set()
        matched_dets = set()
        
        for cost, obj_idx, det_idx, method in costs:
            if obj_idx in matched_objs or det_idx in matched_dets:
                continue
            
            obj = active_objects[obj_idx]
            det_bbox = detections[det_idx]
            obj.add_detection(frame, det_bbox, confidences[det_idx])
            obj.matched_this_cycle = True
            
            # Zone detection
            if zone_polygons:
                centroid = det_centroids[det_idx]
                for zone_name, poly in zone_polygons.items():
                    if poly.contains(Point(centroid)):
                        obj.update_zone_history(zone_name)
                        break
            
            matched_objs.add(obj_idx)
            matched_dets.add(det_idx)
            
            if method == "Dist":
                logging.debug(f"[Tracker] ID:{obj.obj_id} Matched by Dynamic Distance: {cost-1.1:.2f} relative units")
            else:
                logging.debug(f"[Tracker] ID:{obj.obj_id} Matched by IOU: {1.0-cost:.2f}")

        # 3. Create new objects for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                new_obj = TrackedObject(self.next_id, frame, det, confidences[i], self.max_batch_size)
                self.objects.append(new_obj)
                logging.info(f"[Tracker] New Vehicle Assigned | ID: {self.next_id}")
                
                # Zone detection for new object
                if zone_polygons:
                    centroid = det_centroids[i]
                    for zone_name, poly in zone_polygons.items():
                        if poly.contains(Point(centroid)):
                            new_obj.update_zone_history(zone_name)
                            break
                            
                self.next_id += 1

        # 3. Identify completed or expired tracks
        completed_data = []
        remaining_objects = []
        
        for obj in self.objects:
            # Fix: Use the flag we set during matching phase
            if not obj.matched_this_cycle:
                if not hasattr(obj, 'missing_frames'): obj.missing_frames = 0
                obj.missing_frames += 1
                if obj.missing_frames >= self.max_age:
                    obj.has_ended = True
                    logging.info(f"[Tracker] Vehicle ID: {obj.obj_id} exited frame (Expired)")
            else:
                obj.missing_frames = 0
            
            # Reset flag for next cycle
            obj.matched_this_cycle = False

            # FLUSH CONDITION:
            # Case 1: Car has ended (left screen or timed out)
            # Case 2: We reached max_batch_size, but car is still there (Partial Flush)
            if obj.has_ended or obj.needs_flush:
                # Format: obj_id, images, bboxes, confs, zone_history, is_final, best_full, displacement_vector
                completed_data.append((
                    obj.obj_id, 
                    list(obj.images), # copy list
                    list(obj.bboxes), 
                    list(obj.confs), 
                    list(obj.zone_history), 
                    obj.has_ended, # is_final
                    obj.best_full_frame, 
                    obj.get_direction_vector()
                ))
                
                if obj.has_ended:
                    logging.info(f"[Tracker] Vehicle ID: {obj.obj_id} finalized (Exit) - Total images: {len(obj.images)}")
                else:
                    logging.info(f"[Tracker] Vehicle ID: {obj.obj_id} partial flush (Batch) - Images: {len(obj.images)}")
                    # Reset internal image buffer for next batch to save RAM
                    # BUT keep best_full_frame and base tracking info
                    obj.images = []
                    obj.needs_flush = False

            if not obj.has_ended:
                remaining_objects.append(obj)
        
        self.objects = remaining_objects
        return completed_data

    def add_frame_to_all(self, frame):
        # Add the current frame to all active tracks to ensure dense batches
        completed_data = []
        remaining_objects = []
        
        for obj in self.objects:
            obj.add_frame(frame)
            remaining_objects.append(obj)
                
        self.objects = remaining_objects
        return completed_data
