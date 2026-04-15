import json
import logging as log
import os
import time
from datetime import datetime
from difflib import SequenceMatcher
import glob
from queue import Queue
import cv2
import paho.mqtt.client as mqtt
import requests
from pytz import timezone
import pandas as pd
import socket
import uuid
from utils.minio_utils import MinioClient
from utils.detect import ModelContainer, get_vehicle_type
from utils.rabbitmq_utils import RabbitMQProducer

logging = log.getLogger(__name__)
response_500 = Queue()
owner_details = None


def write_fun(image, string, fmt, d_t, parent_folder_name, ocr_batch_results, lp_confs):
    folder_name = d_t.strftime(fmt[:8])
    file_name = d_t.strftime(fmt[9:17]) + '-' + string + ".jpg"
    file_path = "{}/{}/{}".format(parent_folder_name, folder_name, file_name)
    os.makedirs(os.path.join(parent_folder_name, folder_name), exist_ok=True)
    cv2.imwrite(file_path, image)
    try:
        h, w = image.shape[:2]
        image_insights = {
            "height": h, 
            "width": w, 
            "time": d_t.strftime(fmt), 
            "lp_confs": lp_confs,
            "ocr_batch_results": ocr_batch_results  # Structured groupings summary
        }
            
        with open(file_path[:-3] + 'json', 'w') as outfile:
                json.dump(image_insights, outfile, indent=2)
    except Exception as e:
        logging.info(f"Write file Exception : {e}", exc_info=1)
        


def create_plot(folder_name):
    fmt = f"{folder_name}/%d-%m-%Y/"
    real_path = datetime.now().strftime(fmt)
    logging.info(f"Path for plot {real_path}")
    os.makedirs(real_path, exist_ok=True)
    a = glob.glob(real_path + "*.json")
    total = {}
    file_times = []
    checksums = []
    LABELS = ('0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z')
    for i in a:
        data = json.load(open(i))
        for j in data.keys():
            if j == "height" or j == "width" or j == 'time' or j == 'checksum':
                if j == 'checksum':
                    checksums.append(data["checksum"])
                    file_times.append(data["time"].split(' ')[1].strip())
                continue
            if not data[j][0] in total.keys():
                total[data[j][0]] = {"strings":[]}
            total[data[j][0]]['strings'].append(data[j][-1])
    label = []
    avg_value = []

    for k in LABELS:
        if not k in total.keys():
            total[k] = [0]
            label.append(k)
            avg_value.append(0)
            continue
        avg = sum(total[k]["strings"]) / len(total[k]["strings"])
        total[k]["strings"].append(avg)
        label.append(k)
        avg_value.append(avg)
    
    
    bar = {"Labels": label, "Threshold": avg_value}
    bar_df = pd.DataFrame(bar)
    print(label, avg_value)
    bar_df.to_csv(f'{real_path}barchart.csv', sep='\t')

    checksum = {"Time": file_times, "T/F": checksums}
    checksum_df = pd.DataFrame(checksum)
    checksum_df.to_csv(f"{real_path}checsum.csv", sep='\t')



def is_connected(hostname="www.google.com"):
    try:
        # see if we can resolve the host name -- tells us if there is
        # a DNS listening
        host = socket.gethostbyname(hostname)
        # connect to the host -- tells us if the host is actually
        # reachable
        s = socket.create_connection((host, 80), 2)
        s.close()
        return True
    except socket.error as e:
        logging.debug(f'Socket Error  -  {e}')
    return False

#---------------------OWNER DETAILS GETTING------------------#
#------------------------------------------------------------#

def owner_data(config):
    if is_connected() is True:
        global owner_details
        logging.info("Scheduler in running...")

        try:
            owner_details = requests.get(url=config["db"]["resident_endpoint"],
                            headers={"Authorization": config["db"]["Authorization"]})
            logging.info(owner_details)
            owner_details = owner_details.json()
            owner_details = str(owner_details["data"])
            logging.info("----------------------------------")
            logging.info("Total Resident records fetched from DB is {} ".format(len(eval(owner_details))))
            logging.info("Saving fetched records...")
            logging.info("----------------------------------")


            logging.info(owner_details)            
            f=open('data.txt',"w")
            f.write(owner_details)
            f.close()

        except Exception as e:
            logging.info(e)
            owner_details = open('data.txt',"r").read()
            logging.info("-------------Exception---------------------")
            logging.info(owner_details)
            logging.info("----------------------------------")
            logging.info("Total Resident records retained from Local DB is {} ".format(len(eval(owner_details))))
            logging.info("Saving fetched records...")
            logging.info("----------------------------------")
    else:
        logging.info('Internet is disconnected for fetching owner data')
        owner_details = open('data.txt',"r").read()
        logging.info("----------------------------------")
        logging.info("Total Resident records retained from Local DB is {} ".format(len(eval(owner_details))))
        logging.info("Saving fetched records...")
        logging.info("----------------------------------")

class TextProcess:
    def __init__(self, config, camera_name,mqtt_connection_setting):
        self.config = config
        self.camera_name = camera_name
        self.seen_history = {} # {plate_string: (last_seen_time, best_conf)}
        self.new_string = None
        self.payload = {}
        self.time_to_fly = 120
        if 'time_to_fly' in self.config["db"].keys():
            self.time_to_fly = self.config["db"]['time_to_fly']
        self.mqtt_connection_setting = mqtt_connection_setting
        if len(self.mqtt_connection_setting) > 0:
            self.client = mqtt.Client("P1")
            self.client.username_pw_set(username=self.mqtt_connection_setting["username"], password=self.mqtt_connection_setting["password"])
        
        self.payload['vehicle_entries[camera_id]'] = self.config["db"]["camera_id"][self.camera_name]
        self.files = {}
        self.fmt = "%Y-%m-%d %H:%M:%S.%f %z"
        self.last_sent_plates = [] # Memory buffer for the last 2 successfully sent plates
    
        # Initialize MinIO Client
        self.minio_client = None
        if "minio" in self.config:
            self.minio_client = MinioClient(self.config["minio"])
        
        # Get Vehicle Detection Model
        self.vehicle_model = ModelContainer().vehicle_model

        # Initialize RabbitMQ Producer
        self.rabbitmq_producer = None
        if "rabbitmq" in self.config:
            self.rabbitmq_producer = RabbitMQProducer(self.config["rabbitmq"])
    
    def mqtt_publish(self, data):
        try:
            if self.mqtt_connection_setting.get("host") and self.mqtt_connection_setting.get("port"):
                mqtt_data = {'serial_id': self.mqtt_connection_setting.get("serial_id", ""),
                                "data": data}
                self.client.connect(self.mqtt_connection_setting["host"], int(self.mqtt_connection_setting["port"]))
                ret = self.client.publish(self.mqtt_connection_setting.get("topic", ""),
                                            json.dumps(mqtt_data))
                                                                                
                if ret.is_published() is True:
                    logging.info("Yeah ..! MQTT is published")
                else:
                    logging.info(ret)
            else:
                logging.info("Oops...! DCS for {} is not found".format(self.camera_name),exc_info=1)
        except KeyError:
            logging.info("Oops...! DCS for {} is not found".format(self.camera_name),exc_info=1)

    def text_process(self, number_plate_image, number_string, ocr_batch_results, lp_confs, full_frame=None, obj_id=None, plate_bbox=None):
        payload = {}
        files = {}
        try:
            if self.config['bike_lnpr'][self.camera_name] is True:
                min_len = self.config["models"].get("min_plate_length", 4)
            else:
                min_len = self.config["models"].get("min_plate_length", 7)
            bike_cond = min_len <= len(number_string) <= 12
            if bike_cond:
                encoded_lp_image = cv2.imencode('.jpg', number_plate_image)[1].tobytes()
                self.new_string = number_string
                now_utc = datetime.now(timezone('UTC'))
                t_new = now_utc.astimezone(timezone("Asia/Kolkata"))
                
                # Deduplication logic using history
                is_new_event = True
                clean_string = number_string.strip()
                new_conf = sum(lp_confs) / (len(lp_confs) + 1e-6)
                
                # Load deduplication settings
                dedupe_cfg = self.config.get("deduplication", {})
                cooldown = dedupe_cfg.get("cooldown_seconds", self.time_to_fly)
                threshold = dedupe_cfg.get("similarity_threshold", 0.9)

                # Clean up old history entries (> cooldown) to save memory
                current_time = time.time()
                old_count = len(self.seen_history)
                self.seen_history = {p: v for p, v in self.seen_history.items() if current_time - v[0] < cooldown}
                if len(self.seen_history) < old_count:
                    logging.debug(f"[Dedupe] Cleaned up {old_count - len(self.seen_history)} old records from history.")

                # Check if this plate (or a very similar one) was seen recently (Cooldown)
                for seen_plate, (last_time, last_conf) in self.seen_history.items():
                    similarity = SequenceMatcher(None, clean_string, seen_plate).ratio()
                    if similarity >= threshold:
                        if current_time - last_time < cooldown:
                            logging.info(f"[Dedupe] ID:{obj_id} Discarding similar plate (Cooldown): {clean_string} (Matches: {seen_plate}, Sim: {similarity:.2f})")
                            is_new_event = False
                            break
                
                # Secondary Check: Last Sent Buffer (Parked vehicles)
                if is_new_event:
                    for last_plate in self.last_sent_plates:
                        similarity = SequenceMatcher(None, clean_string, last_plate).ratio()
                        if similarity >= threshold:
                            logging.info(f"[Dedupe] ID:{obj_id} Discarding plate (Last Sent): {clean_string} (Matches: {last_plate}, Sim: {similarity:.2f})")
                            is_new_event = False
                            break

                if is_new_event:
                    self.seen_history[clean_string] = (current_time, new_conf)
                    # Update Last Sent Buffer (maintain last 2)
                    self.last_sent_plates.append(clean_string)
                    if len(self.last_sent_plates) > 2:
                        self.last_sent_plates.pop(0)

                    # --- VEHICLE TYPE DETECTION & CROPPING ---
                    veh_type = "car" # Default
                    vehicle_crop = full_frame # Default to full frame if detection fails
                    if full_frame is not None and plate_bbox is not None:
                        veh_type, v_bbox = get_vehicle_type(full_frame, plate_bbox, self.vehicle_model)
                        logging.info(f"[Detector] ID:{obj_id} Detected Vehicle Type: {veh_type}")
                        
                        if v_bbox is not None:
                            try:
                                fh, fw = full_frame.shape[:2]
                                vx1, vy1, vx2, vy2 = v_bbox
                                vh = vy2 - vy1
                                vw = vx2 - vx1
                                
                                # Apply padding to cover the driver/rider
                                # Expand upwards (y1) and sides (x1, x2)
                                pad_up = int(vh * 0.4) # Add 40% height above (for driver)
                                pad_side = int(vw * 0.1) # Add 10% on sides
                                
                                ny1 = max(0, vy1 - pad_up)
                                ny2 = vy2 # Keep the bottom
                                nx1 = max(0, vx1 - pad_side)
                                nx2 = min(fw, vx2 + pad_side)
                                
                                vehicle_crop = full_frame[ny1:ny2, nx1:nx2].copy()
                                logging.info(f"[Detector] ID:{obj_id} Vehicle cropped with driver padding.")
                            except Exception as e:
                                logging.error(f"Cropping error: {e}")
                                vehicle_crop = full_frame

                    # Generate Event ID
                    event_id = f"VA-ENT-{t_new.strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"

                    # MinIO Upload Path Generation
                    folder_path = f"entries/{t_new.strftime('%Y/%m/%d')}/{event_id}"
                    plate_key = f"{folder_path}/plate.jpg"
                    vehicle_key = f"{folder_path}/vehicle.jpg"

                    plate_uri = ""
                    vehicle_uri = ""

                    if self.minio_client:
                        # Upload Plate Image
                        self.minio_client.upload_bytes(encoded_lp_image, plate_key)
                        plate_uri = self.minio_client.get_public_url(plate_key)

                        # Upload Vehicle Crop (was full frame)
                        if vehicle_crop is not None:
                            encoded_full_image = cv2.imencode('.jpg', vehicle_crop)[1].tobytes()
                            self.minio_client.upload_bytes(encoded_full_image, vehicle_key)
                            vehicle_uri = self.minio_client.get_public_url(vehicle_key)

                    # Build New Payload Format
                    payload_json = {
                        "event_id": event_id,
                        "camera_id": self.config["db"]["camera_id"].get(self.camera_name, self.camera_name),
                        "detected_at": t_new.isoformat(),
                        "vehicle_type": veh_type,
                        "number_plate": clean_string,
                        "offline_entry": False,
                        "number_plate_image": plate_uri,
                        "vehicle_image_url": vehicle_uri
                    }

                    if self.config.get("Collect_full_images") and full_frame is not None:
                        training_parent = "training"
                        folder_name = t_new.strftime(self.fmt[:8])
                        training_path = os.path.join(training_parent, folder_name)
                        if not os.path.exists(training_path):
                            os.makedirs(training_path, exist_ok=True)
                        file_name = t_new.strftime(self.fmt[9:17]) + '-' + clean_string + ".jpg"
                        cv2.imwrite(os.path.join(training_path, file_name), full_frame)

                    logging.info(f"[API] ID:{obj_id} Data: {payload_json}")
                    
                    if self.config['outbound'] is True:
                        success = False
                        if self.rabbitmq_producer:
                            # 1. Try RabbitMQ (Primary)
                            success = self.rabbitmq_producer.publish(payload_json)
                            if success:
                                logging.info(f"[RabbitMQ] Published ID:{obj_id} successfully.")
                            else:
                                logging.info(f"[RabbitMQ] Failed to publish ID:{obj_id}. Adding to retry queue.")
                                response_500.put(payload_json.copy())
                        
                        # 2. Fallback to Direct API if RabbitMQ fails or not configured
                        if not success and is_connected():
                            try:
                                r = requests.post(
                                    url=self.config["db"]["api_endpoint"],
                                    headers={
                                        "Authorization": self.config["db"]["Authorization"],
                                        "Content-Type": "application/json"
                                    },
                                    json=payload_json,
                                    timeout=15
                                )
                                logging.info(f"[API] Status: {r.status_code}")
                                
                                if r.status_code in [200, 201]:
                                    success = True
                                    r_json = r.json()
                                    logging.info(r_json)
                                    if 'application_type' in self.config.keys():
                                        resp_data = r_json.get('data', {})

                                        if resp_data.get('open_barricade') and resp_data.get('visit_entry') is True:
                                        
                                            data = {"vehicle":{"number_plate":clean_string,"owner":{"number_plate":clean_string,"operation":"visitor_barricade","status":"visitor_entry"}}} 
                                            logging.info("\n\n")
                                            logging.info("This Number Plate {} is visitor Web checkin entry in building management system ".format(clean_string))
                                            self.mqtt_publish(data)
                                        elif resp_data.get('open_barricade') and resp_data.get('invite_entry') is True:
                                            data = {"vehicle":{"number_plate":clean_string,"owner":{"number_plate":clean_string,"operation":"invite_barricade","status":"invite_entry"}}} 
                                            logging.info("\n\n")
                                            logging.info("This Number Plate {} is visitor Invite entry in building management system ".format(clean_string))
                                            self.mqtt_publish(data)
                                        elif resp_data.get('open_barricade') is True:
                                            data = {"vehicle":{"number_plate":clean_string,"owner":{"number_plate":clean_string,"operation":"open","status":"resident"}}} 
                                            logging.info("\n\n")
                                            logging.info("This Number Plate {} is registered in Visitor management system".format(clean_string))
                                            self.mqtt_publish(data)
                                        elif self.config['application_type'][self.camera_name] == 'normal':
                                            self.mqtt_publish(resp_data)
                                            
                                    try:
                                        if response_500.qsize() > 0:
                                            for _ in range(response_500.qsize()):
                                                r500_data = response_500.get()
                                                logging.info("\n\n")
                                                r500_data[0]['vehicle_entries[offline_entry]']= bool(True)
                                                #self.payload['vehicle_entries[offline_entry]'] = bool(True)
                                                logging.info("retrying datas ----->>>>>>{}".format(r500_data[0]))
                                                r500 = requests.post(url=self.config["db"]["api_endpoint"],
                                                                    headers={"Authorization": self.config["db"]["Authorization"]},
                                                                    files=r500_data[1], data=r500_data[0],timeout=15)
                                                if r500.status_code == 201:
                                                    logging.info("WoW..! Db is published of previous data successfully with response code of 201")
                                                    r500 = r500.json()
                                                    logging.info(r500)

                                    except KeyError:
                                        logging.info("Oops...! failed data for {} is not found".format(self.camera_name))
                                else:
                                    logging.info(r)
                                    response_500.put(payload_json.copy())
                            except Exception as e:
                                logging.info(f"API Request Exception: {e}")
                                response_500.put(payload_json.copy())
                        elif not success:
                            logging.info('Internet is disconnected')
                            response_500.put(payload_json.copy())
                    write_fun(number_plate_image, clean_string, self.fmt, t_new,
                              self.config["camera_numberplate_path"][self.camera_name], ocr_batch_results, lp_confs)
                else:
                    logging.info(f"[Dedupe] ID:{obj_id} Same car detected: {clean_string}")
                logging.info("\n\n\n\n\n")

            # Explicitly free up memory for large image objects
            del number_plate_image
            del full_frame

        except Exception as e:
            logging.info("ocr_processing Exception : {}".format(e), exc_info=1)
            if 'payload_json' in locals():
                response_500.put(payload_json.copy())
            # Ensure cleanup happens even on exception
            if 'number_plate_image' in locals(): del number_plate_image
            if 'full_frame' in locals(): del full_frame
