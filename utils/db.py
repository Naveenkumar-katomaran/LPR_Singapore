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

logging = log.getLogger('Rotating Log')
response_500 = Queue()
owner_details = None


def write_fun(image, string, fmt, d_t, parent_folder_name, raw_data, lp_confs):
    folder_name = d_t.strftime(fmt[:8])
    file_name = d_t.strftime(fmt[9:17]) + '-' + string + ".jpg"
    file_path = "{}/{}/{}".format(parent_folder_name, folder_name, file_name)
    if not os.path.exists(parent_folder_name + '/' + folder_name):
        os.system("mkdir {}/{}".format(parent_folder_name, folder_name))
    cv2.imwrite(file_path, image)
    try:
        h, w = image.shape[:2]
        if len(raw_data) == 0:
            image_insights = {"height": h, "width": w, "time": d_t.strftime(fmt), "lp_confs":lp_confs}
        else:
            image_insights = {"height": h, "width": w, "time": d_t.strftime(fmt),"checksum": raw_data[-1], "lp_confs":lp_confs}
            # print(raw_data)
            for i in range(len(raw_data)-1):
                image_insights[i] = [raw_data[i][0]] + raw_data[i][1] + [raw_data[i][2]]
            
        with open(file_path[:-3] + 'json', 'w') as outfile:
                json.dump(image_insights, outfile, indent=2)
    except Exception as e:
        logging.info(f"Write file Exception : {e}", exc_info=1)
        


def create_plot(folder_name):
    fmt = f"{folder_name}/%d-%m-%Y/"
    real_path = datetime.now().strftime(fmt)
    logging.info(f"Path for plot {real_path}")
    if not os.path.exists(real_path):
        os.system("mkdir {}".format(real_path))
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

def db_hit_status(string1, string2, old_time, time_to_fly):
    if string2 is None:
        return True, string1, time.time()
    else:
        if SequenceMatcher(None, string1[:-1], string2[:-1]).ratio() > 0.8:
            if time.time() - old_time > time_to_fly:
                return True, string1, time.time()
            else:
                return False, string2, time.time()
        else:
            return True, string1, time.time()


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
        self.old_string = None
        self.new_string = None
        self.time = None
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
        self.fmt = "%d-%m-%Y %H:%M:%S %z"
    
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

    def text_process(self, number_plate_image, number_string, raw_data, lp_confs, full_frame=None):
        payload = {}
        files = {}
        try:
            if self.config['bike_lnpr'][self.camera_name] is True:
                bike_cond = 4 <= len(number_string) <= 12
            else:         
                bike_cond = 7 <= len(number_string) <= 12
            if bike_cond:
                encoded_lp_image = cv2.imencode('.jpg', number_plate_image)[1].tobytes()
                self.new_string = number_string
                now_utc = datetime.now(timezone('UTC'))
                t_new = now_utc.astimezone(timezone("Asia/Kolkata"))
                status, self.old_string, self.time = db_hit_status(self.new_string, self.old_string, self.time, self.time_to_fly)
                veh_type = "normal"
                if status:
                    if self.config.get("Collect_full_images") and full_frame is not None:
                        training_parent = "training"
                        folder_name = t_new.strftime(self.fmt[:8])
                        training_path = os.path.join(training_parent, folder_name)
                        if not os.path.exists(training_path):
                            os.makedirs(training_path, exist_ok=True)
                        file_name = t_new.strftime(self.fmt[9:17]) + '-' + self.old_string + ".jpg"
                        cv2.imwrite(os.path.join(training_path, file_name), full_frame)

                    self.payload['vehicle_entries[detected_time]'] = t_new.strftime(self.fmt)
                    self.payload['vehicle_entries[vehicle_type]'] = veh_type
                    self.payload['vehicle_entries[number_plate]'] = self.old_string
                    self.payload['vehicle_entries[offline_entry]'] = bool(False)
                    self.files['vehicle_entries[number_plate_image]'] = (
                        self.old_string + t_new.strftime(self.fmt) + '-' + '.jpg', encoded_lp_image)

                    logging.info(f"Data  :  {self.payload}")
                    payload = self.payload.copy()
                    files = self.files.copy()
                    if self.config['outbound'] is True:
                        if 'application_type' in self.config.keys():
                            if self.config['application_type'][self.camera_name] == 'resident':
                                if self.old_string in eval(owner_details): 
                                    data = {"vehicle":{"number_plate":self.old_string,"owner":{"number_plate":self.old_string,"operation":"open","status":"resident"}}} 
                                    logging.info("\n\n")
                                    logging.info("This Number Plate {} is present in Resident details".format(self.old_string))
                                    self.mqtt_publish(data)
                                else:
                                    logging.info("\n\n")
                                    logging.info("This Number Plate {} is not present in Resident details".format(self.old_string))                        

                        if is_connected() is True:
                            r = requests.post(url=self.config["db"]["api_endpoint"],
                                            headers={"Authorization": self.config["db"]["Authorization"]},
                                            files=files, data=payload,timeout=15)
                            if self.config["test_db"]["status"] is True:
                                test_db_res = requests.post(url=self.config["test_db"]['url'],
                                                            headers={"Authorization": self.config["test_db"]['api_key']},
                                                            files=files,
                                                            data=payload,timeout=15)
                                logging.info(f"For test db : {test_db_res.json()}")
                            if r.status_code == 201:
                                logging.info("WoW..! Db is published successfully with response code of 201")
                                r = r.json()
                                logging.info(r)
                                if 'application_type' in self.config.keys():
                                    resp_data = r.get('data', {})

                                    if resp_data.get('open_barricade') and resp_data.get('visit_entry') is True:
                                    
                                        data = {"vehicle":{"number_plate":self.old_string,"owner":{"number_plate":self.old_string,"operation":"visitor_barricade","status":"visitor_entry"}}} 
                                        logging.info("\n\n")
                                        logging.info("This Number Plate {} is visitor Web checkin entry in building management system ".format(self.old_string))
                                        self.mqtt_publish(data)
                                    elif resp_data.get('open_barricade') and resp_data.get('invite_entry') is True:
                                        data = {"vehicle":{"number_plate":self.old_string,"owner":{"number_plate":self.old_string,"operation":"invite_barricade","status":"invite_entry"}}} 
                                        logging.info("\n\n")
                                        logging.info("This Number Plate {} is visitor Invite entry in building management system ".format(self.old_string))
                                        self.mqtt_publish(data)
                                    elif resp_data.get('open_barricade') is True:
                                        data = {"vehicle":{"number_plate":self.old_string,"owner":{"number_plate":self.old_string,"operation":"open","status":"resident"}}} 
                                        logging.info("\n\n")
                                        logging.info("This Number Plate {} is registered in Visitor management system".format(self.old_string))
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
                                response_500.put([payload.copy(), files.copy()])
                        else:
                            logging.info('Internet is disconnected')
                            response_500.put([payload.copy(), files.copy()])
                    write_fun(number_plate_image, self.old_string, self.fmt, t_new,
                              self.config["camera_numberplate_path"][self.camera_name], raw_data, lp_confs)
                else:
                    logging.info("Same car detected")
                logging.info("\n\n\n\n\n")

            # Explicitly free up memory for large image objects
            del number_plate_image
            del full_frame

        except Exception as e:
            logging.info("ocr_processing Exception : {}".format(e), exc_info=1)
            response_500.put([payload.copy(), files.copy()])
            # Ensure cleanup happens even on exception
            if 'number_plate_image' in locals(): del number_plate_image
            if 'full_frame' in locals(): del full_frame

