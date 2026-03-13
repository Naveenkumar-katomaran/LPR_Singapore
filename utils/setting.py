
import logging
import os
from posixpath import pardir
import requests
from time import strftime
from requests.auth import HTTPDigestAuth
from multiprocessing import Process 
import schedule
import json
from pytz import timezone, utc
from datetime import datetime

def custom_time(*args):
    utc_dt = utc.localize(datetime.utcnow())
    my_tz = timezone("Singapore")
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()

headers={'Accept-Encoding': 'identity'}

'''
Video color config : /cgi-bin/configManager.cgi?action=getConfig&name=VideoColor [GET]
                     /cgi-bin/configManager.cgi?action=setConfig&VideoColor[0][0].Brightness=80 [SET]

VideoInOptions : 

Get videoinputcaps : /cgi-bin/devVideoInput.cgi?action=getCaps&channel=1

Get VideoInOptionsConfig : /cgi-bin/configManager.cgi?action=getConfig&name=VideoInOptions
                           /cgi-bin/configManager.cgi?action=setConfig&VideoInOptions[0].DayNightColor=1

/cgi-bin/configManager.cgi?action=getConfig&name=VideoInFocus
/cgi-bin/configManager.cgi?action=getConfig&name=VideoInExposure

'''

SET_CONFIG_URL = "cgi-bin/configManager.cgi?action=setConfig&"

## Exposure Settings - Anti flicker and shutter priority
'''
shutter priority modes

0: "Auto" by default
1: Low noise
2: Anti-smear
4: Manual (range)
5: Aperture priority
6: Manual (fixed)
7: Gain priority
8: Shutter priority
9: Flash light matching mode
'''
GET_CONFIG_URL = "cgi-bin/configManager.cgi?action=getConfig&name=VideoInOptions"

def set_exposure(config,camera_name):
    
    url = config["url"][camera_name] + SET_CONFIG_URL + "VideoInExposure[0][0].AntiFlicker=0" + "&" +   "VideoInExposure[0][1].Mode=8" +"&"+ "VideoInExposure[0][1].Value1="+config["exposure"][camera_name]["value1"] + "&" + "VideoInExposure[0][1].Value2="+config["exposure"][camera_name]["value2"]
    #url = "http://riverislesanpr.dvrdns.org:8081/" + SET_CONFIG_URL + "VideoInExposure[0][0].AntiFlicker=0" + "&" +   "VideoInExposure[0][1].Mode=6"
    try:
        response = requests.get(url, auth=HTTPDigestAuth("admin", "admin0864"),timeout=10)
   
        if response.status_code == 200:
            logging.info(response.text)
            logging.info("Exposure is updated")
    except requests.Timeout as err:
        logging.error({"message": err.message})
    except Exception as e:
        logging.info("Oops...!!! Exposure setup failed")

def set_schedule_time(config,camera_name):
    url = config["url"][camera_name] + SET_CONFIG_URL + "VideoInOptions[0].NightOptions.SunriseHour="+config["schedule_time"][camera_name]["sunrisehour"] + "&" + "VideoInOptions[0].NightOptions.SunriseMinute="+config["schedule_time"][camera_name]["sunriseminute"]+"&"+"VideoInOptions[0].NightOptions.SunriseSecond="+config["schedule_time"][camera_name]["sunriseseconds"]+"&"+"VideoInOptions[0].NightOptions.SunsetHour="+config["schedule_time"][camera_name]["sunsethour"]+"&"+"VideoInOptions[0].NightOptions.SunsetMinute="+config["schedule_time"][camera_name]["sunsetminute"]+"&" + "VideoInOptions[0].NightOptions.SunsetSecond="+config["schedule_time"][camera_name]["sunsetseconds"]
    #print(url)
    try:
        response = requests.get(url, auth=HTTPDigestAuth("admin", "admin0864"),timeout=10)
        if response.status_code == 200:
            logging.debug("Schedule Time is updated")
            logging.info(response.text)
    except requests.Timeout as err:
        logging.error({"message": err.message})
    except Exception as e:
        logging.info("Oops...!!! Time management setup failed")
## Day and Night mode - {"Color", "Brightness", "BlackWhite}

def set_daynight_mode(config,camera_name):
    url = config["url"][camera_name] + "cgi-bin/configManager.cgi?action=setConfig&VideoInDayNight[0][0].Mode="+config["brightness"][camera_name]["mode"]
    #print(url)
    try:
        response = requests.get(url, auth=HTTPDigestAuth("admin", "admin0864"),timeout=10)
        print(response)
        if response.status_code == 200:
            logging.info("Day and Night mode blackandwhite updated")
            logging.info(response.text)
    except requests.Timeout as err:
        logging.error({"message": err.message})
    except Exception as e:
        logging.info("Oops...!!! Daynight setup failed")
## IR Light configure


def set_autoir(config,camera_name):
    #response = requests.get("http://riverislesanpr.dvrdns.org:8081/cgi-bin/configManager.cgi?action=getConfig&name=VideoInDayNight", auth=HTTPDigestAuth("admin", "admin0864"))      
    try:
        response = requests.get(config["url"][camera_name] + "cgi-bin/configManager.cgi?action=setConfig&Lighting[0][0].Mode=Manual&Lighting[0][0].MiddleLight[0].Light=24",auth=HTTPDigestAuth("admin", "admin0864"),timeout=10)
        #http://118.189.158.250:8081/cgi-bin/configManager.cgi?action=setConfig&Lighting[0][0].Mode=Manual&Lighting[0][0].MiddleLight[0].Light=24
        # logging.info("Camera Response ::: {}".format(response))
        
        if response.status_code == 200:
            logging.info(response.text)
    except requests.Timeout as err:
        logging.error({"message": err.message})
    except Exception as e:
        logging.info("Oops...!!! IR setup failed")

def set_zoomfocus(config,camera_name):
    try:
        res = requests.get(url= config["url"][camera_name] + "cgi-bin/devVideoInput.cgi?action=getFocusStatus&channel=1", auth=requests.auth.HTTPDigestAuth("admin", "admin0864"),timeout=10)
        tet = res.text.split()
        focus_percentage = tet[0].split('=')[1]
        
        value = config["zoom_focus"][camera_name]["zoom_value"]
        total_value = int(tet[-1].split('=')[1])
        zoom_percentage = value/total_value
        
        logging.info(zoom_percentage)
        
        url = config["url"][camera_name] + "cgi-bin/devVideoInput.cgi?action=adjustFocus&focus={}&zoom={}".format(focus_percentage,zoom_percentage)
        response = requests.get(url, auth=requests.auth.HTTPDigestAuth("admin", "admin0864"),timeout=10)
        logging.info(response)
        if response.status_code == 200:
            logging.info("Zoom and Focus is updated")
            logging.info(response.text)
    except requests.Timeout as err:
        logging.error({"message": err.message})
    except Exception as e:
        logging.info("Oops...!!! ZoomFocus setup failed")
def set_fps(config,camera_name):
    
    frame_rate = config["fps"][camera_name]["fps_value"]
    try:
        url= config["url"][camera_name] + "cgi-bin/configManager.cgi?action=setConfig&Encode[0].MainFormat[0].Video.FPS={}".format(frame_rate)
        response = requests.get(url, auth=HTTPDigestAuth("admin", "admin0864"),timeout=10)
        logging.info(url)
        if response.status_code == 200:
            logging.info(response.text)
            logging.info("frame rate fps is updated")
    except requests.Timeout as err:
        logging.error({"message": err.message})
    except Exception as e:
        logging.info("Oops...!!! FPS setup failed")
def set_time(config,camera_name):

    timetitle = strftime("%Y-%m-%d %H:%M:%S", custom_time())
    logging.info(timetitle)
    url = config["url"][camera_name] + "cgi-bin/global.cgi?action=setCurrentTime&time={}".format(timetitle)
    try:
        res =  requests.get(url, auth=HTTPDigestAuth("admin", "admin0864"),timeout=10)
        if res.status_code == 200:
            logging.info("time title is updated")
            logging.info(res.text) 
        else:
            print(res.text) 
    except requests.Timeout as err:
        logging.error({"message": err.message})
    except Exception as e:
        logging.info("Oops...!!! TIME update setup failed")