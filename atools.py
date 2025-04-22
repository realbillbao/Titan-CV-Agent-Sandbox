import os 
import yaml
from typing import List, Dict,Tuple,Union
import shutil
import json
import requests
import uuid
import requests
import zipfile
from urllib.parse import urljoin
import logging
from logging.handlers import TimedRotatingFileHandler
from base.base_config import *

def get_config(config_path:str = None, keys:List = []) -> str:
    config_path = "config/config.yml" if config_path == None else config_path
    with open(config_path, "r") as conf:
        conf_data = yaml.safe_load(conf)
    return conf_data[keys[0]][keys[1]]

def get_models(config_path: str = None, keys: List = []) -> str:
    config_path = "model_config.yml" if config_path is None else config_path
    
    with open(config_path, "r") as conf:
        conf_data = yaml.safe_load(conf)
    
    value = conf_data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    
    return value

def get_available_models(mapping_path:str = None, query_list:List = []) -> str:
    mapping_path = get_config(keys=["detection", "mapping_path"]) if mapping_path == None else mapping_path

    with open(mapping_path, "r") as map:
        map_data = yaml.safe_load(map)
    
    for model_type in ["base_models", "trained_models", "custom_models"]:  
        for model_id, classes in map_data[model_type].items():
            if set(query_list) <= set(classes):
                return os.path.join(get_config(keys=["detection", model_type+"_"+"path_preffix"]), model_id) + ".pt"
    return None

def update_yaml(yaml_path:str = None,kv:List =["","",""]):
    if yaml_path is None or kv ==["","",""]:
        print("Empty yaml path or key&value!")
        return
    with open(yaml_path, "r") as y:
        yaml_data = yaml.safe_load(y)
    yaml_data[kv[0]][kv[1]]=kv[2]
    with open(yaml_path, "w") as y:
        yaml.dump(yaml_data, y, default_flow_style=False, sort_keys=False)

def read_yaml(yaml_path:str = None):
    if yaml_path is None:
        return None
    with open (yaml_path, "r") as y:
        return yaml.safe_load(y)

def copy_dir(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)        
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

def remove_file(file_path:str = None):
    if os.path.exists(file_path):
        os.remove(file_path)

def remove_dir(file_dir:str = None):
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)

def write_log(log_path:str, log_data:Dict):
    with open(log_path,"a") as log:
        recode = json.dumps(log_data)
        log.write(recode + '\n') 

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def convert_bbox_to_yolo(img_width, img_height, bbox):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return (x_center, y_center, width, height)

def post_request(url, data):
    try:
        headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "Accept-Charset": "utf-8"
        }
        response = requests.post(url, data=json.dumps(data, ensure_ascii=False).encode('utf-8'), headers=headers)
        if response.status_code == 200:
            print("Sent request to " + url)
            return response.text
        else:
            print("Err:", response.status_code)
    except Exception as e:
        print("Err:", e)

def merge_dict_list(dic_list):
    if dic_list is None or dic_list==[]:
        return {}
    merged_set = set() 
    for dict_obj in dic_list:
        for k,v in dict_obj.items():
            merged_set.update({v})
    return {idx:v for idx,v in enumerate(merged_set)}

def return_empty_result(input_triggered_paramter:List = None):
    if input_triggered_paramter is not None:
        for input_para in input_triggered_paramter:
            if input_para in [[],None,"",{}]:
                return True
    return False

def get_yolo_result(results, save_dir=None):
    image_path_list=[]
    predict_img_list=[]
    boxes_list=[]
    classes_name_list=[]
    predict_speeds_list=[]
    video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'm4v', 'webm', 'mpeg', 'mpg']
    is_video = True if isinstance(results[0].path, str) and results[0].path.split(".")[-1] in video_extensions else False
    
    for idx, r in enumerate(results):
        
        if save_dir and is_video:
            predict_img = os.path.join(save_dir, f"frame_{str(idx)}.jpg")
        else:
            predict_img = get_output_image_uuid_name()
        r.save(filename = predict_img)
    
        single_predict_boxes=[]
        for box in r.boxes:
            raw_box = box.data.tolist()[0]
            if len(raw_box) == 7:       # The object ID is also returned -> delete
                raw_box = raw_box[:4]+raw_box[-2:]
            raw_box_class_id = int(raw_box.pop())
            raw_box.append(r.names[raw_box_class_id])
            single_predict_boxes.append(raw_box)

        image_path_list.append(r.path)
        predict_img_list.append(predict_img)
        boxes_list.append(single_predict_boxes)
        #classes_name_list.append(r.names)
        if isinstance(r.names, dict):
            classes_name_list.append(r.names)
        elif isinstance(r.names, list):
            classes_name_list.append({i: name for i, name in enumerate(r.names)})
        predict_speeds_list.append(r.speed)
    
    if image_path_list and len(set(image_path_list))==1 and is_video:
        image_path_list = [image_path_list[0]]
    
    results_dict = {
        "image_path_list": image_path_list,
        "predict_img_list": predict_img_list,
        "boxes_list": boxes_list,
        "classes_name_dict": merge_dict_list(classes_name_list),
        "predict_speeds_dict": predict_speeds_list[-1]
    }
    
    return results_dict

def get_output_image_uuid_name():
    return os.path.join(BASE_LOCAL_OUTPUT_PREFIX, str(uuid.uuid4())+".jpg")


def zip_files(file_paths):
    file_paths = [file_paths] if isinstance(file_paths,str) else file_paths
    zip_path = os.path.join(BASE_LOCAL_OUTPUT_PREFIX, str(uuid.uuid4())+".zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in file_paths:
            if os.path.isfile(file_path):
                zipf.write(file_path, arcname=os.path.basename(file_path))
            else:
                print(f"Warning: '{file_path}' does not exist and was not added to the ZIP file.")
    return zip_path
        

def zip_folder(folder_path):
    zip_path = os.path.join(BASE_LOCAL_OUTPUT_PREFIX, str(uuid.uuid4())+".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))
    return zip_path

def write_random_name_txt(result_obj:str) -> str:
    txt_path = os.path.join(BASE_LOCAL_OUTPUT_PREFIX, str(uuid.uuid4())+".txt")
    with open(txt_path, 'w') as file:
        file.write(result_obj)
    return txt_path

def calculate_iou(bbox1, bbox2):
    bbox1 = list(map(float, bbox1))
    bbox2 = list(map(float, bbox2))
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    x_inter_min = max(x_min1, x_min2)
    y_inter_min = max(y_min1, y_min2)
    x_inter_max = min(x_max1, x_max2)
    y_inter_max = min(y_max1, y_max2)
    width_inter = max(0, x_inter_max - x_inter_min)
    height_inter = max(0, y_inter_max - y_inter_min)
    area_inter = width_inter * height_inter
    area_bbox1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area_bbox2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    area_union = area_bbox1 + area_bbox2 - area_inter
    iou = area_inter / area_union if area_union > 0 else 0
    return iou

def return_wrapper(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return {
                "code": 1,
                "message": "success",
                "data": result
            }
        except Exception as e:
            return {
                "code": -1,
                "message": str(e),
                "data": None
            }
    return wrapper

def convert_string_to_list(input_str:Union[str,List[str]]) -> List:
    return [input_str] if isinstance(input_str, str) else input_str


def download_url_images(image_path_list, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for index, image_url in enumerate(image_path_list):
        try:
            response = requests.get(image_url)
            response.raise_for_status()  
            image_name = os.path.join(save_folder, f'image_{index + 1}.jpg')
            with open(image_name, 'wb') as file:
                file.write(response.content)
        except Exception as e:
            print(f"err {image_url}: {e}")

def setup_logger(log_file: str, log_level=logging.INFO, when="W6", interval=1, backup_count=4) -> logging.Logger:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = TimedRotatingFileHandler(log_file, when=when, interval=interval, backupCount=backup_count, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger



