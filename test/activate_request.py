import json
import requests
import time
def send_request(url, data):
    try:
        headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "Accept-Charset": "utf-8"
        }
        if data is not None:
            response = requests.post(url, data=json.dumps(data, ensure_ascii=False).encode('utf-8'), headers=headers)
        else:
            response = requests.get(url, headers=headers)
        if response.status_code in [200,202]:
            #print(response.text)
            return True
            
        else:
            print("Data sent failed! Code:", response.status_code, response.text)
            return False
    except Exception as e:
        print("Data sent error:", e)
        return False

url = "http://127.0.0.1:52001/call_function"

image_path_list = ["road.jpg"]
video_path = "cv_demo_short.mp4"
query_list = ["person"]

FUNC_EXEC_MAPPING = {
    "preprocess": {
        "image_path_list":image_path_list, 
        "resize_dims": (256,256),
        "crop_coords": (1,1,255,255), 
        "rotate_angle": 2.05, 
        "rotate_center":(128,128), 
        "rotate_scale": 1.1,
        "flip_code": -1, 
        "affine_translation":(128,128), 
        "color_conversion": 6
        },
    "postprocess": {
        "image_path_list":image_path_list,  
        "blur_type":'median', 
        "blur_ksize":[3,3], 
        "canny_thresholds":[23, 116],
        "equalize_hist":True, 
        "thresholding":[188, 255, 0],  
        "draw_contours":True,
        "morphology": [0, [3, 3], 1],
        "sobel_edges":[1, 1, 3],
        "laplacian_edges":True,
        "sharpen":True,
        "add_noise":True,
        "change_brightness_contrast":[3, 5]
        },
    "detection": {
        "image_path_list":image_path_list,
        "query_list":query_list, 
        "method":"dino", 
        "confidence_threshold":0.5,
        "iou_threshold":0.4,
        "image_resize":[512,512],
        "fp16_inference":True,
        "max_bboxes":100,
        "high_resolution":False
       },
    "segmentation": {
        "image_path_list":image_path_list,
        "query_list":query_list,
        "method":"lan_sam",
        "bbox_threshold":0.3,
        "text_threshold":0.25,
        },
    "classification": {
        "image_path_list":image_path_list,
        "query_list":query_list,
        "method":"dino",
        "confidence_threshold":0.5,
        "iou_threshold":0.4
        },
    "counting": {
        "image_path_list":image_path_list,
        "query_list":query_list
    },
    "tracking": {
        "video_path":video_path,
        "query_list":query_list
    },
    "pose": {
        "video_path":video_path
    },
    "optical_flow": {
        "video_path":video_path
    },  
    "ocr": {
        "image_path_list":image_path_list,  
        "bbox_region":[["1.0122", "1.5678", "233.1731", "257.17776", "0.6756495", "text"]],
        "ocr_threshold":0.1
    },
    "vlm": {
        "image_path":image_path_list[0], 
        "query":"描述图片"
    },
    "llm": {
        "query":["先告诉我你是谁，然后回答我","1+","3","=","?"]
    },
    "alarm": {
        "result_obj":{ 
            "image_path_list":["/models/baohan/data/images/road.jpg"],
            "boxes_list":[[["28.19196","66.53552","46.9448","9.20325","0.37918216","people"]]]
        },
        "detect_region":["1.0122", "1.5678", "133.1731", "157.17776"],
        "detect_class" : "people",
        "detect_method":"appear",
        "notice_email_or_tel":"example@example_zxtt93dueww.com",
        "alarm_threshold":1
    },
    "output": {
        "result_obj":{
            "image_path_list":image_path_list,
            "boxes_list":[[1,1,200,200]],
        },
        "method":"show"
    },
    "videoprecess": {
        "video_path":video_path,
        "interval":1,
        "method":"get_frames"
    }
}


for function_name, function_args in FUNC_EXEC_MAPPING.items():
    start_time = time.time()
    data = {
        "function_name" : function_name,
        "arguments" : function_args,
        "only_local" : True
        }
    result = send_request(url, data)
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"[SUCCESS] {function_name}:{exec_time}") if result else print(f"[FAILED] {function_name}:{exec_time}")

