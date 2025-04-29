import os
import re
import sys
import time
import base64
import random
import numpy as np
from typing import List, Dict, Literal

import cv2
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api import get_models
yolo_world_model, langsam_model, groundingdino_model, tracking_model, pose_model, ocr_tool = get_models()

from ultralytics import YOLO
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from base_config import *
from atools import *

def get_response(url:str = None, data:dict = None):
    if url is None or data is None:
        raise ValueError("Both 'url' and 'data' cannot be None!")
    try:
        if ACCESS_KEY is not None and ACCESS_KEY!="":
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Accept-Charset": "utf-8",
                "Authorization": f"Bearer {ACCESS_KEY}"
            }
        else:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "Accept-Charset": "utf-8"
            }
        response = requests.post(url, data=json.dumps(data, ensure_ascii=False).encode('utf-8'), headers=headers)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print(f"Data sent failed! Code:{response.status_code}" )
    except Exception as e:
        print(f"Data sent error:{e}")


# Bask class for most task, define available gpu or other required system parameter.
class TaskBase:
    def __init__(self) -> None:
        gpu_id = os.environ.get('GPU_ID',0)
        self.device = f"cuda:{gpu_id}"
        print(f"Use GPU:{self.device}")

class Preprocess:
    def __init__(self,
        image_path_list:Union[str, List[str]] = None, 
        resize_dims:Tuple[int, int] = None, 
        crop_coords:Tuple[int, int, int, int] = None, 
        rotate_angle:float = None, 
        rotate_center:Tuple[int, int] = None, 
        rotate_scale:float = None,
        flip_code:int = None, 
        affine_translation:Tuple[int, int] = None, 
        color_conversion:int = None
    ) -> None:
        
        self.save_path_list = []
        self.image_path_list = convert_string_to_list(image_path_list)

        self.resize_dims = tuple(resize_dims) if isinstance(resize_dims, list) else resize_dims
        self.crop_coords = tuple(crop_coords) if isinstance(crop_coords, list) else crop_coords
        self.rotate_center = tuple(rotate_center) if isinstance(rotate_center, list) else rotate_center
        self.affine_translation = (tuple(affine_translation) if isinstance(affine_translation, list) else affine_translation)
        
        self.rotate_angle = rotate_angle
        self.rotate_scale = rotate_scale
        self.flip_code = flip_code
        self.color_conversion = color_conversion     
    
    def run(self):

        if return_empty_result([self.image_path_list]):
            return {"preprocessed_path_list":[]}
        
        for image_path in self.image_path_list:
            # 0. Read image
            src = cv2.imread(image_path)
            if src is None:
                raise FileNotFoundError(f"Image not found at the path: {image_path}")

            # 2. Crop
            if self.crop_coords is not None:
                x, y, width, height = self.crop_coords
                src = src[y:y+height, x:x+width]

            # 3. Rotate
            if self.rotate_angle is not None and self.rotate_scale is not None:
                if self.rotate_center is None:
                    self.rotate_center = (src.shape[1] // 2, src.shape[0] // 2)
                M = cv2.getRotationMatrix2D(self.rotate_center, self.rotate_angle, self.rotate_scale)
                src = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))

            # 4. Flip
            if self.flip_code is not None:
                src = cv2.flip(src, self.flip_code)

            # 5. Affine Translate
            if self.affine_translation is not None:
                tx, ty = self.affine_translation
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                src = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))

            # 1. Resize
            if self.resize_dims is not None:
                src = cv2.resize(src, self.resize_dims, interpolation=cv2.INTER_LINEAR)

            # 6. Color Conversion
            if self.color_conversion is not None and len(src.shape) == 3:
                src = cv2.cvtColor(src, self.color_conversion)

            # save image
            save_path = get_output_image_uuid_name()
            cv2.imwrite(save_path, src)
            self.save_path_list.append(save_path)

        return {"preprocessed_path_list":self.save_path_list}


class Postprocess:
    def __init__(self,
        image_path_list:Union[str, List[str]] = None,  
        blur_type:Literal['gaussian', 'median', 'mean'] = None, 
        blur_ksize:Tuple[int, int] = None, 
        canny_thresholds:List[int] = None,
        equalize_hist:bool = False, 
        thresholding:Tuple[int, int, int] = None,  
        draw_contours:bool = False,
        morphology:Tuple[int, Tuple[int, int], int] = None,
        sobel_edges:Tuple[int, int, int] = None,
        laplacian_edges:bool = False,
        sharpen:bool = False,
        add_noise:bool = False,
        change_brightness_contrast:Tuple[int, int] = None,
    ) -> None:
        
        self.save_path_list = []
        self.image_path_list = convert_string_to_list(image_path_list)
        
        self.blur_ksize = tuple(blur_ksize) if isinstance(blur_ksize, list) else blur_ksize
        self.canny_thresholds = tuple(canny_thresholds) if isinstance(canny_thresholds, list) else canny_thresholds
        self.thresholding = tuple(thresholding) if isinstance(thresholding, list) else thresholding
        self.morphology = tuple(morphology) if isinstance(morphology, list) else morphology
        self.sobel_edges = tuple(sobel_edges) if isinstance(sobel_edges, list) else sobel_edges
        self.change_brightness_contrast = (tuple(change_brightness_contrast) if isinstance(change_brightness_contrast, list) else change_brightness_contrast)

        self.blur_type = blur_type
        self.equalize_hist = equalize_hist
        self.draw_contours = draw_contours
        self.laplacian_edges = laplacian_edges
        self.sharpen = sharpen
        self.add_noise = add_noise
        
    def run(self):
        if return_empty_result([self.image_path_list]):
            return {"preprocessed_path_list":[]}
        
        for image_path in self.image_path_list:
            # 0. Read image
            src = cv2.imread(image_path)
            if src is None:
                raise FileNotFoundError(f"Image not found at the path: {image_path}")
            
            if self.blur_type is not None and self.blur_ksize is not None:
                # 1. Blur
                if self.blur_type == 'gaussian':
                    src = cv2.GaussianBlur(src, self.blur_ksize, 0)
                elif self.blur_type == 'median':
                    src = cv2.medianBlur(src, self.blur_ksize[0])
                elif self.blur_type == 'mean':
                    src = cv2.blur(src, self.blur_ksize)

            # 2. Canny Edge Detection
            if self.canny_thresholds is not None:
                src = cv2.Canny(src, self.canny_thresholds[0], self.canny_thresholds[1])

            # 3. Histogram Equalization
            if self.equalize_hist:
                if len(src.shape) == 3:
                    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                src = cv2.equalizeHist(src)

            # 4. Thresholding
            if self.thresholding is not None:
                thresh, maxval, threshold_type = self.thresholding
                _, src = cv2.threshold(src, thresh, maxval, threshold_type)

            # 6. Find Contours
            if self.draw_contours:
                if len(src.shape) == 3:
                    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                _, src = cv2.threshold(src, 30, 255, cv2.THRESH_BINARY)
                contour_image = np.zeros_like(src)
                contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)


            # 7. Morphological Operations
            if self.morphology is not None and len(self.morphology) == 3: 
                if len(src.shape) == 3:          
                    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                _, src = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)
                op_type, kernel, iterations = self.morphology
                if isinstance(kernel,list) and len(kernel) == 2:
                    kernel=(kernel[0],kernel[1])
                src = cv2.morphologyEx(src, op_type, kernel, iterations=iterations)


            # 8. Sobel Edge Detection
            if self.sobel_edges is not None:
                dx, dy, ksize = self.sobel_edges
                sobelx = cv2.Sobel(src, cv2.CV_16S, dx, 0, ksize=ksize)
                sobely = cv2.Sobel(src, cv2.CV_16S, 0, dy, ksize=ksize)
                sobelx = np.float32(sobelx)
                sobely = np.float32(sobely)
                src = cv2.magnitude(sobelx, sobely)

            # 9. Laplacian Edge Detection
            if self.laplacian_edges:
                ddepth = cv2.CV_32F
                src = cv2.Laplacian(src, ddepth)

            # 10. Sharpening
            if self.sharpen:
                kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
                src = cv2.filter2D(src, -1, kernel)

            # 11. Add Noise
            if self.add_noise:
                noise = np.random.randint(0, 256, src.shape, dtype=np.uint8).astype(np.float32)
                src = cv2.add(src, noise)

            # 12. Change Brightness and Contrast
            if self.change_brightness_contrast is not None:
                brightness, contrast = self.change_brightness_contrast
                src = np.int16(src)
                src = src * (contrast/127+1) - contrast + brightness
                src = np.clip(src, 0, 255)
                src = np.uint8(src)

            # save image
            save_path = get_output_image_uuid_name()
            cv2.imwrite(save_path, src)
            self.save_path_list.append(save_path)
        return {"postprocessed_path_list":self.save_path_list}

class _Visualization:
    def plot_bboxes(self, original_img_path, predict_boxes, output_img_path):
        img = cv2.imread(original_img_path)
        color_list = [(255, 42, 4),(79, 68, 255),(255, 0, 189),(255, 180, 0),(186, 0, 221),
            (0, 192, 38),(255, 36, 125),(104, 0, 123),(108, 27, 255),(47, 109, 252),(104, 31, 17)]
        color = color_list[random.randint(0, len(color_list)-1)]
        for box in predict_boxes:
            x1, y1, x2, y2, score, label = box
            x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
            txt_color = (255, 255, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} {float(score):.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            outside = y1 - h >= 0
            label_position = (x1, y1 if outside else y1 + h)
            cv2.rectangle(img, (x1, y1 - h if outside else y1), (x1 + w, y1), color, -1)
            cv2.putText(img, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)
        cv2.imwrite(output_img_path, img)

    def plot_masks(self, input_masks, predict_img, im_gpu, mask_color=(255,255,0), alpha=0.5):
        """
        Plot masks on an image.

        Args:
            masks (numpy.array): Predicted masks, shape: [n, h, w]
            im_gpu (numpy.array): Original image in numpy array, shape: [h, w, 3]
        """
        if os.path.exists(predict_img):
            im_gpu = Image.open(predict_img).convert("RGB")
        if not isinstance(im_gpu, np.ndarray):
            im_gpu = np.array(im_gpu)
        im_gpu = im_gpu.astype(np.float32) / 255.0
        mask_color = tuple(value / 255.0 for value in mask_color)
        for i, mask in enumerate(input_masks):
            mask = mask[..., None]
            im_gpu = (1 - alpha * mask) * im_gpu + alpha * mask * mask_color  
        im_gpu = (im_gpu * 255).astype(np.uint8)
        output_image = Image.fromarray(im_gpu)
        output_image.save(predict_img)

class Detection(TaskBase):
    print("###1")
    def __init__(self,
                 image_path_list:Union[str, List[str]] = None, 
                 query_list:Union[str, List[str]] = None, 
                 method:str = None, 
                 confidence_threshold:float = None,
                 iou_threshold:float = None,
                 image_resize:Tuple[int, int] = None,
                 fp16_inference:bool = False,
                 max_bboxes:int = None,
                 detect_classes:list[str] = None,
                 high_resolution:bool = False
                 ):
        super().__init__()

        self.image_path_list = convert_string_to_list(image_path_list)
        self.query_list = convert_string_to_list(query_list)   
        self.method = method
        self.kwargs = {}
        args_candidate = {
            "conf" : confidence_threshold, 
            "iou" : iou_threshold, 
            "imgsz" : image_resize, 
            "half" : fp16_inference, 
            "max_det" : max_bboxes, 
            #"classes" : detect_classes, 
            "retina_masks" : high_resolution
            }
        for k,v in args_candidate.items():
            if v is not None:
                self.kwargs[k] = v
    
    def post_process_classes(self, result: Dict):     
        for idx in range(len(result)):
            result[idx] = result[idx].replace(" ", "")
        return result
    
    def run(self):
        if return_empty_result([self.image_path_list,self.query_list]):
            return {"image_path_list": [], "predict_img_list": [], "boxes_list": [], "classes_name_dict": [], "predict_speeds_dict": {'preprocess':0.0,'inference':0.0,'postprocess':0.0}}
        
        if self.method in ["yolo_world","yw"]:
            return self.yolo_world()  
        else:
            return self.dino()

    def yolo_world(self):
        yolo_world_model.set_classes(self.query_list)
        print("self.kwargs",self.kwargs)
        results = yolo_world_model.predict(self.image_path_list, device=self.device, **self.kwargs)
        return get_yolo_result(results)
    
    def dino(self):
        image_resize = self.kwargs.get("imgsz",None)
        box_threshold = self.kwargs.get("conf",0.3)
        text_threshold = 0.2
        max_det = self.kwargs.get("max_det",None)
        preprecess_start_time = time.time()
        phrases_list=[]
        boxes_list=[]
        predict_img_list =[]
        
        dino_start_time = time.time()
        
        for single_image in self.image_path_list:
            single_image_boxes_list=[]
            predict_img = get_output_image_uuid_name()
            image_pil = Image.open(single_image).convert("RGB")
            if image_resize is not None:
                image_pil = image_pil.resize(image_resize, Image.Resampling.LANCZOS)
            for query in self.query_list:
                boxes, logits, phrases = langsam_model.predict_dino(
                    image_pil, 
                    query, 
                    box_threshold = box_threshold, 
                    text_threshold = text_threshold,
                    groundingdino = groundingdino_model,
                    device = self.device
                    )
                phrases = self.post_process_classes(phrases)
                phrases_list.append(phrases)
                boxes = np.column_stack((boxes.numpy(),logits.numpy(),phrases))
                single_image_boxes_list = single_image_boxes_list + boxes.tolist()
                if max_det is not None:
                    boxes = boxes[:max_det]   
                if not os.path.exists(predict_img):
                    predict_img_list.append(predict_img)
                    _Visualization().plot_bboxes(single_image, boxes, predict_img)
                else:
                    _Visualization().plot_bboxes(predict_img, boxes, predict_img)
            boxes_list.append(single_image_boxes_list)

        postprecess_start_time = time.time()
        predict_classes_set=set()
        for phrases in phrases_list:
            predict_classes_set.update(set(phrases))
        class_names = {idx:class_name for idx, class_name in enumerate(sorted(predict_classes_set))}
        postprecess_end_time = time.time()
        predict_speeds = {
            'preprocess': dino_start_time-preprecess_start_time,
            'inference': postprecess_start_time-dino_start_time,
            'postprocess': postprecess_end_time-postprecess_start_time
        }

        detection_results_dict = {
            "image_path_list": self.image_path_list,
            "predict_img_list": predict_img_list,
            "boxes_list": boxes_list,
            "classes_name_dict": class_names,
            "predict_speeds_dict": predict_speeds
        }
        print("###18")
        return detection_results_dict


class Segmentation(TaskBase):
    def __init__(self,
                image_path_list:Union[str, List[str]] = None, 
                query_list:Union[str, List[str]] = None, 
                method:str = "lan_sam", 
                bbox_threshold:int = 0.3, 
                text_threshold:int = 0.25
                ) -> None:
    
        super().__init__()
        
        self.image_path_list = convert_string_to_list(image_path_list)
        self.query_list = convert_string_to_list(query_list)
        self.method = method
        self.box_threshold = bbox_threshold
        self.text_threshold = text_threshold
        self.is_semantic = True

    def run(self):
        if return_empty_result([self.image_path_list,self.query_list]):
            return {"image_path_list": [], "predict_img_list": [], "boxes_list": [], "mask_list": [], "classes_name_dict": [], "predict_speeds_dict": {'preprocess':0.0,'inference':0.0,'postprocess':0.0}}

        if self.method == "lan_sam":
            self.is_semantic = True
            return self.lan_sam()
        else:
            raise NotImplementedError("Segmentation only support 'lan_sam' method!")
            
    def lan_sam(self):
        preprecess_start_time = time.time()
        predict_img_list = []
        boxes_list = []
        mask_list = []
        classes_name_list = []
        color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),
                      (128,128,0),(0,128,255),(128,0,128),(128,0,0),(0,128,0),(0,128,128),
                      (64,64,0),(0,64,64),(64,0,64),(64,0,0),(0,64,0),(0,64,64),(0,0,0),(255,255,255)]

        sam_start_time = time.time()
        
        for single_image in self.image_path_list:
            image_pil = Image.open(single_image).convert("RGB")
            predict_img = get_output_image_uuid_name()
            for idx, query in enumerate(self.query_list):
                masks, boxes, phrases, logits = langsam_model.predict(
                    image_pil, 
                    query, 
                    box_threshold = self.box_threshold, 
                    text_threshold = self.text_threshold,
                    groundingdino = groundingdino_model,
                    device = self.device
                    )       
                boxes = np.column_stack((boxes.numpy(),logits.numpy(),phrases))
                masks = masks.numpy()
                boxes_list.append(boxes.tolist())
                mask_list.append(masks.tolist())
                classes_name_list.append(phrases)
                
                if idx<=len(color_list)-1 and self.is_semantic:
                    mask_color = color_list[idx]
                    print("Selected mask color: ",mask_color)
                else:
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    mask_color = (r,g,b)
                    print("Random mask color: ",mask_color)
                _Visualization().plot_masks(input_masks=masks, predict_img=predict_img, im_gpu=image_pil, mask_color=mask_color)
            predict_img_list.append(predict_img)

        postprecess_start_time = time.time()
        predict_classes_set=set()
        for single_classes_list in classes_name_list:
            predict_classes_set.update(set(single_classes_list))
        class_names = {idx:class_name for idx, class_name in enumerate(sorted(predict_classes_set))}
        postprecess_end_time = time.time()
        
        predict_speeds = {
            'preprocess': sam_start_time-preprecess_start_time,
            'inference': postprecess_start_time-sam_start_time,
            'postprocess': postprecess_end_time-postprecess_start_time
        }
        
        mask_image_list = self.draw_masks(self.image_path_list, mask_list)

        results_dict = {
            "image_path_list": self.image_path_list,
            "predict_img_list": predict_img_list,
            "boxes_list": boxes_list,
            #"mask_list": mask_list,
            "mask_image_list": mask_image_list,
            "classes_name_dict": class_names,
            "predict_speeds_dict": predict_speeds
        }
        return results_dict
    
    def draw_masks(self, image_path_list, mask_list):
        all_mask = []
        for idx, image_path in enumerate(image_path_list):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image {image_path}")
                continue
            mask_overlay = np.zeros_like(img, dtype=np.uint8)
            for mask in mask_list[idx]:
                color = tuple(np.random.randint(0, 256, size=3).tolist())
                mask_resized = cv2.resize(np.array(mask, dtype=np.uint8) * 255, (img.shape[1], img.shape[0]))
                mask_overlay[mask_resized > 0] = color
            blended = mask_overlay
            mask_path = get_output_image_uuid_name()
            cv2.imwrite(mask_path, blended)
            print(f"Saved blended image to {mask_path}")
            all_mask.append(mask_path)
        return all_mask

class Classification(TaskBase):
    def __init__(self,
                image_path_list:Union[str, List[str]] = None, 
                query_list:Union[str, List[str]] = None,
                method:str = "dino",
                confidence_threshold:float = 0.0,
                iou_threshold:float = 0.0,
                classes:List = None,
                ) -> None:
    
        super().__init__()

        self.image_path_list = convert_string_to_list(image_path_list)
        self.query_list = convert_string_to_list(query_list)
        self.method = method
        
        self.kwargs = {}

        if self.method == "yolo_world":
            args_candidate = {
                "confidence_threshold" : confidence_threshold, 
                "iou_threshold" : iou_threshold, 
                "detect_classes" : classes 
                }
        elif self.method == "dino":
            args_candidate = {
                "confidence_threshold" : confidence_threshold, 
                }
        else:
            args_candidate = {}
            
        for k,v in args_candidate.items():
            if v is not None and v != 0.0:
                self.kwargs[k] = v

    def run(self):
        if return_empty_result([self.image_path_list,self.query_list]):
            return {"image_path_list": [], "classes_name_dict": [], "predict_speeds_dict": {'preprocess':0.0,'inference':0.0,'postprocess':0.0}}
        
        classification_result_dict = {}
        if self.method is None or self.method in ["dino","yolo_world"]:
            classification_result_dict = self.zero_shot_cls()
        else:
            self.model_path = get_models(keys=["models",self.method])
            classification_result_dict = self.yolo_cls()
        return classification_result_dict
            
    def zero_shot_cls(self):
        results_dict = Detection(
            image_path_list = self.image_path_list, 
            query_list = self.query_list,
            method = self.method,
            **self.kwargs
            ).run()
        return self.yolo_dect_result_to_classfity_result(results_dict)
    
    def yolo_cls(self):
        model = YOLO(self.model_path, device=self.device,**self.kwargs)
        results = model(self.image_path_list)
        return self.yolo_dect_result_to_classfity_result(get_yolo_result(results))

    def yolo_dect_result_to_classfity_result(self, results_dict):  
        from collections import OrderedDict
        import operator
        
        src_img_list = results_dict["image_path_list"]
        classes_of_paths = {label_name: [] for label_name in results_dict['classes_name_dict'].values()}
        
        for image_idx, image_bboxes in enumerate(results_dict["boxes_list"]):
            if not image_bboxes:
                continue
            label_counts = {label_name: 0 for label_name in results_dict['classes_name_dict'].values()}
            label_areas = {label_name: 0 for label_name in results_dict['classes_name_dict'].values()}
            for one_bbox in image_bboxes:
                # [左上角横坐标, 左上角纵坐标, 右下角横坐标, 右下角纵坐标, 信心值, 类别]
                x1, y1, x2, y2 = map(float, one_bbox[:4])
                label = one_bbox[-1]
                label_counts[label] += 1
                label_areas[label] += (x2-x1) * (y2-y1)
            
            label_counts = OrderedDict(label_counts)
            label_counts = sorted(label_counts.items(),key=operator.itemgetter(1),reverse=True)

            if len(label_counts)>1 and label_counts[0][1] == label_counts[1][1]:
                # has at least two labels with same number of detection occurances
                candidate_labels = [label for (label, count) in label_counts if count==label_counts[0][1]]
                # compare area
                label_areas = OrderedDict(label_areas)
                label_areas = sorted(label_areas.items(),key=operator.itemgetter(1),reverse=True)
                
                for label, _ in label_areas:
                    if label in candidate_labels:
                        print(label)
                        classes_of_paths[label].append(src_img_list[image_idx])
                        break
                
            else:
                classes_of_paths[label_counts[0][0]].append(src_img_list[image_idx])        # label_counts[0][0] is class label
        
        
        classification_results_dict = {
            "image_path_list": results_dict["image_path_list"],
            "classes_name_dict": classes_of_paths,
            "predict_speeds_dict": results_dict["predict_speeds_dict"],
            "boxes_list": results_dict["boxes_list"]
        }
        return classification_results_dict

class Counting(TaskBase):
    def __init__(self,
                image_path_list:Union[str, List[str]] = None, 
                query_list:Union[str, List[str]] = None
                ) -> None:
    
        super().__init__()
        
        self.image_path_list = convert_string_to_list(image_path_list)
        self.query_list = convert_string_to_list(query_list)
    
    def run(self):
        if return_empty_result([self.image_path_list,self.query_list]):
            return {"image_path_list": [], "predict_img_list": [], "counting": 0, "counting_sum": 0, "counting_avg": 0.0}
        return self.dino_counting()

    def dino_counting(self):        
        results_dict = Detection(
            image_path_list = self.image_path_list, 
            query_list = self.query_list,
            method = "dino").run()
        return self.yolo_dect_result_to_counting_result(results_dict)
    
    def yolo_dect_result_to_counting_result(self, results_dict):
        counting = []           # List of each image's result
        counting_sum = {}       # Global item counts
        if results_dict['boxes_list'] is not None:
            for box_list in results_dict['boxes_list']:     # Loop all images
                image_count = {}
                for box in box_list:                        # In one image
                    category = box[-1]
                    if category in image_count:
                        image_count[category] += 1
                    else:
                        image_count[category] = 1
                    
                    if category in counting_sum:
                        counting_sum[category] += 1
                    else:
                        counting_sum[category] = 1
                counting.append(image_count)
            counting_avg = {category: round(count / len(results_dict['boxes_list']), 3) for category, count in counting_sum.items()}
        else:
            counting = [] 
            counting_sum = {}
            counting_avg = {}
        
        counting_results_dict = {
            'image_path_list': results_dict['image_path_list'],
            "predict_img_list": results_dict['predict_img_list'],
            'counting': counting,
            'counting_sum': counting_sum,
            'counting_avg': counting_avg
        }
        return counting_results_dict

class Tracking(TaskBase):
    def __init__(self,
            video_path:str = None, 
            query_list:Union[str, list[str]] = None
            ) -> None:

        super().__init__()
        
        self.video_path = video_path
        self.query_list = convert_string_to_list(query_list)
   
    def run(self):
        if return_empty_result([self.video_path,self.query_list]):
            return {"image_path_list": [], "predict_img_list": [], "predict_video": "", "boxes_list": [], "classes_name_dict": [], "predict_speeds_dict": {'preprocess':0.0,'inference':0.0,'postprocess':0.0}}
        return self.yolo_world_tracking()
    
    def yolo_world_tracking(self):
        tracking_model.set_classes(self.query_list)
        save_dir = os.path.join(BASE_LOCAL_OUTPUT_PREFIX, 'video', str(uuid.uuid4()))
        results = tracking_model.track(
            source=self.video_path, 
            save=True
        )
        #docker_pos = "base/" if os.environ.get('CV_AGENT_SERVER_PORT') in ["52002","52003"] else ""
        current_directory = os.getcwd()
        copy_dir(os.path.join(current_directory, "runs/detect/track"), save_dir)
        remove_dir(os.path.join(current_directory, "runs"))
        
        result_dict = get_yolo_result(results, save_dir)
        save_video_path = os.path.join(save_dir, self.video_path.split("/")[-1].split('.')[0] + ".avi")
        result_dict["predict_video"] = save_video_path
        return result_dict

class Pose(TaskBase):
    def __init__(self,
            video_path:str = None,  
            ) -> None:

        super().__init__()
        
        self.video_path = video_path
        
    def run(self):
        if return_empty_result([self.video_path]):
            return {"image_path_list": [], "predict_img_list": [], "predict_video": "", "boxes_list": [], "classes_name_dict": [], "predict_speeds_dict": {'preprocess':0.0,'inference':0.0,'postprocess':0.0}}
        return self.yolo_world_pose()
    
    def yolo_world_pose(self):
        save_dir = os.path.join(BASE_LOCAL_OUTPUT_PREFIX, 'video', str(uuid.uuid4()))
        results = pose_model.track(source = self.video_path, save=True)
        current_directory = os.getcwd()
        copy_dir(os.path.join(current_directory, "runs/pose/track"), save_dir)
        remove_dir(os.path.join(current_directory, "runs"))
        
        result_dict = get_yolo_result(results, save_dir)
        if isinstance(self.video_path,str):
            save_video_path = os.path.join(save_dir, self.video_path.split("/")[-1].split('.')[0] + ".avi")
        else:
            save_video_path = os.path.join(save_dir, str(uuid.uuid4()), ".avi")
        result_dict["predict_video"] = save_video_path
        
        return result_dict
        
class OpticalFlow:
    def __init__(self,
            video_path:str = None,  
            ) -> None:
        
        super().__init__()
        self.video_path = video_path

    def run(self):
        if return_empty_result([self.video_path]):
            return {"image_path_list": [], "predict_img_list": [], "predict_video": "", "boxes_list": [], "classes_name_dict": [], "predict_speeds_dict": {'preprocess':0.0,'inference':0.0,'postprocess':0.0}}
        save_dir = os.path.join(BASE_LOCAL_OUTPUT_PREFIX, str(uuid.uuid4()))
        video_path = os.path.join(save_dir,"optical_flow_result.mp4")
        frame_count = self.extract_frames(video_path = self.video_path, output_folder = save_dir)
        self.process_images_and_create_video(input_folder = save_dir, 
                                             output_folder = save_dir, 
                                             frame_count = frame_count, 
                                             video_path = video_path)
        return {"predict_video": video_path}

    def extract_frames(self, video_path, output_folder, frame_prefix='frame', file_format='jpg'):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Error: Unable to open video file.")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = f"{frame_prefix}_{frame_count:04d}.{file_format}"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame {frame_count} as {frame_filename}")
            frame_count += 1
        cap.release()
        return frame_count - 1 

    def compute_optical_flow(self, prev_img_path, next_img_path, output_path=None):
        prev_img = cv2.imread(prev_img_path, cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(next_img_path, cv2.IMREAD_GRAYSCALE)
        if prev_img is None or next_img is None:
            raise ValueError("Failed to read one or both images.")
        flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR))
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if output_path:
            cv2.imwrite(output_path, flow_bgr)

        return flow_bgr

    def process_images_and_create_video(self, input_folder, output_folder, frame_count, video_path):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        images = []
        for i in range(frame_count):
            image1_path = os.path.join(input_folder, f"frame_{i:04d}.jpg")
            image2_path = os.path.join(input_folder, f"frame_{i+1:04d}.jpg")
            dealt_image = self.compute_optical_flow(image1_path, image2_path)
            dealt_image_path = os.path.join(output_folder, f"optical_flow_image_{i+1:04d}.jpg")
            cv2.imwrite(dealt_image_path, dealt_image)
            images.append(dealt_image_path)

        if len(images) > 0:
            frame = cv2.imread(images[0])
            height, width, layers = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            
            for image_path in images:
                img = cv2.imread(image_path)
                video.write(img)
            
            video.release()
            print(f"Video saved to {video_path}")

class Ocr(TaskBase):
    def __init__(self,
            image_path_list:Union[str, list[str]] = None,  
            bbox_region:list[str] = None,
            ocr_threshold:float = 0.1
            ) -> None:

        self.image_path_list = convert_string_to_list(image_path_list)
        self.bbox_region = bbox_region
        self.ocr_threshold = ocr_threshold

    def get_list_layernum(self, list_in):
        if not list_in:
            return 1
        if not isinstance(list_in, list):
            return 0
        return 1 + self.get_list_layernum(list_in[0])
    
    def run(self):
        if return_empty_result([self.image_path_list]):
            return {"ocr_result": []}
        return self.paddle_ocr(self.image_path_list, self.bbox_region, self.ocr_threshold)

    def paddle_ocr(self, image_path_list:str, boxes:List = None, ocr_threshold=0.1):
        share_bbox = False 
        if boxes is not None and boxes not in [[],[[]],[[[]]]]:
            bboxes_layer_num = self.get_list_layernum(boxes)
            assert bboxes_layer_num==3 or bboxes_layer_num==2, "The 'bbox_region' fed to OCR function should be an list of 2 or 3 dims."
            share_bbox = False if bboxes_layer_num==3 else True
            # 不share需要满足
            if share_bbox == False and (len(image_path_list) != len(boxes)):
                # number of images should match the number of bbox sets
                share_bbox = True
                #boxes = boxes[0]        # all images use the 1st set of bboxes
                if isinstance(boxes[0], list):
                    boxes = boxes[0]
                else:
                    boxes = []  # fallback to empty list to avoid cras
            print(f"bboxes_layer_num = {bboxes_layer_num}, share_bbox = {share_bbox}")
        else:
            boxes = None
            
        format_result = {"ocr_result":[]}
        for image_idx, image_path in enumerate(image_path_list):
            image_pil = Image.open(image_path).convert("RGB")
            image_np = np.array(image_pil)
            this_image_result = []
            if boxes is not None:
                if not share_bbox:
                    # TODO always trigger out of range exception
                    if image_idx < len(boxes):
                        this_img_boxes = boxes[image_idx]
                    else:
                        this_img_boxes = []  
                else: # All images share the same bbox
                    this_img_boxes = boxes

                for one_box in this_img_boxes:
                    print(f"image_idx = {image_idx}, one_box = {one_box}")
                    if not one_box:
                        continue
                    if len(one_box) == 6:
                        box_conf = float(one_box[4])
                        if box_conf < ocr_threshold:
                            continue
                    if isinstance(one_box[0], str):             # If item is string of floats
                        one_box = list(map(float, one_box[:4]))       # only take axis positions
                    x1,y1,x2,y2 = one_box[:4]
                    crop_img = image_np[int(y1):int(y2),int(x1):int(x2)]
                    result = ocr_tool.ocr(crop_img, cls=True)
                    if result and result != [None]:
                        result = "\n".join([item[1][0] for group in result for item in group])
                        this_image_result.append(result)
            else:
                result = ocr_tool.ocr(image_np, cls=True)
                if result and result != [None]:
                    result = "\n".join([item[1][0] for group in result for item in group])
                    this_image_result.append(result)
            format_result["ocr_result"].append(this_image_result)
        return format_result

class Vlm(TaskBase):
    def __init__(self,
            image_path:str = None, 
            query = Union[str, List]
        ) -> None:
        super().__init__()

        self.image_path = image_path
        
        if isinstance(self.image_path,List):
            self.image_path = self.image_path[0]

        if isinstance(query,str):
            self.query = query
        elif isinstance(query,List):
            self.query = "".join(list(map(str, query)))
        else:
            self.query = str(query)

    def run(self):
        if return_empty_result([self.image_path,self.query]):
            return {"vlm_result": ""}
        return self.qwen_vl()

    def qwen_vl(self):
        with open(self.image_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        image_url = f'data:image/jpeg;base64,{img_base64}'
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': image_url}},
                {'type': 'text', 'text': self.query},
            ]
        }]
        request = {
            "model": BASE_VLM_MODEL_NAME, 
            "messages":messages,
            "temperature": 0,
            "stream": False
        }
        response = get_response(url = BASE_VLM_URL, data = request)
        return {"vlm_result":response['choices'][0]['message']['content']}

class Llm(TaskBase):
    def __init__(self,
            query = Union[str, List]
        )-> None:
        super().__init__()
    
        if isinstance(query,str):
            self.query = query
        elif isinstance(query,List):
            self.query = "".join(list(map(str, query)))
        else:
            self.query = str(query)

    def run(self):
        if return_empty_result([self.query]):
            return {"llm_result": ""}
        return self.qwen_chat()

    def qwen_chat(self):
        messages = [{"role": "user", "content": self.query}]
        request = {
            "model": BASE_LLM_MODEL_NAME, 
            "messages":messages,
            "temperature": 0,
            "stream": False
        }
        response = get_response(url = BASE_LLM_URL, data = request)
        return {"llm_result": response['choices'][0]['message']['content']}


class Alarm(TaskBase):
    def __init__(self,
            result_obj,
            detect_region = None,
            detect_class = None,            # list
            detect_method = "appear",       # appear | disappear
            notice_email_or_tel=None,       # 【CHANGE: 此处直接输入email或tel即可，无需tag】show | email | sms
            alarm_threshold:int = None
            ) -> None:

        self.alarm_image_list = []
        self.is_alarm = 0
        self.result_obj = result_obj
        self.detect_region = detect_region
        self.detect_class = detect_class
        self.detect_method = detect_method
        self.notice_email_or_tel = notice_email_or_tel
        self.alarm_threshold = alarm_threshold if alarm_threshold else 0

    def alarm_bbox(self):
        if not self.result_obj['image_path_list'] or not self.result_obj['boxes_list']:
            return {"is_alarm":0, "alarm_str":"未触发警报，检测对象空缺。", "detect_method": self.detect_method, "alarm_method":"output"}

        if len(self.result_obj['image_path_list']) != len(self.result_obj['boxes_list']):
            return {"is_alarm":0, "alarm_str":"未触发警报，image_path_list和boxes_list维度不对齐。", "detect_method": self.detect_method, "alarm_method":"output"}
        
        item_count = 0
        
        for idx, single_image_bboxes in enumerate(self.result_obj["boxes_list"]):
            temp_class_list = []
            is_alarm = 0
            for bbox in single_image_bboxes:
                if not bbox:
                    continue
                if self.detect_method == "appear" and bbox[-1] in self.detect_class:
                    if self.detect_region is None:      # alarm for existence in the whole image 
                        is_alarm = 1
                        item_count += 1
                    else:
                        iou = calculate_iou(bbox[:4], self.detect_region)
                        if iou > 0:
                            is_alarm = 1
                            item_count += 1
                            
                elif self.detect_method == "disappear":
                    if self.detect_region is None:
                        if bbox[-1] in self.detect_class:
                            temp_class_list.append(bbox[-1])
                    else:
                        iou = calculate_iou(bbox[:4],self.detect_region)
                        if iou <= 0:
                            temp_class_list.append(bbox[-1])
            
            if self.detect_method == "disappear":
                if len(set(temp_class_list)) > len(set(self.detect_class)):
                    is_alarm = 1
                else:
                    is_alarm = 0    
                        
            if is_alarm == 1:
                self.alarm_image_list.append(self.result_obj["image_path_list"][idx])        
        
        # if len(self.alarm_image_list) > self.alarm_threshold: 
        if item_count > self.alarm_threshold: 
            
            if self.is_email(self.notice_email_or_tel):
                is_alarm_success = self.email()
                return {"is_alarm":1, "alarm_str":"触发警报", "detect_method": self.detect_method, "alarm_method":"email", "is_alarm_success":is_alarm_success, "alarm_image_list":self.alarm_image_list}
            elif self.is_phone_number(self.notice_email_or_tel):                
                is_alarm_success = self.sms()   
                return {"is_alarm":1, "alarm_str":"触发警报", "detect_method": self.detect_method, "alarm_method":"phone", "is_alarm_success":is_alarm_success, "alarm_image_list":self.alarm_image_list} 
            else:
                return {"is_alarm":1, "alarm_str":"触发警报", "detect_method": self.detect_method, "alarm_method":"output", "alarm_image_list":self.alarm_image_list}
        else:
            return {"is_alarm":0, "alarm_str":"未触发警报", "detect_method": self.detect_method, "alarm_method":"output"}
    
    def run(self):
        if isinstance(self.result_obj, dict) and "image_path_list" in self.result_obj and "boxes_list" in self.result_obj:
            return self.alarm_bbox()
        elif isinstance(self.result_obj, dict) and "is_alarm" in self.result_obj:
            if self.result_obj["is_alarm"]:
                return {"is_alarm":1, "alarm_str":f"前序处理有报警信号直接传入，触发警报！\n{str(self.result_obj)}", "alarm_method":"output"}
            else:
                return {"is_alarm":0, "alarm_str":f"前序处理提示不必报警，未触发警报！\n{str(self.result_obj)}", "alarm_method":"output"}
        else:
            #raise ValueError(f"Current alarm only supports bounding box detection, input argument 'result_obj' must have keys: image_path_list, boxes_list.")
            return {"is_alarm":0, "alarm_str":f"不是标准的警报输入，未触发警报！\n{str(self.result_obj)}", "detect_method": self.detect_method, "alarm_method":"output"}
        
    def is_email(self, string) -> bool:
        if string is None:
            return False
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, string) is not None

    def is_phone_number(self, string) -> bool:
        if string is None:
            return False
        phone_pattern = r'^\+?[1-9]\d{1,14}$'
        return re.match(phone_pattern, string) is not None

    def email(self) -> bool:
        pass
    def sms(self) -> bool:
        pass

class Output(TaskBase):
    def __init__(self,
            result_obj, 
            method="show", # show | write | zip
            ):
        self.result_obj = result_obj
        self.method = method

    def run(self):
        if  isinstance(self.result_obj, list):
            self.result_obj = str(self.result_obj)
    
        if isinstance(self.result_obj, str):
            if self.method == "show":
                return {"output": "以下是agent运行输出的结果: \n" + self.result_obj}
            elif self.method == "write":
                txt_path = write_random_name_txt(result_obj = self.result_obj)
                return {"output_path": txt_path}
            elif self.method == "zip":
                txt_path = write_random_name_txt(result_obj = self.result_obj)
                zip_path = zip_files(txt_path)
                return {"output_path": zip_path}
            else:
                raise ValueError("Wrong method!")
            
        elif isinstance(self.result_obj, Dict):
            if self.method == "show":
                output = "以下是agent运行输出的结果: \n"
                for k,v in self.result_obj.items():
                    if k == "image_path_list":
                        output += "原始图片服务器地址:" + str(v)
                    elif k == "predict_img_list":
                        output += "预测图片服务器地址:" + str(v)
                    elif k == "boxes_list":
                        output += "预测边界框和类别列表:" + str(v)
                    elif k == "classes_name_list":
                        output += "预测类别列表:" + str(v)
                    elif k == "predict_speeds_list":
                        output += "预测速度:" + str(v)
                return {"output": output}
            elif self.method == "write":
                txt_path = write_random_name_txt(result_obj = str(self.result_obj))
                return {"output_path": txt_path}
            elif self.method == "zip":
                txt_path = write_random_name_txt(result_obj = str(self.result_obj))
                zip_path = ""
                for k,v in self.result_obj.items():
                    if k == "predict_img_list":
                        zip_path = zip_files(v.append(txt_path))
                return {"output_path": zip_path}
            else:
                raise ValueError("Wrong method!")       
        else:
            return {"output": "以下是agent运行输出的结果: \n"+str(self.result_obj)}

class VideoPrecess(TaskBase):
    def __init__(self,
            video_path:str = None,
            interval:int = 1,
            method:str = "get_frames",

            rtmp_url:str = None,
            rtmp_duration:int = 2
            ) -> None:
        
        super().__init__()
    
        self.video_path = video_path
        self.interval= interval
        self.method = method

        self.rtmp_url = rtmp_url
        self.rtmp_duration = rtmp_duration

    def run(self):

        if self.method == "get_frames" and self.video_path is not None:
            return {"processed_path_list":self.get_frames()}
        elif self.method == "rtmp_capturer" and self.rtmp_url is not None:
            return {"processed_path_list":self.capture_rtmp_stream()}
        else:
            raise ValueError("The 'method' should be in 'get_frames' and 'rtmp_capturer', and either video_path or rtmp_url should not be None.")

    def get_frames(self, frame_prefix='video_frame', file_format='jpg'):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("Error: Unable to open video file.")
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.interval)

        frame_count = 0
        save_dir = os.path.join(BASE_LOCAL_OUTPUT_PREFIX, frame_prefix, str(uuid.uuid4()))
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir,exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_filename = f"{frame_count:08d}.{file_format}"
                frame_path = os.path.join(save_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
            frame_count += 1
        cap.release()
        return frames

    def capture_rtmp_stream(self, frame_prefix="rtmp_frame", file_format='jpg'):
        save_dir = os.path.join(BASE_LOCAL_OUTPUT_PREFIX, frame_prefix, str(uuid.uuid4())) 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir,exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(self.rtmp_url)
            start_time = time.time()
            while not cap.isOpened():
                if time.time() - start_time > 10:  # timeout after 10 seconds
                    raise RuntimeError("Can not open RTMP stream: Timeout.")
        except Exception as e:
            print("Can not open RTMP steam or read frame." + str(e))
            return [] 

        image_paths = []
        start_time = time.time()
        while time.time() - start_time < self.rtmp_duration:
            ret, frame = cap.read()
            if not ret:
                print("Can not read frame.")
                break
            
            current_time = time.time()
            if int(current_time - start_time) % self.interval == 0:
                frame_filename = os.path.join(save_dir, f'frame_{int(current_time - start_time):08d}.{file_format}')
                cv2.imwrite(frame_filename, frame)
                image_paths.append(frame_filename)
                time.sleep(self.interval)

        cap.release()
        return image_paths