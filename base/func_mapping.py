import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import base.task as task

FUNC_NAME_MAPPING = {
    "preprocess": {"class":task.Preprocess, "run_function_name": "run"},
    "postprocess": {"class":task.Postprocess, "run_function_name": "run"},
    "detection": {"class":task.Detection, "run_function_name": "run"},
    "segmentation": {"class":task.Segmentation, "run_function_name": "run"},
    "classification": {"class":task.Classification, "run_function_name": "run"},
    "counting": {"class":task.Counting, "run_function_name": "run"},
    "tracking": {"class":task.Tracking, "run_function_name": "run"},
    "pose": {"class":task.Pose, "run_function_name": "run"},
    "optical_flow": {"class":task.OpticalFlow, "run_function_name": "run"},
    "ocr": {"class":task.Ocr, "run_function_name": "run"},
    "vlm": {"class":task.Vlm, "run_function_name": "run"},
    "llm": {"class":task.Llm, "run_function_name": "run"},
    "alarm": {"class":task.Alarm, "run_function_name": "run"},
    "output": {"class":task.Output, "run_function_name": "run"},
    #"auto_yolo_train": {"class":task.AutoyoloTrainer, "run_function_name": "run"},
    "videoprecess": {"class":task.VideoPrecess, "run_function_name": "run"}
}