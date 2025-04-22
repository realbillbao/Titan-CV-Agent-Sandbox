import os
import sys
import json
import traceback
from celery.result import AsyncResult

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from atools import setup_logger
from base.base_config import *

from celery_worker import app as celery_app
from flask import Flask, request
from base.base_config import *

os.makedirs(os.path.dirname(BASE_LOGGER_APP_PATH), exist_ok=True)
app = Flask(__name__)
logger = setup_logger(BASE_LOGGER_APP_PATH)

os.environ["HF_HOME"] = BASE_HF_HOME
os.environ["TORCH_HOME"] = BASE_TORCH_HOME
os.environ["HF_ENDPOINT"] = BASE_HF_ENDPOINT

# model package
from ultralytics import YOLO
from paddleocr import PaddleOCR
from lang_sam import LangSAM, load_model_hf

# Model Register
yolo_world_model = YOLO(YOLOV8X_WORLDV2)
langsam_model = LangSAM()
groundingdino_model = load_model_hf()
tracking_model = YOLO(YOLOV8X_WORLDV2)  # or choose yolov8m/l-world.pt
pose_model = YOLO(YOLOV8X_POSE)  # or choose yolov8m/l-world.pt
ocr_tool = PaddleOCR(use_angle_cls=True, lang="ch")


initial_states = {
    "yolo_world_model": yolo_world_model.model.state_dict().copy(),
    "tracking_model": tracking_model.model.state_dict().copy(),
    "pose_model": pose_model.model.state_dict().copy(),
}

app.config["yolo_world_model"] = yolo_world_model
app.config["langsam_model"] = langsam_model
app.config["groundingdino_model"] = groundingdino_model
app.config["tracking_model"] = tracking_model
app.config["pose_model"] = pose_model
app.config["ocr_tool"] = ocr_tool

def get_models():
    yolo_world_model.model.load_state_dict(initial_states["yolo_world_model"])
    tracking_model.model.load_state_dict(initial_states["tracking_model"])
    pose_model.model.load_state_dict(initial_states["pose_model"])
    return (
            app.config["yolo_world_model"], 
            app.config["langsam_model"], 
            app.config["groundingdino_model"],
            app.config["tracking_model"],
            app.config["pose_model"],
            app.config["ocr_tool"], 
            )

@app.route('/call_function', methods=['POST'])
def call_function():
    try:
        data = json.loads(request.data)
        logger.info(f"Received call_function request: {data}")

        function_name = data["function_name"]
        func_args = data["arguments"]
        exec_mode = data.get("exec_mode", "sync") # "async"

        kwargs={
                "function_name": function_name,
                "func_args": func_args
            }

        if function_name in CPU_TASK:
            task = celery_app.send_task(
                'cv_tasks.execute_function',
                kwargs=kwargs,
                queue='cpu_queue',
                routing_key='cpu.#'
            )
        elif function_name in GPU_HEAVY_TASK:
            task = celery_app.send_task(
                'cv_tasks.execute_function',
                kwargs=kwargs,
                queue='gpu_heavy_queue',
                routing_key='gpu.heavy.#'
            )
        else:
            task = celery_app.send_task(
                'cv_tasks.execute_function',
                kwargs=kwargs,
                queue='gpu_light_queue',
                routing_key='gpu.light.#'
            )

        if exec_mode == "sync":
            try:
                result = task.get(timeout=900)
                return {
                    "code": 1,
                    "message": "[SUCCESS] Synchronous Task completed successfully!",
                    "data": result
                }, 200
            except Exception as e:
                tb_str = traceback.format_exception(type(e), e, e.__traceback__)
                detailed_error_message = ''.join(tb_str)
                logger.error(f"[FAILED] Synchronous error while waiting for task result: {detailed_error_message}")
                return {
                    "code": -1,
                    "message": f"[FAILED] Synchronous task execution failed: {str(e)}",
                    "data": detailed_error_message
                }, 500
        else:
            return {
                "code": 2,
                "message": "[SUBMITTED] Asynchronous task submitted successfully",
                "data": task.id
            }, 202
    except Exception as e:
        tb_str = traceback.format_exception(type(e), e, e.__traceback__)
        detailed_error_message = ''.join(tb_str)
        logger.error(detailed_error_message)
        return {
            "code": -1,
            "message": f"[FAILED] System-level task execution failed: {str(e)}",
            "data": detailed_error_message
        }

@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    try:
        task_result = AsyncResult(task_id, app=celery_app)

        if task_result.state == 'PENDING':
            response = {
                "code": 2,
                "message": "[PENDING] Asynchronous task is pending execution.",
                "data": task_result.state
                
            }
        elif task_result.state == 'SUCCESS':
            response = {
                "code": 1,
                "message": "[SUCCESS] Asynchronous Task completed successfully!",
                "data": task_result.result
            }
        elif task_result.state == 'FAILURE':
            response = {
                "code": -1,
                "message": f"[FAILED] Asynchronous task execution failed: {str(e)}",
                "data": str(task_result.result),  
            }
        else:
            response = {
                "code": 2,
                "message": "[In Progress] Asynchronous Task is in progress.",
                "data": task_result.state,  
            }
        return response
    
    except Exception as e:
        tb_str = traceback.format_exception(type(e), e, e.__traceback__)
        detailed_error_message = ''.join(tb_str)
        logger.error(detailed_error_message)
        return {
            "code": -1,
            "message": f"[FAILED] System-level asynchronous task execution failed: {str(e)}",
            "data": detailed_error_message
        }

if __name__ == '__main__':
    app.run(host=BASE_FLASK_ADDR, debug=False, port=BASE_FLASK_PORT)