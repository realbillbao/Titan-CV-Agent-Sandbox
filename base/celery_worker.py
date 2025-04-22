import os
import sys
import shutil
import importlib.util
from kombu import Queue
from celery import Celery
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from atools import setup_logger
from base_config import *

os.makedirs(os.path.dirname(BASE_LOGGER_WORKER_PATH), exist_ok=True)
logger = setup_logger(BASE_LOGGER_WORKER_PATH)

app = Celery('cv_tasks', broker=REDIS_BROKER, backend=REDIS_BACKEND)
app.conf.update(broker_url=REDIS_BROKER,result_backend=REDIS_BACKEND,task_serializer='json',
    accept_content=['json'],result_serializer='json',timezone='UTC',enable_utc=True)

app.conf.task_queues = (
    Queue('cpu_queue', routing_key='cpu.#'),
    Queue('gpu_light_queue', routing_key='gpu.light.#'),
    Queue('gpu_heavy_queue', routing_key='gpu.heavy.#'),
)

app.conf.task_routes ={
    'cv_tasks.execute_function': {
        'queue': 'cpu_queue',
        'routing_key': 'cpu.#',
    },
    'cv_tasks.execute_function': {
        'queue': 'gpu_light_queue',
        'routing_key': 'gpu.light.#',
    },
    'cv_tasks.execute_function': {
        'queue': 'gpu_heavy_queue',
        'routing_key': 'gpu.heavy.#',
    }
}

def garbage_collection(gc_path:str = None):
    if gc_path is None:
        return False
    try:
        for filename in os.listdir(gc_path):
            file_path = os.path.join(gc_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Can not clearup file {file_path} with error: {str(e)}")        
        logger.info(f"Clearup: {gc_path}")
        return True
    except Exception as e:
        logger.error(f"Can not clearup {gc_path} for file {filename} with error: {str(e)}")

def _load_local_functions(functions, folder):
    for filename in os.listdir(folder):
        if filename.endswith(".py") and filename != "__init__.py":
            print(f"load file {filename}")
            try:
                module_name = filename[:-3]
                module_path = os.path.join(folder, filename)
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for func_name in dir(module):
                    func = getattr(module, func_name)
                    if callable(func):
                        if (not func_name.startswith("_")) and (not func_name in functions):
                            print(f"load python {func_name}")
                            functions[func_name] = {
                                "class": module,
                                "run_function_name": func_name
                            }
            except Exception as e:
                logger.error(f"Can not load {filename} with error: {str(e)}")
                continue
    return functions

@app.task(name='cv_tasks.execute_function')
def execute_function(function_name, func_args):
    from base.func_mapping import FUNC_NAME_MAPPING
    functions = FUNC_NAME_MAPPING.copy()

    function_info = functions[function_name]
    tgt_class = function_info["class"]
    run_func_name = function_info["run_function_name"]

    if '__init__' in dir(tgt_class):
        class_obj = tgt_class(**func_args)
        class_func = getattr(class_obj, run_func_name)
        result = class_func()
    else:
        class_func = getattr(tgt_class, run_func_name)
        result = class_func(**func_args)

    return result
