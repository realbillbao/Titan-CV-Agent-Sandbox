import os 
AGENT_ROOT = os.getenv("AGENT_ROOT","/datacanvas/titan_cv_agent_sandbox")
MODELS_ROOT = os.getenv("MODELS_ROOT","/datacanvas/titan_cv_agent_sandbox/models")
ACCESS_KEY = os.getenv("ACCESS_KEY", "")
BASE_LLM_MODEL_NAME = os.getenv("BASE_LLM_MODEL_NAME","gpt-4o")
BASE_LLM_URL = os.getenv("BASE_LLM_URL", "https://api.openai.com/v1/chat/completions")
BASE_VLM_MODEL_NAME = os.getenv("BASE_VLM_MODEL_NAME", "gpt-4o")
BASE_VLM_URL = os.getenv("BASE_VLM_URL", "https://api.openai.com/v1/chat/completions")

BASE_FLASK_ADDR = os.getenv("BASE_FLASK_ADDR","0.0.0.0")
BASE_FLASK_PORT = os.getenv("BASE_FLASK_PORT", 52001)
BASE_LOGGER_APP_PATH = os.getenv("BASE_LOGGER_APP_PATH", os.path.join(AGENT_ROOT, "base/logs/app.log"))
BASE_LOGGER_WORKER_PATH = os.getenv("BASE_LOGGER_WORKER_PATH", os.path.join(AGENT_ROOT, "base/logs/worker.log"))
BASE_LOCAL_OUTPUT_PREFIX = os.getenv("BASE_LOCAL_OUTPUT_PREFIX", os.path.join(AGENT_ROOT, "output/"))

CPU_TASK = ["preprocess","postprocess","alarm","output","videoprecess"]
GPU_HEAVY_TASK = ["llm", "vlm"]
REDIS_BROKER='redis://localhost:6379/0'
REDIS_BACKEND='redis://localhost:6379/1'
YOLOV8N = os.path.join(MODELS_ROOT, "base/yolov8n.pt")
YOLOV8X_POSE = os.path.join(MODELS_ROOT, "base/yolov8x-pose.pt")
YOLOV8X_WORLDV2 = os.path.join(MODELS_ROOT, "base/yolov8x-worldv2.pt")
BASE_HF_HOME = os.path.join(MODELS_ROOT,"cache/hf")
BASE_TORCH_HOME = os.path.join(MODELS_ROOT,"cache/torch")
BASE_HF_ENDPOINT = "https://hf-mirror.com"
