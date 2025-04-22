#!/bin/bash

echo "Stopping all services..."

echo "Stopping Redis..."
redis_pid=$(ps aux | grep '[r]edis-server' | awk '{print $2}')
if [ -n "$redis_pid" ]; then
  kill -9 $redis_pid
  echo "Redis stopped."
else
  echo "Redis is not running."
fi

echo "Stopping Celery workers..."
celery_pids=$(ps aux | grep '[c]elery' | awk '{print $2}')
if [ -n "$celery_pids" ]; then
  kill -9 $celery_pids
  echo "Celery workers stopped."
else
  echo "No Celery workers are running."
fi

echo "Stopping API service..."
api_pid=$(ps aux | grep '[p]ython base/api.py' | awk '{print $2}')
if [ -n "$api_pid" ]; then
  kill -9 $api_pid
  echo "API service stopped."
else
  echo "API service is not running."
fi

echo "All services have been stopped."
