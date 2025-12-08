"""
Configuración de Gunicorn para producción
"""
import multiprocessing
import os

bind = "0.0.0.0:8000"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
threads = 4

timeout = 120
graceful_timeout = 30
keepalive = 5

max_requests = 1000
max_requests_jitter = 50

preload_app = True

accesslog = "-"
errorlog = "-"
loglevel = "info"

pidfile = "/tmp/gunicorn_backend_rag.pid"

# user = "felipep"
# group = "felipep"

worker_tmp_dir = "/dev/shm"

raw_env = [
    f"GOOGLE_APPLICATION_CREDENTIALS={os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')}",
    f"PYTHONPATH={os.getenv('PYTHONPATH', '')}",
]

def on_starting(server):
    print("Gunicorn iniciando...")
    print(f"   Workers: {workers}")
    print(f"   Threads: {threads}")
    print(f"   Timeout: {timeout}s")

def on_reload(server):
    print("Gunicorn recargando...")

def worker_int(worker):
    print(f"Worker {worker.pid} interrumpido")

def worker_abort(worker):
    print(f"Worker {worker.pid} abortó")

def post_fork(server, worker):
    print(f"Worker {worker.pid} iniciado")
    
    import tensorflow as tf
    tf.config.set_soft_device_placement(True)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
