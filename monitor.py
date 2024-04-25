import psutil
import time
import csv
import subprocess
from datetime import datetime

subprocess.Popen(["python", "app.py"])

def get_process(name):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.info['name'] in ('python.exe', 'pythonw.exe'):
            try:
                if name in ' '.join(proc.info['cmdline']):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return None
# flask startup
time.sleep(5)

flask_process = get_process("app.py")
if flask_process is None:
    raise Exception("Flask application is not running")

with open('metrics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "CPU Usage (%)", "Memory Usage (MB)", "Memory Usage (%)", "Disk Read Bytes", "Disk Write Bytes"])
    try:
        while True:
            cpu_usage = flask_process.cpu_percent(interval=1)
            memory_info = flask_process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024) # Convert to MB
            memory_usage_percent = (memory_info.rss / psutil.virtual_memory().total) * 100
            # Format CPU usage to 4 decimal places
            formatted_cpu_usage = f"{cpu_usage:.4f}"

        
            io_counters = flask_process.io_counters()
            disk_read_bytes = io_counters.read_bytes
            disk_write_bytes = io_counters.write_bytes

            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), formatted_cpu_usage, memory_usage_mb, memory_usage_percent, disk_read_bytes, disk_write_bytes])
            file.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Monitoring stopped.")