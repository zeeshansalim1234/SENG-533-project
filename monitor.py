import psutil
import time
import csv
import subprocess
from datetime import datetime

subprocess.Popen(["python", "app.py"])

def get_process(name):
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == 'python.exe' or proc.info['name'] == 'pythonw.exe':
            try:
                for line in proc.cmdline():
                    if name in line:
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return None


flask_process = get_process("app.py")
if flask_process is None:
    raise Exception("Flask application is not running")

with open('memory_and_cpu.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "CPU Usage (%)", "Memory Usage (MB)"])
    #write the data to the csv file every 1 seconds
    try:
        while True:
            cpu_usage = flask_process.cpu_percent(interval=1)
            memory_info = flask_process.memory_info()
            memory_usage = memory_info.rss / (1024 * 1024) #mb measureent
            formatted_cpu_usage = f"{cpu_usage:.10f}"
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), formatted_cpu_usage, memory_usage])
    except KeyboardInterrupt:
        print("Monitoring stopped.")
