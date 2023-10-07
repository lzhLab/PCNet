import os
import time

cmd = "python main.py"


def gpu_info():
    gpu_status = os.popen("nvidia-smi | grep %").read().split("|")
    gpu_memory = int(gpu_status[14].split("/")[0].split("M")[0].strip())
    gpu_percent = int(gpu_status[15].split("%")[0].strip())
    return gpu_memory, gpu_percent


def narrow_setup(interval=600):
    gpu_memory, gpu_percent = gpu_info()
    while gpu_memory > 5000 or gpu_percent > 20:
        gpu_memory, gpu_percent = gpu_info()
        print(
            "\r GPU_Memory_Usage: {}MiB / 32510MiB ; GPU_Util: {}%;".format(
                gpu_memory, gpu_percent
            )
        )
        time.sleep(interval)
    os.system(cmd)


if __name__ == "__main__":
    narrow_setup(60)
