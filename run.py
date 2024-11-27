import os
import time
import subprocess

tasks_template = "python main.py --config ./exps/engine/{}"
files = os.listdir('./exps/engine/')

tasks = [tasks_template.format(task) for task in files]

print("all tasks:")
for task in tasks:
    print(task)

def get_gpu_users():
    cmd = "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits"
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()
    gpu_user_map = {}
    if output == "":
        return gpu_user_map

    for line in output.split("\n"):
        gpu_uuid, pid = line.split(", ")
        user_cmd = f"ps -o user= -p {pid}"
        user_result = subprocess.run(user_cmd.split(), stdout=subprocess.PIPE)
        user = user_result.stdout.decode('utf-8').strip()

        if gpu_uuid not in gpu_user_map:
            gpu_user_map[gpu_uuid] = set()
        gpu_user_map[gpu_uuid].add(user)

    return gpu_user_map

def get_gpu_info():
    cmd = "nvidia-smi --query-gpu=index,uuid,power.draw,memory.used,memory.total --format=csv,noheader,nounits"
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()

    gpu_info_list = []
    for line in output.split("\n"):
        gpu_index, gpu_uuid, power_draw, memory_used, memory_total = line.split(", ")

        gpu_info_list.append({
            'index': gpu_index,
            'uuid': gpu_uuid,
            'power_draw': float(power_draw),
            'memory_used': float(memory_used),
            'memory_total': float(memory_total)
        })

    return gpu_info_list

def get_available_gpus(exclude_user='your_name'):
    gpu_user_map = get_gpu_users()
    gpu_info_list = get_gpu_info()

    available_gpus = []

    for gpu_info in gpu_info_list:
        gpu_index = gpu_info['index']
        gpu_uuid = gpu_info['uuid']
        power_draw = gpu_info['power_draw']
        memory_used = gpu_info['memory_used']
        memory_total = gpu_info['memory_total']

        # if exclude_user not in gpu_user_map.get(gpu_uuid, set()) and power_draw <= 300 and (memory_used / memory_total) <= 0.8:
        if power_draw <= 300 and (memory_used / memory_total) <= 0.8:
            available_gpus.append(gpu_index)

    return available_gpus

def run_task_on_gpu(gpu_id, task_command):
    timestamp = int(time.time())
    screen_name = f"exp_gpu{gpu_id}_{timestamp}"
    os.system(f"screen -dmS {screen_name} bash -c 'CUDA_VISIBLE_DEVICES={gpu_id} {task_command}'")

def main():
    pending_tasks = tasks
    
    while pending_tasks:
        free_gpus = get_available_gpus()
        
        if free_gpus:
            for gp in free_gpus:
                if not pending_tasks:
                    break
                gpu_id = gp
                task_command = pending_tasks.pop(0)
                print(f"Running task on GPU {gpu_id}: {task_command}")
                run_task_on_gpu(gpu_id, task_command)
        
        time.sleep(30)

if __name__ == "__main__":
    main()
