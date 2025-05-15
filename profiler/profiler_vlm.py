import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import json
import time
import power.AGXPowerLogger as APL
from dvfs.lib import setCpu, setGpu, getCpuStatus, getGpuStatus
from task.VLMPipeline import VLMPipeline

# MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
# CONTEXT_PATH = "./context/hp/context-0.15k.txt"
# MAX_TOKEN = 30

CONFIG_NAME = f"gemma-3-4B-prefill"
CPU_CONFIGS = [
    115200, 
    192000, 
    268800, 
    345600, 
    422400, 
    499200, 
    576000, 
    652800, 
    729600,
    806400,
    883200, 
    960000,
    1036800,
    1113600, 
    1190400,
    1267200,
    1344000, 
    1420800,
    1497600,
    1574400, 
    1651200,
    1728000,
    1804800, 
    1881600,
    1958400,
    2035200, 
    2112000,
    2188800,
    2201600,
]
GPU_CONFIGS = [
    306000000, 
    408000000, 
    510000000, 
    612000000, 
    714000000, 
    816000000, 
    918000000, 
    1020000000, 
    1122000000, 
    1224000000,
    1300500000,
]

if __name__ == "__main__":
    result = {}
    for cpu_config in CPU_CONFIGS:
        for gpu_config in GPU_CONFIGS:
            setCpu(cpu_config)
            setGpu(gpu_config)
            time.sleep(5)
            print(CONFIG_NAME, " CPU: ", getCpuStatus(), " GPU: ", getGpuStatus())

            pipeline = VLMPipeline()
            pipeline.load_model()
            pipeline.load_processor()
            msgs = pipeline.get_messages()
            pipeline.load_inputs(msgs)
            pipeline.setup_cache()
            
            # Logging
            logger = APL.AGXPowerLogger()
            logger.start()
            t0 = time.perf_counter()

            pipeline.prefill()

            t1 = time.perf_counter()
            logger.stop()

            latency = t1 - t0
            energy = logger.getTotalEnergy()
            print("Latency: ", latency)
            print("GPU Energy Consumption: ", energy[0])
            print("CPU Energy Consumption: ", energy[1])
            print("Memory Energy Consumption: ", energy[2])
            result[str(cpu_config) + ":" + str(gpu_config)] = (float(latency), float(energy[0]), float(energy[1]), float(energy[2]))
            logger.reset()
    
    with open(CONFIG_NAME + ".json", 'w') as file:
        json.dump(result, file, indent=4)