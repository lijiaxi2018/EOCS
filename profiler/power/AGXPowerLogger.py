# Reference: https://github.com/hongpeng-guo/BoFL

import threading
import time

agx_orin_nodes = [
    ('module/gpu', '0040', '0', '1'),
    ('module/cpu', '0040', '0', '2'),
    ('module/ddr', '0041', '1', '2'),
]

def readValue(i2cAddr='0041', index='3', channel='1'):
    """Reads all values (voltage, current, power) from one node"""

    voltage, current = None, None

    fname_voltage = '/sys/bus/i2c/drivers/ina3221/1-%s/hwmon/hwmon%s/in%s_input' % (i2cAddr, index, channel)
    with open(fname_voltage, 'r') as f:
        voltage = f.read()
    
    fname_current = '/sys/bus/i2c/drivers/ina3221/1-%s/hwmon/hwmon%s/curr%s_input' % (i2cAddr, index, channel)
    with open(fname_current, 'r') as f:
        current = f.read()
    
    return [float(voltage), float(current), float(voltage) * float(current)]

def readAllValue(nodes = agx_orin_nodes):
    """Reads all values (voltage, current, power) from all nodes"""
    
    values = [readValue(i2cAddr=node[1], index=node[2], channel=node[3]) for node in nodes]
    return values

def readPowerValue(i2cAddr='0041', index='3', channel='1'):
    """Reads power value from one node"""

    voltage, current = None, None

    fname_voltage = '/sys/bus/i2c/drivers/ina3221/1-%s/hwmon/hwmon%s/in%s_input' % (i2cAddr, index, channel)
    with open(fname_voltage, 'r') as f:
        voltage = f.read()
    
    fname_current = '/sys/bus/i2c/drivers/ina3221/1-%s/hwmon/hwmon%s/curr%s_input' % (i2cAddr, index, channel)
    with open(fname_current, 'r') as f:
        current = f.read()
    
    return float(voltage) * float(current)

def readAllPowerValue(nodes = agx_orin_nodes):
    """Reads power value from all nodes"""
    
    values = [readPowerValue(i2cAddr=node[1], index=node[2], channel=node[3]) for node in nodes]
    return values

class AGXPowerLogger:

    def __init__(self, interval=0.1, nodes=agx_orin_nodes):
        """Constructs the power logger and sets a sampling interval (default: 0.1s)"""

        self.interval = interval
        self.startTime = -1
        self.dataLog = []
        self.nodes = nodes

    def start(self):
        "Starts the logging activity"""

        # define the inner function called regularly by the thread to log the data
        def threadFun():
            # start next timer
            self.start()
            # log data
            t = time.time() - self.startTime
            self.dataLog.append((t, readAllPowerValue(self.nodes)))

        # setup the timer and launch it
        self.tmr = threading.Timer(self.interval, threadFun)
        self.tmr.start()
        
        if self.startTime < 0:
            self.startTime = time.time()
    
    def stop(self):
        """Stops the logging activity"""

        self.tmr.cancel()

    def reset(self):
        """Reset the logger as newly initialized"""

        self.startTime = -1
        self.dataLog = []
    
    def getDataLog(self):
        return self.dataLog

    def getTotalEnergy(self):
        data = self.dataLog
        
        total_gpu_energy = 0.0
        total_cpu_energy = 0.0
        total_memory_energy = 0.0

        # Iterate through the list, except the last item
        for i in range(len(data) - 1):
            # Calculate the time difference between the current and next timestamp
            time_diff = data[i+1][0] - data[i][0]

            # Calculate energy for each component by multiplying the power by the time difference
            gpu_energy =  0.5 * (data[i+1][1][0] + data[i][1][0]) * time_diff
            cpu_energy = 0.5 * (data[i+1][1][1] + data[i][1][0]) * time_diff
            memory_energy = 0.5 * (data[i+1][1][2] + data[i][1][0]) * time_diff

            # Add the calculated energy to the total energy counters
            total_gpu_energy += gpu_energy
            total_cpu_energy += cpu_energy
            total_memory_energy += memory_energy

        return total_gpu_energy, total_cpu_energy, total_memory_energy