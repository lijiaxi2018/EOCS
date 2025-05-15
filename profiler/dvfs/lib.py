# Reference: https://github.com/hongpeng-guo/BoFL

def setCpuFreq(cpuFreq, cpuFreq_cur=0):
	"""Set all ARM CPUs frequencies based on the given param"""

	for i in [0, 1, 2, 3, 4, 5, 6, 7]:
		max_fname = "/sys/devices/system/cpu/cpu{:d}/cpufreq/scaling_max_freq".format(i)
		min_fname = "/sys/devices/system/cpu/cpu{:d}/cpufreq/scaling_min_freq".format(i)
		
		first, second = max_fname, min_fname
		if cpuFreq < cpuFreq_cur:
			first, second = min_fname, max_fname

		with open(first, 'w') as f:
			f.write(str(cpuFreq))
		with open(second, 'w') as f:
			f.write(str(cpuFreq))


def setGpuFreq(gpuFreq, gpuFreq_cur=0):
	"""Set the GPU frequency based on the given param"""

	max_fname = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/max_freq"
	min_fname = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/min_freq"

	first, second = max_fname, min_fname
	if gpuFreq < gpuFreq_cur:
		first, second = min_fname, max_fname

	with open(first, 'w') as f:
		f.write(str(gpuFreq))
	with open(second, 'w') as f:
		f.write(str(gpuFreq))


def setEmcFreq(emcFreq, emcFreq_cur=0):
	"""Set the memory frequency based on the given param"""

	lock_fname = "/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked"
	state_fname = "/sys/kernel/debug/bpmp/debug/clk/emc/state"
	rate_fname =  "/sys/kernel/debug/bpmp/debug/clk/emc/rate"
	cap_fname = "/sys/kernel/nvpmodel_clk_cap/emc"

	with open(lock_fname, 'w') as f:
		f.write('1')
	with open(state_fname, 'w') as f:
		f.write('1')

	first, second = cap_fname, rate_fname
	if emcFreq < emcFreq_cur:
		first, second = rate_fname, cap_fname

	with open(first, 'w') as f:
		f.write(str(emcFreq))
	with open(second, 'w') as f:
		f.write(str(emcFreq))

def getcurStatus():
	"""Get current system knob status, including cpu/gpu/memory freqs
	as well as the hotplug status of the Denver cores"""
	
	cpuFreq_fname = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
	gpuFreq_fname = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq"
	emcFreq_fname = "/sys/kernel/debug/bpmp/debug/clk/emc/rate"

	cpuFreq, gpuFreq, emcFreq = None, None, None
	with open(cpuFreq_fname, 'r') as f:
		cpuFreq = int(f.read().strip('\n'))
	with open(gpuFreq_fname, 'r') as f:
		gpuFreq = int(f.read().strip('\n'))
	with open(emcFreq_fname, 'r') as f:
		emcFreq = int(f.read().strip('\n'))

	return cpuFreq, gpuFreq, emcFreq

def setDVFS(conf):
	"""Set the system knobs, which include DVFS setting on cpu gpu
	and emc, as well as CPU hotplug based on the given parameters"""
	cpuFreq, gpuFreq, emcFreq = conf
	cpuFreq_cur, gpuFreq_cur, emcFreq_cur = getcurStatus()
	
	if cpuFreq != cpuFreq_cur:
		setCpuFreq(cpuFreq, cpuFreq_cur)
	if gpuFreq != gpuFreq_cur:
		setGpuFreq(gpuFreq, gpuFreq_cur)
	if emcFreq != emcFreq_cur:
		setEmcFreq(emcFreq, emcFreq_cur)

	print("Current Frequency", cpuFreq_cur, gpuFreq_cur, emcFreq_cur)

def getCpuStatus():
	"""Get current system knob status, including cpu freqs
	as well as the hotplug status of the Denver cores"""
	
	cpuFreq_fname = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"

	cpuFreq = None
	with open(cpuFreq_fname, 'r') as f:
		cpuFreq = int(f.read().strip('\n'))

	return cpuFreq

def setCpu(cpuFreq):
	"""Set the system knobs, which include DVFS setting on cpu, as well as CPU hotplug based on the given parameters"""
	cpuFreq_cur = getCpuStatus()
	
	if cpuFreq != cpuFreq_cur:
		setCpuFreq(cpuFreq, cpuFreq_cur)

	# print("Current CPU Frequency", cpuFreq_cur)

def getGpuStatus():
	"""Get current system knob status, including gpu freqs
	as well as the hotplug status of the Denver cores"""
	
	gpuFreq_fname = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq"

	gpuFreq = None
	with open(gpuFreq_fname, 'r') as f:
		gpuFreq = int(f.read().strip('\n'))

	return gpuFreq

def setGpu(gpuFreq):
	"""Set the system knobs, which include DVFS setting on gpu, as well as GPU hotplug based on the given parameters"""
	gpuFreq_cur = getGpuStatus()
	
	if gpuFreq != gpuFreq_cur:
		setGpuFreq(gpuFreq, gpuFreq_cur)

	# print("Current GPU Frequency", gpuFreq_cur)

def getEmcStatus():
	"""Get current system knob status, including memory freqs
	as well as the hotplug status of the Denver cores"""
	
	emcFreq_fname = "/sys/kernel/debug/bpmp/debug/clk/emc/rate"

	emcFreq = None
	with open(emcFreq_fname, 'r') as f:
		emcFreq = int(f.read().strip('\n'))

	return emcFreq

def setEmc(emcFreq):
	"""Set the system knobs, which include DVFS setting on
	emc, as well as CPU hotplug based on the given parameters"""
	emcFreq_cur = getEmcStatus()
	
	if emcFreq != emcFreq_cur:
		setEmcFreq(emcFreq, emcFreq_cur)

	# print("Current EMC Frequency", emcFreq_cur)