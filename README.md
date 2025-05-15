# EOCS
The code for the final project of CS 598: Systems for Generative AI, Spring 2025, offered at the University of Illinois Urbana-Champaign.

## How to run the code
Run the stimulation experiment from ``bo/simulate.ipynb``.

Run the profiler for prefill and decode workloads by ``python profiler/profiler_vlm.py``.

Run the profiler for object detection workload by ``python profiler/profiler_yolo.py``.

## File Description
``bo/DVFSController.py``
The python code that conducts the SLO-aware Bayesian Optimization.

``bo/simulate.ipynb``
The python notebook code that simulates the evaluation experiment and plots the results.

``profiler/profiler_vlm.py``
The python code that profiles the frequency-performance Pareto optimality of the prefill and decode workload.

``profiler/profiler_yolo.py``
The python code that profiles the frequency-performance Pareto optimality of the object detection workload.

``profiler/dvfs``
The code module for dynamic voltage and frequency scaling.

``profiler/power``
The code module for power measurement.

``profiler/task``
The code module that conducts the specific workload, including YOLO object detection, Gemma prefill, and Gemma decode.

``dataset``
The SLO trace for object detection, prefill, and decode workloads used in the evaluation.

``result``
The frequency-performance profiling results.
