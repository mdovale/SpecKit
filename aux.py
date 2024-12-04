"""This module contains auxiliary functions.

Miguel Dovale (Hannover, 2024)
E-mail: spectools@pm.me
"""
import math
import torch
import numpy as np
from spectools.schedulers import lpsd_plan, ltf_plan, new_ltf_plan

def kaiser_alpha(psll):
    a0 = -0.0821377
    a1 = 4.71469
    a2 = -0.493285
    a3 = 0.0889732

    x = psll / 100
    return (((((a3 * x) + a2) * x) + a1) * x + a0)

def kaiser_rov(alpha):
    a0 = 0.0061076
    a1 = 0.00912223
    a2 = -0.000925946
    a3 = 4.42204e-05
    x = alpha
    return (100 - 1 / (((((a3 * x) + a2) * x) + a1) * x + a0)) / 100

def round_half_up(val):
    if (float(val) % 1) >= 0.5:
        x = math.ceil(val)
    else:
        x = round(val)
    return x

def chunker(iter, chunk_size):
    chunks = []
    if chunk_size < 1:
        raise ValueError('Chunk size must be greater than 0.')
    for i in range(0, len(iter), chunk_size):
        chunks.append(iter[i:(i+chunk_size)])
    return chunks

# Function to check if a function is contained in the dictionary
def is_function_in_dict(function_to_check, function_dict):
    return function_to_check in function_dict.values()

# Function to get the key corresponding to a function
def get_key_for_function(function_to_check, function_dict):
    for key, func in function_dict.items():
        if func == function_to_check:
            return key
    return None  # Return None if the function is not found

def find_Jdes_binary_search(scheduler, target_nf, min_Jdes=100, max_Jdes=5000, *args):
     while (min_Jdes <= max_Jdes):
        Jdes = (min_Jdes + max_Jdes) // 2
        if scheduler == lpsd_plan:
            output = scheduler(*args[:3], Jdes, *args[3:])
        else:
            output = scheduler(*args[:5], Jdes, *args[5:])
        nf = output.get("nf")
        if nf is None:
            raise ValueError("Scheduler did not return 'nf' in output.")
        if nf == target_nf:
            return Jdes
        elif nf < target_nf:
            min_Jdes = Jdes + 1
        else:
            max_Jdes = Jdes - 1

def check_gpu_availability():
    """
    Check if GPU-accelerated devices (MPS, CUDA, or others) are available.

    Returns
    -------
    dict: A dictionary containing the status of MPS, CUDA, and other GPU devices.
    """
    availability = {
        "CUDA": torch.cuda.is_available(),
        "MPS": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        "GPU Devices": [],
    }

    if availability["CUDA"]:
        num_cuda_devices = torch.cuda.device_count()
        availability["GPU Devices"].extend(
            [torch.cuda.get_device_name(i) for i in range(num_cuda_devices)]
        )
    elif availability["MPS"]:
        availability["GPU Devices"].append("Metal Performance Shaders (MPS)")

    return availability


# Print the results
if __name__ == "__main__":
    availability = check_gpu_availability()
    print("CUDA Available:", availability["CUDA"])
    print("MPS Available:", availability["MPS"])
    if availability["GPU Devices"]:
        print("Available GPU Devices:")
        for device in availability["GPU Devices"]:
            print(f"- {device}")
    else:
        print("No GPU-accelerated devices detected.")