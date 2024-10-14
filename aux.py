"""This module contains auxiliary functions.

Miguel Dovale (Hannover, 2024)
"""
import math
import numpy as np

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

def find_Jdes_binary_search(scheduler, params, target_nf, min_Jdes=100, max_Jdes=5000):
    while min_Jdes <= max_Jdes:
        Jdes = (min_Jdes + max_Jdes) // 2
        params["Jdes"] = Jdes
        output = scheduler(params)
        nf = output["nf"]
        if nf == target_nf:
            return Jdes
        elif nf < target_nf:
            min_Jdes = Jdes + 1
        else:
            max_Jdes = Jdes - 1