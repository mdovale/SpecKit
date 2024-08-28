"""This module contains auxiliary functions.

Miguel Dovale (Hannover, 2024)
"""

def chunker(iter, chunk_size):
    chunks = []
    if chunk_size < 1:
        raise ValueError('Chunk size must be greater than 0.')
    for i in range(0, len(iter), chunk_size):
        chunks.append(iter[i:(i+chunk_size)])
    return chunks