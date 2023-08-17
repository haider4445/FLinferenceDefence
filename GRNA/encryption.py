from dataclasses import dataclass
from typing import Any, Dict
from time import time, process_time
import pickle
import tenseal as ts
import torch
import sys
#from pympler import asizeof
import statistics
import humanize

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def encrypt_vector(weights, context):
    start = process_time()
    res = {}
    shapes = {}
    # Do encryption
    v1 = weights.view(-1)
    if len(v1) > 8192//2:
        vals = chunks(v1, 8192//2)
        broken = []
        for chunk in vals:
            broken.append(ts.ckks_vector(context, chunk))
        res = broken
    else:
        res= ts.ckks_vector(context, v1)
    stop = process_time()
    return Results(stop-start, res, shapes)

def encrypt_vector_n(weights, context, n):
	start = process_time()
	res = []
	shapes = weights.shape
	# Do encryption
	v1 = weights.view(-1)
	print('v1 len', len(v1))
	print('v1 type', type(v1))
	print('v1 shape', v1.shape)
	if len(v1) > n//2:
		vals = chunks(v1, n//2)
		broken = []
		for chunk in vals:
			broken.append(ts.ckks_vector(context, chunk.detach().numpy()))
		res = broken
	else:
		res= ts.ckks_vector(context, v1)
	print(res)
	stop = process_time()
	return Results(stop-start, res, shapes)

def decrypt_vector(weights, shapes:Dict):
    start = process_time()
    res = {}
    # Do deencryption
    if isinstance(weights, list):
        lst = []
        for val in weights:
            lst.extend(val.decrypt())
        res = torch.Tensor(lst).view(shapes)
    else:
        res = torch.Tensor(weights.decrypt())
    stop = process_time()
    return Results(stop-start, res, None)

def decrypt(weights, shapes:Dict):
    start = process_time()
    res = {}
    # Do deencryption
    for key in weights:
        if isinstance(weights[key], list):
            lst = []
            for val in weights[key]:
                lst.extend(val.decrypt())
            res[key] = torch.Tensor(lst).view(shapes[key])
                
        else:
            res[key] = torch.Tensor(weights[key].decrypt())
    stop = process_time()
    return Results(stop-start, res, None)


def get_human_readable_bytes(byte_count):
    return humanize.naturalsize(byte_count)

def fsize(stuff, shapes)->int:
    """ The file size in bytes"""
    bytes_s = 0
    for val in stuff:
        if isinstance(stuff[val], ts.tensors.ckksvector.CKKSVector):
            proto = stuff[val].serialize()
            pickle_data = pickle.dumps(proto)
            bytes_s += len(pickle_data)
        else:
            for item in stuff[val]:
                proto = item.serialize()
                pickle_data = pickle.dumps(proto)
                bytes_s += len(pickle_data)
    return get_human_readable_bytes(len(pickle.dumps(shapes)) + bytes_s)


def fsize2(stuff)->int:
    """ The file size in bytes"""
    bytes_s = 0
    for val in stuff:
        if isinstance(stuff[val], ts.tensors.ckksvector.CKKSVector):
            proto = stuff[val].serialize()
            print(type(proto))
            bytes_s += len(proto)
        else:
            for item in stuff[val]:
                proto = item.serialize()
                bytes_s += len(proto)
    return bytes_s