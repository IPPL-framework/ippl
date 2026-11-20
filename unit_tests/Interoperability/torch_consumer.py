"""
PyTorch consumer used by the DLPackExport unit test.

Receives DLPack capsules exported from IPPL ParticleAttribs, then imports them
into PyTorch. Reads and writes happen in torch, on the same device the attribute lives on.
"""

import torch

# tensors kept alive on the Python side, to test that they keep IPPL memory alive
_held = []


def roundtrip(capsule):
    """Imports the attribute, reports what torch sees, then doubles it in place."""
    t = torch.from_dlpack(capsule)
    info = {
        "device": t.device.type,
        "device_index": -1 if t.device.index is None else t.device.index,
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "strides": list(t.stride()),
        "data": t.data_ptr(),
        "seen": t.reshape(-1).tolist(),  # copied to host
    } # returns this dict of observations to C++ to run assertions on
    t.mul_(2)  # written in place, by a torch kernel on device of t
    if t.is_cuda:
        torch.cuda.synchronize(t.device)
    return info


def hold(capsule):
    _held.append(torch.from_dlpack(capsule))


def drop():
    _held.clear()  # frees the tensors, thereby calling IPPL's deleter


def held_count():
    return len(_held)


def ignore(capsule):
    """Receive a capsule without importing it, its destructor must clean it up."""
    return None


def finalize():
    _held.clear()
