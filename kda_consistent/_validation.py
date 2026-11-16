"""Shared host-side scalar and buffer validation."""

import math
import struct

def _contract_scalars(scale, eps, lower_bound):
    bits = []
    for name, value in (("scale", scale), ("eps", eps), ("lower_bound", lower_bound)):
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite host scalar")
        try:
            encoded = struct.pack("!f", value)
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{name} must be representable as finite FP32") from error
        bits.append(encoded)
    if struct.unpack("!f", bits[1])[0] <= 0 or not -5 <= lower_bound <= 0:
        raise ValueError("eps must be positive in FP32 and lower_bound must lie in [-5, 0]")
    return tuple(bits)


def _range(t):
    start = t.data_ptr()
    return start, start + t.numel() * t.element_size()


