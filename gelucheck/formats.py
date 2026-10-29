"""16-bit rounding (FP16, BF16) and rounding cells, computed exactly in float64."""
import numpy as np


def bf16_bits(v):
    """Round float64 values to BF16 (nearest, ties to even) and return the 16-bit encodings.

    Values that are already FP32 (such as the polynomial output p) are rounded exactly. For other
    float64 values, going through FP32 is a double rounding: it can differ from direct rounding
    only when the FP32 value lands exactly on a BF16 midpoint.
    """
    f = np.asarray(v, dtype=np.float64).astype(np.float32).view(np.uint32).astype(np.uint64)
    f = (f + 0x7FFF + ((f >> 16) & 1)) & 0xFFFF0000
    return (f >> 16).astype(np.uint16)


def bf16_value(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def to_bf16(v):
    return bf16_value(bf16_bits(v))


def to_fp16(v):
    return np.asarray(v, dtype=np.float64).astype(np.float16).astype(np.float64)


def fp16_cell(v):
    """Encoding of round_fp16(v) and the two midpoints that bound its rounding cell."""
    h = np.asarray(v, dtype=np.float64).astype(np.float16)
    dn = np.nextafter(h, np.float16(-np.inf)).astype(np.float64)
    up = np.nextafter(h, np.float16(np.inf)).astype(np.float64)
    hv = h.astype(np.float64)
    return h.view(np.uint16), (hv + dn) / 2, (hv + up) / 2


def bf16_cell(v):
    b = bf16_bits(v).astype(np.uint32)
    vv = bf16_value(b)
    n1 = bf16_value((b - 1) & 0xFFFF)
    n2 = bf16_value((b + 1) & 0xFFFF)
    lo, hi = np.minimum(n1, n2), np.maximum(n1, n2)   # bit order flips for negative values
    return b.astype(np.uint16), (vv + lo) / 2, (vv + hi) / 2


CELLS = {"FP16": fp16_cell, "BF16": bf16_cell}
ROUND = {"FP16": to_fp16, "BF16": to_bf16}
