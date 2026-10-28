"""調色盤 PNG（只用標準函式庫的 zlib）：淹水格網嵌進地圖用。

每格一個位元組（等級），調色盤和透明度由呼叫端決定。大片相同顏色，deflate 壓得很小。
"""

import struct
import zlib

import numpy as np


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def palette_png(indices: np.ndarray, palette: list[tuple[int, int, int, int]]) -> bytes:
    """indices：(高, 寬) 的 uint8，值是 palette 的索引；palette：(R, G, B, A)。"""
    if indices.ndim != 2 or indices.dtype != np.uint8:
        raise ValueError("indices 要是 2 維的 uint8")
    if int(indices.max(initial=0)) >= len(palette) or len(palette) > 256:
        raise ValueError("indices 超出調色盤")
    h, w = indices.shape
    raw = np.zeros((h, w + 1), dtype=np.uint8)  # 每列開頭是 filter 位元組（0 = 不過濾）
    raw[:, 1:] = indices
    header = struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0)  # 8 位元、調色盤
    plte = bytes(c for r, g, b, _ in palette for c in (r, g, b))
    trns = bytes(a for *_, a in palette)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"PLTE", plte)
        + _chunk(b"tRNS", trns)
        + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 9))
        + _chunk(b"IEND", b"")
    )
