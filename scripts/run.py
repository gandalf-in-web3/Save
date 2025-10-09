"""
用于预约机器的脚本
"""

from concurrent.futures import ProcessPoolExecutor

PAGE = 4096

def occupy_3g():
    size: int = 3 * 1 << 30
    buf = bytearray(size)
    for i in range(0, size, PAGE):
        buf[i] = 1
    
    s: float = 0
    while True:
        s += 1e-10


if __name__ == "__main__":
    with ProcessPoolExecutor(64) as ex:
        futures = [ex.submit(occupy_3g) for _ in range(64)]
        for future in futures:
            future.result()
