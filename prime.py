import math
import os
from math import isqrt
from multiprocessing import Pool, cpu_count

FILENAME = 'prime.txt'
SEGMENT_SIZE = 1 << 20      # 每段 100 万个数
TASKS_PER_WORKER = 4        # 每个进程一次领多少个段，减少通信开销


# ---------- 基础工具 ----------

def load_primes(filename):
    try:
        with open(filename, 'r') as f:
            primes = [int(line) for line in f if line.strip()]
    except FileNotFoundError:
        return [2]
    return primes or [2]


def simple_sieve(limit):
    """普通筛，生成 [2, limit] 内所有素数。"""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            start = i * i
            sieve[start:limit + 1:i] = b'\x00' * (((limit - start) // i) + 1)
    return [i for i, v in enumerate(sieve) if v]


# ---------- 工作进程 ----------

def sieve_segment(args):
    """
    筛一个区间 (start, end]，返回该区间内的素数列表。
    base_primes 通过全局变量传入，避免每次 pickle 传输（见下方 initializer）。
    """
    start, end = args
    base_primes = _BASE_PRIMES  # 由 initializer 注入

    if end < 2 or end <= start:
        return []

    size = end - start
    sieve = bytearray([1]) * size  # 索引 i 对应数字 start + 1 + i

    for p in base_primes:
        if p * p > end:
            break
        first = max(p * p, ((start // p) + 1) * p)
        if first > end:
            continue
        offset = first - start - 1
        sieve[offset::p] = b'\x00' * (((size - 1 - offset) // p) + 1)

    return [start + 1 + i for i, v in enumerate(sieve) if v]


_BASE_PRIMES = None

def _init_worker(base_primes):
    """在每个工作进程启动时调用一次，把基础素数表放进进程内存。"""
    global _BASE_PRIMES
    _BASE_PRIMES = base_primes


# ---------- 主流程 ----------

def main():
    primes = load_primes(FILENAME)
    cur = primes[-1]
    print(f"last number recorded was: {cur}")

    # 保证基础素数表至少覆盖一个初始范围
    base_limit = max(1000, isqrt(cur) + 1)
    if primes[-1] < base_limit:
        primes = simple_sieve(base_limit)

    num_workers = max(1, cpu_count() - 1)  # 留一个核给主进程
    print(f"using {num_workers} worker processes")

    with open(FILENAME, 'a') as f:
        with Pool(num_workers, initializer=_init_worker,
                  initargs=(primes,)) as pool:
            while True:
                # 规划一批任务：num_workers * TASKS_PER_WORKER 个连续段
                batch = num_workers * TASKS_PER_WORKER
                tasks = []
                for _ in range(batch):
                    start = cur
                    end = start + SEGMENT_SIZE
                    # 保证基础素数覆盖 sqrt(end)
                    needed = isqrt(end)
                    if primes[-1] < needed:
                        primes = simple_sieve(needed)
                        # 基础表变了，需要重建 pool 才能让 worker 用上新表
                        pool.terminate()
                        pool = Pool(num_workers, initializer=_init_worker,
                                    initargs=(primes,))
                    tasks.append((start, end))
                    cur = end

                # 并行筛
                results = pool.map(sieve_segment, tasks)

                # 按区间顺序合并（map 保证顺序）
                all_new = []
                for seg_primes in results:
                    all_new.extend(seg_primes)

                if all_new:
                    for p in all_new:
                        f.write(f"{p}\n")
                    f.flush()
                    primes.extend(all_new)
                    print(f"batch done: +{len(all_new)} primes, "
                          f"total {len(primes)}, "
                          f"last {all_new[-1]}")
                else:
                    print("batch done: no new primes")

if __name__ == '__main__':
    main()
