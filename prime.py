import math
import os
import json
from math import isqrt
from multiprocessing import Pool, cpu_count

FILENAME = 'prime.txt'
STATE_FILE = 'prime_state.json'

# 目标：每段大约产出这么多个素数（自适应段长会围绕它调整）
TARGET_PRIMES_PER_SEGMENT = 50_000
MIN_SEGMENT = 1 << 16        # 最小段长 6.5 万
MAX_SEGMENT = 1 << 24        # 最大段长 1600 万
TASKS_PER_WORKER = 4


# ---------- 基础工具 ----------

def load_primes(filename):
    try:
        with open(filename, 'r') as f:
            primes = [int(line) for line in f if line.strip()]
    except FileNotFoundError:
        return [2]
    return primes or [2]


def load_state():
    """读取断点状态；不存在则返回 None。"""
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_state(state):
    """原子写入状态文件，避免崩溃时写坏。"""
    tmp = STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f)
    os.replace(tmp, STATE_FILE)


def simple_sieve(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            start = i * i
            sieve[start:limit + 1:i] = b'\x00' * (((limit - start) // i) + 1)
    return [i for i, v in enumerate(sieve) if v]


def adaptive_segment_size(cur):
    """根据当前数值量级调整段长，让每段产出素数个数大致稳定。"""
    ln = math.log(cur) if cur > 2 else 1.0
    size = int(TARGET_PRIMES_PER_SEGMENT * ln)
    size = max(MIN_SEGMENT, min(MAX_SEGMENT, size))
    # 对齐到 2 的倍数，方便处理（非必须）
    return size & ~1


# ---------- 工作进程 ----------

_BASE_PRIMES = None

def _init_worker(base_primes):
    global _BASE_PRIMES
    _BASE_PRIMES = base_primes


def sieve_segment(args):
    start, end = args
    base_primes = _BASE_PRIMES

    if end < 2 or end <= start:
        return []

    size = end - start
    sieve = bytearray([1]) * size

    for p in base_primes:
        if p * p > end:
            break
        first = max(p * p, ((start // p) + 1) * p)
        if first > end:
            continue
        offset = first - start - 1
        sieve[offset::p] = b'\x00' * (((size - 1 - offset) // p) + 1)

    return [start + 1 + i for i, v in enumerate(sieve) if v]


# ---------- 主流程 ----------

def main():
    primes = load_primes(FILENAME)

    state = load_state()
    if state is None:
        # 首次运行：用文件里最后一个素数作为起点
        cur = primes[-1]
        last_written = primes[-1]
        print(f"[fresh start] last number recorded was: {cur}")
    else:
        cur = state['next_start']
        last_written = state['last_written']
        print(f"[resume] next_start={cur}, last_written={last_written}")
        # 从文件尾部校验：文件最后一个数应与 last_written 一致
        if primes[-1] != last_written:
            print(f"[warn] file last={primes[-1]} != state last={last_written}")
            # 以文件中实际存在的为准，回退到文件末尾
            cur = primes[-1]
            last_written = primes[-1]
            print(f"[warn] falling back to file tail: {cur}")

    # 基础素数表：覆盖 sqrt(第一个段 end) 以内
    first_seg = adaptive_segment_size(cur)
    base_limit = max(1000, isqrt(cur + first_seg * TASKS_PER_WORKER * cpu_count()) + 1)
    if primes[-1] < base_limit:
        primes = simple_sieve(base_limit)

    num_workers = max(1, cpu_count() - 1)
    print(f"using {num_workers} worker processes")

    # 打开文件追加
    f = open(FILENAME, 'a')
    try:
        pool = Pool(num_workers, initializer=_init_worker, initargs=(primes,))

        while True:
            batch = num_workers * TASKS_PER_WORKER
            tasks = []
            for _ in range(batch):
                size = adaptive_segment_size(cur)
                start = cur
                end = start + size

                # 保证基础表覆盖 sqrt(end)
                needed = isqrt(end)
                if primes[-1] < needed:
                    primes = simple_sieve(needed)
                    pool.terminate()
                    pool = Pool(num_workers, initializer=_init_worker,
                                initargs=(primes,))

                tasks.append((start, end))
                cur = end

            results = pool.map(sieve_segment, tasks)

            all_new = []
            for seg_primes in results:
                all_new.extend(seg_primes)

            # 写盘
            if all_new:
                f.write('\n'.join(str(p) for p in all_new) + '\n')
                f.flush()
                os.fsync(f.fileno())  # 强制落盘，断点才可靠
                primes.extend(all_new)
                last_written = all_new[-1]

            # 更新状态（每批一次，代价可忽略）
            save_state({
                'next_start': cur,
                'last_written': last_written,
                'total_primes': len(primes),
            })

            print(f"batch done: +{len(all_new)} primes, "
                  f"total {len(primes)}, last {last_written}, "
                  f"next_start {cur}")

    except KeyboardInterrupt:
        print("\n[interrupted] saving state...")
    finally:
        save_state({
            'next_start': cur,
            'last_written': last_written,
            'total_primes': len(primes),
        })
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass
        f.close()


if __name__ == '__main__':
    main()
