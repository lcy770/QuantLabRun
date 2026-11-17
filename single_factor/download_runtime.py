"""用Python标准库并行下载实验机的CUDA版PyTorch，支持续传和官方SHA256校验。"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import hashlib
from http.client import HTTPException
import os
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen


WHEEL = "torch-2.11.0+cu128-cp311-cp311-win_amd64.whl"
URL = "https://mirrors.aliyun.com/pytorch-wheels/cu128/" + WHEEL.replace("+", "%2B")
SIZE = 2753148611
# 来自PyTorch官方cu128/torch索引中这个wheel的sha256锚点。
SHA256 = "90ef0c2454e5296a9fb021ddd42252e4ce1abe2c0a4988a173ef90a6cded0bf5"


def file_digest(path):
    """流式计算SHA256，内存占用与安装包大小无关。"""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@contextmanager
def download_lock(path):
    """同一目标仅允许一个下载进程写入，退出或崩溃时由操作系统释放锁。"""
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("同一安装包已有下载进程，请保留一个下载窗口") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def fetch_part(url, path, first, last, total, stopped):
    """单连接续传一个分段，严格验证响应范围，网络中断最多重试四次。"""
    length = last - first + 1
    if path.exists() and path.stat().st_size > length:
        path.unlink()
    for attempt in range(4):
        if stopped.is_set():
            return
        received = path.stat().st_size if path.exists() else 0
        if received == length:
            return
        begin = first + received
        headers = {"Range": f"bytes={begin}-{last}", "Accept-Encoding": "identity"}
        try:
            with urlopen(Request(url, headers=headers), timeout=20) as response:
                expected = f"bytes {begin}-{last}/{total}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("服务器未正确响应分段请求，已停止下载以保护缓存")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("服务器对分段内容进行了额外压缩")
                remaining = length - received
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != remaining:
                    raise ValueError("分段响应长度与请求范围不一致")
                with path.open("ab") as destination:
                    while remaining and not stopped.is_set():
                        block = response.read(min(64 * 1024, remaining))
                        if not block:
                            raise OSError("分段连接提前结束")
                        destination.write(block)
                        remaining -= len(block)
                if stopped.is_set():
                    return
                if response.read(1):
                    raise ValueError("服务器返回的数据超出了分段边界")
            return
        except (OSError, HTTPException):
            if attempt == 3:
                raise
            stopped.wait(attempt + 1)


def clear_parts(directory, parts):
    """仅删除本次下载拥有的分段文件，保留目录中其他文件。"""
    for part in parts:
        part.unlink(missing_ok=True)
    try:
        directory.rmdir()
    except OSError:
        pass


def download_file(url, output, checksum, size, *, workers=8, part_size=16 * 1024 * 1024):
    """并发下载并原子发布已校验文件，重试时复用完整分段和未完成分段前缀。"""
    if workers < 1 or part_size < 1 or size < 1:
        raise ValueError("连接数、分段大小和文件大小必须为正数")
    if len(checksum) != 64 or any(letter not in "0123456789abcdef" for letter in checksum):
        raise ValueError("需要完整的小写SHA256校验值")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with download_lock(output.with_name(output.name + ".lock")):
        if output.exists():
            if output.stat().st_size != size or file_digest(output) != checksum:
                raise FileExistsError("目标文件已存在但校验失败，请另存现有文件后重试")
            print(f"Verified existing file: {output}", flush=True)
            return output
        # 在大文件传输前先验证Range能力和已知文件大小。
        with urlopen(Request(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"}), timeout=20) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes 0-0/{size}":
                raise ValueError("下载源未正确支持分段请求或安装包大小发生变化")
            if len(response.read(1)) != 1:
                raise ValueError("下载源返回空内容")
        directory = output.with_name(f"{output.name}.parts-{checksum}-{size}-{part_size}")
        directory.mkdir(exist_ok=True)
        ranges = [(first, min(first + part_size, size) - 1) for first in range(0, size, part_size)]
        parts = [directory / f"{index:06d}.part" for index in range(len(ranges))]
        for part, (first, last) in zip(parts, ranges):
            if part.exists() and part.stat().st_size > last - first + 1:
                part.unlink()
        initial = sum(part.stat().st_size for part in parts if part.exists())
        print(f"Size={size / 1024**2:.1f} MiB  cached={initial / 1024**2:.1f} MiB  workers={workers}", flush=True)
        started = last_report = time.monotonic()
        stopped = threading.Event()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fetch_part, url, part, first, last, size, stopped)
                       for part, (first, last) in zip(parts, ranges)]
            pending = set(futures)
            try:
                while pending:
                    done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                    now = time.monotonic()
                    if now - last_report >= 1 or not pending:
                        received = sum(part.stat().st_size for part in parts if part.exists())
                        speed = (received - initial) / max(now - started, 0.001)
                        eta = f"{max(0, size - received) / speed / 60:.1f} min" if speed > 0 else "pending"
                        print(f"\r{received / size:.1%}  DL={speed / 1024**2:.2f} MiB/s  ETA={eta}     ", end="", flush=True)
                        last_report = now
            except BaseException:
                stopped.set()
                for future in futures:
                    future.cancel()
                raise
        print("\nChecking SHA256...", flush=True)
        temporary = output.with_name(output.name + ".assembling")
        result = hashlib.sha256()
        with temporary.open("wb") as destination:
            for part in parts:
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        result.update(block)
                        destination.write(block)
        if temporary.stat().st_size != size or result.hexdigest() != checksum:
            temporary.unlink()
            clear_parts(directory, parts)
            raise ValueError("官方SHA256校验失败，损坏分段已清理，请重新运行下载命令")
        os.replace(temporary, output)
        clear_parts(directory, parts)
        print(f"Verified download: {output}", flush=True)
        return output


def main():
    """提供无需安装Torch或第三方下载器的Windows实验机入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8, help="并行连接数，默认8")
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "var" / "wheels" / WHEEL
    try:
        download_file(URL, output, SHA256, SIZE, workers=args.workers)
    except KeyboardInterrupt:
        print("\n下载已中断，分段已保留，重复原命令可以续传。", flush=True)
        raise SystemExit(130) from None
    except (OSError, HTTPException, ValueError, RuntimeError) as error:
        parser.exit(1, f"\n下载失败：{error}\n处理连接或磁盘问题后，可以重复原命令重试。\n")


if __name__ == "__main__":
    main()
