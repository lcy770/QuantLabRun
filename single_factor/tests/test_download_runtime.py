"""用本地HTTP服务器核验分段、续传、重试和校验，测试不会下载大型安装包。"""
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from single_factor.download_runtime import download_file, download_lock


@pytest.fixture
def source():
    """提供可记录连接与请求范围的合成下载源。"""
    state = SimpleNamespace(data=bytes(range(256)) * 40, requests=[], mode="normal",
                            fail_once=False, drop_once=False, resume_range=None,
                            active=0, peak=0, lock=threading.Lock())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            """避免测试输出逐条HTTP访问日志。"""

        def do_GET(self):
            """模拟Range、错误状态码和不规范服务器响应。"""
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"])
            first, last = map(int, match.groups())
            with state.lock:
                state.requests.append((first, last))
                state.active += 1
                state.peak = max(state.peak, state.active)
                failure = state.fail_once and (first, last) != (0, 0)
                if failure:
                    state.fail_once = False
                dropped = state.drop_once and (first, last) != (0, 0)
                if dropped:
                    state.drop_once = False
                    state.resume_range = (first + (last - first + 1) // 2, last)
            try:
                if failure:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = state.data[first:last + 1]
                self.send_response(200 if state.mode == "ignore_range" else 206)
                reported_first = first
                if state.mode == "wrong_range" and (first, last) != (0, 0):
                    reported_first += 1
                self.send_header("Content-Range", f"bytes {reported_first}-{last}/{len(state.data)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                time.sleep(0.02)
                if dropped:
                    body = body[:len(body) // 2]
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with state.lock:
                    state.active -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/wheel"
    state.checksum = sha256(state.data).hexdigest()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_parallel_download_retries_and_reuses_verified_file(source, tmp_path):
    """实际建立多个连接，失败重试后校验结果，重复运行直接复用完整文件。"""
    source.fail_once = True
    output = tmp_path / "runtime.whl"
    assert download_file(source.url, output, source.checksum, len(source.data), workers=4, part_size=1024) == output
    assert output.read_bytes() == source.data
    assert source.peak >= 2
    count = len(source.requests)
    download_file(source.url, output, source.checksum, len(source.data), workers=4, part_size=1024)
    assert len(source.requests) == count
    assert not list(tmp_path.glob("*.parts-*"))


def test_connection_drop_resumes_received_prefix(source, tmp_path):
    """连接在响应中途断开时，从已落盘的偏移继续传输。"""
    source.drop_once = True
    output = tmp_path / "runtime.whl"
    download_file(source.url, output, source.checksum, len(source.data), workers=4, part_size=1024)
    assert output.read_bytes() == source.data
    assert source.resume_range in source.requests


def test_resume_complete_parts_and_incomplete_prefix(source, tmp_path):
    """已完成分段无需重下，未完成分段从已保存的偏移继续。"""
    output = tmp_path / "runtime.whl"
    directory = tmp_path / f"runtime.whl.parts-{source.checksum}-{len(source.data)}-1024"
    directory.mkdir()
    (directory / "000000.part").write_bytes(source.data[:1024])
    (directory / "000001.part").write_bytes(source.data[1024:1424])
    (directory / "000002.part").write_bytes(b"bad" * 1024)
    download_file(source.url, output, source.checksum, len(source.data), workers=4, part_size=1024)
    assert output.read_bytes() == source.data
    assert (0, 1023) not in source.requests
    assert (1424, 2047) in source.requests


@pytest.mark.parametrize("mode", ["ignore_range", "wrong_range"])
def test_bad_range_response_is_rejected(source, tmp_path, mode):
    """服务器忽略范围或返回错误位置时，不发布安装包。"""
    source.mode = mode
    output = tmp_path / "runtime.whl"
    with pytest.raises(ValueError):
        download_file(source.url, output, source.checksum, len(source.data), workers=2, part_size=1024)
    assert not output.exists()


def test_checksum_failure_clears_bad_parts_and_can_retry(source, tmp_path):
    """损坏缓存校验失败后清理，修复校验值后可以直接重试。"""
    output = tmp_path / "runtime.whl"
    with pytest.raises(ValueError, match="SHA256"):
        download_file(source.url, output, "0" * 64, len(source.data), workers=4, part_size=1024)
    assert not output.exists()
    assert not list(tmp_path.glob("*.parts-*"))
    assert not list(tmp_path.glob("*.assembling"))
    download_file(source.url, output, source.checksum, len(source.data), workers=4, part_size=1024)
    assert output.read_bytes() == source.data


def test_existing_unrelated_file_is_preserved(source, tmp_path):
    """目标已有其他内容时保留文件，避免覆盖用户数据。"""
    output = tmp_path / "runtime.whl"
    output.write_bytes(b"keep existing data")
    with pytest.raises(FileExistsError):
        download_file(source.url, output, source.checksum, len(source.data))
    assert output.read_bytes() == b"keep existing data"
    assert source.requests == []


def test_concurrent_download_of_same_file_is_rejected(tmp_path):
    """两个进程指向同一目标时，第二个进程应在写数据前退出。"""
    output = tmp_path / "runtime.whl"
    with download_lock(tmp_path / "runtime.whl.lock"):
        with pytest.raises(RuntimeError, match="下载进程"):
            download_file("http://127.0.0.1:1", output, sha256(b"x").hexdigest(), 1)


def test_bootstrap_cli_needs_only_standard_library():
    """禁用第三方site-packages后，下载入口仍能启动。"""
    result = subprocess.run([sys.executable, "-S", "-m", "single_factor.download_runtime", "--help"],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True)
    assert result.returncode == 0, result.stderr
    assert b"--workers" in result.stdout
