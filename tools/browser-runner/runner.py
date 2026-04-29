#!/usr/bin/env python3
"""Local HTTP runner for browser-based IPPL examples.

The service is intentionally small and dependency-free. It accepts one C++ source file, compiles it
against an installed IPPL package, runs the executable under MPI, and returns stdout/stderr as JSON.
It is meant to be run inside a Docker container bound to 127.0.0.1 on the host.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = r'''#include "Ippl.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    *ippl::Info << "Hello from rank "
                << ippl::Comm->rank()
                << " of "
                << ippl::Comm->size()
                << ippl::endl;

    ippl::finalize();
    return 0;
}
'''


def run_command(command: list[str], cwd: Path, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timedOut": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\nCommand timed out after {timeout} seconds.\n",
            "timedOut": True,
        }


class RunnerState:
    def __init__(self, token: str, prefix: str, max_source_bytes: int) -> None:
        self.token = token
        self.prefix = prefix
        self.max_source_bytes = max_source_bytes


class RunnerHandler(BaseHTTPRequestHandler):
    server_version = "IPPLRunner/0.1"

    @property
    def state(self) -> RunnerState:
        return self.server.state  # type: ignore[attr-defined]

    def end_headers(self) -> None:
        origin = self.headers.get("Origin")
        allowed = os.environ.get(
            "IPPL_RUNNER_ALLOWED_ORIGINS",
            "https://ippl-framework.github.io,https://IPPL-framework.github.io,http://127.0.0.1:8080,http://localhost:8080,null",
        ).split(",")
        if origin in allowed:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self.write_json(
                {
                    "ok": True,
                    "service": "ippl-runner",
                    "tokenRequired": True,
                    "prefix": self.state.prefix,
                }
            )
            return
        if self.path == "/example/hello":
            self.write_json({"source": DEFAULT_SOURCE, "ranks": 2})
            return
        self.write_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/compile-run":
            self.write_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
            return

        payload = self.read_json()
        if payload is None:
            return

        if payload.get("token") != self.state.token:
            self.write_json({"ok": False, "error": "invalid runner token"}, HTTPStatus.FORBIDDEN)
            return

        source = payload.get("source", "")
        if not isinstance(source, str) or not source.strip():
            self.write_json({"ok": False, "error": "source must be a non-empty string"}, HTTPStatus.BAD_REQUEST)
            return
        if len(source.encode("utf-8")) > self.state.max_source_bytes:
            self.write_json(
                {"ok": False, "error": f"source exceeds {self.state.max_source_bytes} bytes"},
                HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            ranks = int(payload.get("ranks", 2))
        except (TypeError, ValueError):
            ranks = 2
        ranks = max(1, min(ranks, 4))

        result = self.compile_and_run(source, ranks)
        self.write_json(result, HTTPStatus.OK if result["ok"] else HTTPStatus.BAD_REQUEST)

    def read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self.write_json({"ok": False, "error": "empty request body"}, HTTPStatus.BAD_REQUEST)
            return None
        try:
            data = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            self.write_json({"ok": False, "error": f"invalid JSON: {exc}"}, HTTPStatus.BAD_REQUEST)
            return None
        if not isinstance(data, dict):
            self.write_json({"ok": False, "error": "request body must be a JSON object"}, HTTPStatus.BAD_REQUEST)
            return None
        return data

    def compile_and_run(self, source: str, ranks: int) -> dict[str, Any]:
        workdir = Path(tempfile.mkdtemp(prefix="ippl-runner-"))
        try:
            (workdir / "main.cpp").write_text(source, encoding="utf-8")
            (workdir / "CMakeLists.txt").write_text(
                """cmake_minimum_required(VERSION 3.24)
project(ippl_browser_example LANGUAGES CXX)
find_package(IPPL REQUIRED CONFIG)
add_executable(ippl_browser_example main.cpp)
target_link_libraries(ippl_browser_example PRIVATE IPPL::ippl)
""",
                encoding="utf-8",
            )

            configure = run_command(
                [
                    "cmake",
                    "-S",
                    ".",
                    "-B",
                    "build",
                    "-G",
                    "Ninja",
                    "-DCMAKE_CXX_COMPILER=mpicxx",
                    f"-DCMAKE_PREFIX_PATH={self.state.prefix}",
                ],
                workdir,
                timeout=30,
            )
            if configure["returncode"] != 0:
                return {"ok": False, "stage": "configure", "configure": configure}

            build = run_command(
                ["cmake", "--build", "build", "--target", "ippl_browser_example", "--parallel", "2"],
                workdir,
                timeout=60,
            )
            if build["returncode"] != 0:
                return {"ok": False, "stage": "build", "configure": configure, "build": build}

            run = run_command(
                ["mpiexec", "--oversubscribe", "-n", str(ranks), "build/ippl_browser_example"],
                workdir,
                timeout=15,
            )
            return {
                "ok": run["returncode"] == 0,
                "stage": "run",
                "ranks": ranks,
                "configure": configure,
                "build": build,
                "run": run,
            }
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("IPPL_RUNNER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("IPPL_RUNNER_PORT", "5050")))
    parser.add_argument("--prefix", default=os.environ.get("IPPL_PREFIX", "/opt/ippl"))
    parser.add_argument("--max-source-bytes", type=int, default=64 * 1024)
    args = parser.parse_args()

    token = os.environ.get("IPPL_RUNNER_TOKEN") or secrets.token_urlsafe(18)
    server = ThreadingHTTPServer((args.host, args.port), RunnerHandler)
    server.state = RunnerState(token=token, prefix=args.prefix, max_source_bytes=args.max_source_bytes)  # type: ignore[attr-defined]

    print(f"IPPL runner listening on http://{args.host}:{args.port}", flush=True)
    print(f"IPPL runner token: {token}", flush=True)
    print("Bind the Docker port to 127.0.0.1 on the host.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
