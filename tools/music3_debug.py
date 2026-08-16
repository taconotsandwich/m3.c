#!/usr/bin/env python3

"""Run the public Music3 benchmark with macOS memory/GPU monitoring."""

from __future__ import annotations

import argparse
import glob
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TextIO


@dataclass(frozen=True)
class SystemSample:
    memory_free_percent: int
    swap_used_mib: float
    swap_free_mib: float


@dataclass(frozen=True)
class ProcessSample:
    cpu_percent: float
    rss_kib: int


@dataclass(frozen=True)
class GpuSample:
    device_percent: int
    renderer_percent: int
    tiler_percent: int


@dataclass(frozen=True)
class Settings:
    harness: Path
    model_root: Path
    wave_path: Path
    log_path: Path
    maximum_frames: int
    seed: int
    sequence: int
    caption: str
    lyrics: str
    baseline_samples: int
    baseline_interval: float
    poll_interval: float
    minimum_baseline_memory: int
    maximum_baseline_swap_range_mib: float
    minimum_baseline_swap_free_mib: float
    minimum_running_memory: int
    maximum_running_swap_growth_mib: float
    minimum_running_swap_free_mib: float
    maximum_running_rss_mib: float
    grace_seconds: float
    delete_wave: bool
    sample_only: bool
    quiet: bool


def run_text(arguments: list[str]) -> str:
    return subprocess.run(
        arguments,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def amount_mib(value: str, unit: str) -> float:
    amount = float(value)
    if unit == "G":
        return amount * 1024.0
    if unit == "K":
        return amount / 1024.0
    if unit == "B":
        return amount / (1024.0 * 1024.0)
    return amount


def system_sample() -> SystemSample:
    pressure = run_text(["/usr/bin/memory_pressure", "-Q"])
    swap = run_text(["/usr/sbin/sysctl", "vm.swapusage"])
    free_match = re.search(
        r"System-wide memory free percentage:\s*(\d+)%", pressure
    )
    swap_match = re.search(
        r"used\s*=\s*([0-9.]+)([BKMG])\s+free\s*=\s*([0-9.]+)([BKMG])",
        swap,
    )
    if free_match is None or swap_match is None:
        raise RuntimeError("unable to parse memory or swap state")
    return SystemSample(
        memory_free_percent=int(free_match.group(1)),
        swap_used_mib=amount_mib(swap_match.group(1), swap_match.group(2)),
        swap_free_mib=amount_mib(swap_match.group(3), swap_match.group(4)),
    )


def process_sample(process_id: int) -> ProcessSample:
    output = run_text(
        ["/bin/ps", "-o", "%cpu=,rss=", "-p", str(process_id)]
    )
    fields = output.split()
    if len(fields) != 2:
        return ProcessSample(0.0, 0)
    return ProcessSample(float(fields[0]), int(fields[1]))


def gpu_sample() -> GpuSample:
    output = run_text(
        [
            "/usr/sbin/ioreg",
            "-r",
            "-c",
            "AGXAccelerator",
            "-d",
            "1",
            "-k",
            "PerformanceStatistics",
        ]
    )
    section_match = re.search(
        r'"PerformanceStatistics"\s*=\s*\{([^\n]+)\}', output
    )
    if section_match is None:
        raise RuntimeError("unable to parse GPU statistics")
    section = section_match.group(1)

    def percentage(name: str) -> int:
        match = re.search(rf'"{re.escape(name)}"=(\d+)', section)
        if match is None:
            raise RuntimeError(f"missing GPU statistic: {name}")
        return int(match.group(1))

    return GpuSample(
        percentage("Device Utilization %"),
        percentage("Renderer Utilization %"),
        percentage("Tiler Utilization %"),
    )


def colima_running() -> bool:
    output = run_text(["/bin/ps", "-axo", "command="])
    return (
        "/opt/homebrew/opt/colima/bin/colima" in output
        or "Virtualization.VirtualMachine.xpc" in output
    )


class Monitor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.write_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.log_file: TextIO | None = None
        self.child: subprocess.Popen[str] | None = None
        self.output_thread: threading.Thread | None = None
        self.external_stop = False
        self.current_phase = "open"
        self.phase_samples: dict[str, list[tuple[float, int]]] = {}
        self.wave_cleanup_allowed = False

    def emit(self, message: str) -> None:
        line = f"MONITOR {message}\n"
        with self.write_lock:
            if not self.settings.quiet:
                sys.stdout.write(line)
                sys.stdout.flush()
            if self.log_file is not None:
                self.log_file.write(line)
                self.log_file.flush()

    def forward_signal(
        self, signal_number: int, frame: FrameType | None
    ) -> None:
        del signal_number, frame
        self.external_stop = True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)

    def drain_output(self, stream: TextIO) -> None:
        for line in stream:
            phase_match = re.search(r"PHASE_BEGIN phase=([a-z_]+)", line)
            if phase_match is not None:
                with self.state_lock:
                    self.current_phase = phase_match.group(1)
            with self.write_lock:
                if not self.settings.quiet:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                if self.log_file is not None:
                    self.log_file.write(line)
                    self.log_file.flush()

    def cleanup_wave_artifacts(self, include_final: bool) -> None:
        paths = glob.glob(f"{self.settings.wave_path}.tmp.*")
        if include_final:
            paths.append(str(self.settings.wave_path))
        for path in paths:
            try:
                os.unlink(path)
                self.emit(f"REMOVE path={path}")
            except FileNotFoundError:
                pass

    def baseline(self) -> SystemSample | None:
        settings = self.settings
        samples: list[SystemSample] = []

        for index in range(settings.baseline_samples):
            if self.external_stop:
                self.emit("BASELINE_ABORT reason=external_stop")
                return None
            sample = system_sample()
            if colima_running():
                self.emit(
                    f"BASELINE_REJECT index={index + 1} reason=colima_running"
                )
                return None
            if self.external_stop:
                self.emit("BASELINE_ABORT reason=external_stop")
                return None
            samples.append(sample)
            self.emit(
                f"BASELINE index={index + 1} "
                f"memory_free={sample.memory_free_percent}% "
                f"swap_used={sample.swap_used_mib:.2f}MiB "
                f"swap_free={sample.swap_free_mib:.2f}MiB"
            )
            if index + 1 < settings.baseline_samples:
                time.sleep(settings.baseline_interval)
        swap_range = max(item.swap_used_mib for item in samples) - min(
            item.swap_used_mib for item in samples
        )
        accepted = (
            min(item.memory_free_percent for item in samples)
            >= settings.minimum_baseline_memory
            and swap_range <= settings.maximum_baseline_swap_range_mib
            and min(item.swap_free_mib for item in samples)
            >= settings.minimum_baseline_swap_free_mib
        )
        if not accepted:
            self.emit(f"BASELINE_REJECT swap_range={swap_range:.2f}MiB")
            return None
        self.emit(f"BASELINE_ACCEPT swap_range={swap_range:.2f}MiB")
        return samples[-1]

    def command(self) -> list[str]:
        settings = self.settings
        return [
            str(settings.harness),
            str(settings.model_root),
            str(settings.wave_path),
            str(settings.maximum_frames),
            str(settings.seed),
            str(settings.sequence),
            settings.caption,
            settings.lyrics,
        ]

    def launch(self, baseline: SystemSample) -> int:
        settings = self.settings
        self.child = subprocess.Popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if self.child.stdout is None:
            raise RuntimeError("benchmark stdout pipe was not created")
        self.output_thread = threading.Thread(
            target=self.drain_output,
            args=(self.child.stdout,),
            daemon=True,
        )
        self.output_thread.start()
        self.emit(
            f"START pid={self.child.pid} "
            f"baseline_swap_used={baseline.swap_used_mib:.2f}MiB"
        )
        return self.monitor_child(baseline)

    def monitor_child(self, baseline: SystemSample) -> int:
        assert self.child is not None
        settings = self.settings
        stop_reason: str | None = None
        stop_time = 0.0
        last_report = 0.0
        run_start = time.monotonic()
        minimum_memory_free = 100
        maximum_swap_used = baseline.swap_used_mib
        maximum_swap_growth = 0.0
        maximum_rss_kib = 0
        cpu_values: list[float] = []
        gpu_values: list[int] = []

        while self.child.poll() is None:
            sample: SystemSample | None = None
            process = ProcessSample(0.0, 0)
            gpu = GpuSample(-1, -1, -1)
            colima_active = False
            try:
                sample = system_sample()
                process = process_sample(self.child.pid)
                colima_active = colima_running()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                if self.child.poll() is None:
                    stop_reason = f"monitor sampling failed: {error}"
            if sample is not None and stop_reason is None:
                try:
                    gpu = gpu_sample()
                except (
                    OSError,
                    ValueError,
                    subprocess.SubprocessError,
                    RuntimeError,
                ):
                    gpu = GpuSample(-1, -1, -1)
                swap_growth = sample.swap_used_mib - baseline.swap_used_mib
                minimum_memory_free = min(
                    minimum_memory_free, sample.memory_free_percent
                )
                maximum_swap_used = max(maximum_swap_used, sample.swap_used_mib)
                maximum_swap_growth = max(maximum_swap_growth, swap_growth)
                maximum_rss_kib = max(maximum_rss_kib, process.rss_kib)
                cpu_values.append(process.cpu_percent)
                if gpu.device_percent >= 0:
                    gpu_values.append(gpu.device_percent)
                    with self.state_lock:
                        self.phase_samples.setdefault(
                            self.current_phase, []
                        ).append((process.cpu_percent, gpu.device_percent))
                if sample.memory_free_percent <= settings.minimum_running_memory:
                    stop_reason = "memory free threshold"
                elif (
                    swap_growth >= settings.maximum_running_swap_growth_mib
                ):
                    stop_reason = "swap growth threshold"
                elif (
                    sample.swap_free_mib <= settings.minimum_running_swap_free_mib
                ):
                    stop_reason = "swap free threshold"
                elif process.rss_kib >= settings.maximum_running_rss_mib * 1024.0:
                    stop_reason = "RSS threshold"
                elif colima_active:
                    stop_reason = "Colima restarted"
                now = time.monotonic()
                if now - last_report >= 10.0:
                    self.emit(
                        f"SAMPLE memory_free={sample.memory_free_percent}% "
                        f"swap_used={sample.swap_used_mib:.2f}MiB "
                        f"swap_growth={swap_growth:.2f}MiB "
                        f"swap_free={sample.swap_free_mib:.2f}MiB "
                        f"rss={process.rss_kib / 1024.0:.2f}MiB "
                        f"cpu={process.cpu_percent:.1f}% "
                        f"gpu={gpu.device_percent}% "
                        f"renderer={gpu.renderer_percent}% "
                        f"tiler={gpu.tiler_percent}%"
                    )
                    last_report = now
            if self.external_stop and stop_reason is None:
                stop_reason = "external stop"
            if stop_reason is not None and stop_time == 0.0:
                self.emit(f"CANCEL reason={stop_reason}")
                self.child.send_signal(signal.SIGTERM)
                stop_time = time.monotonic()
            if (
                stop_time != 0.0
                and time.monotonic() - stop_time >= settings.grace_seconds
            ):
                self.emit("KILL grace_period_expired")
                self.child.kill()
                break
            time.sleep(settings.poll_interval)
        return_code = self.child.wait()
        if self.output_thread is not None:
            self.output_thread.join()
            self.output_thread = None
        runtime = time.monotonic() - run_start
        average_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
        maximum_cpu = max(cpu_values) if cpu_values else 0.0
        average_gpu = sum(gpu_values) / len(gpu_values) if gpu_values else 0.0
        maximum_gpu = max(gpu_values) if gpu_values else 0
        self.emit(
            f"SUMMARY runtime={runtime:.6f}s samples={len(cpu_values)} "
            f"min_memory_free={minimum_memory_free}% "
            f"max_swap_used={maximum_swap_used:.2f}MiB "
            f"max_swap_growth={maximum_swap_growth:.2f}MiB "
            f"max_rss={maximum_rss_kib / 1024.0:.2f}MiB "
            f"avg_cpu={average_cpu:.2f}% max_cpu={maximum_cpu:.2f}% "
            f"avg_gpu={average_gpu:.2f}% max_gpu={maximum_gpu}%"
        )
        with self.state_lock:
            collected = dict(self.phase_samples)
        for phase_name, values in collected.items():
            phase_cpu = [value[0] for value in values]
            phase_gpu = [value[1] for value in values]
            self.emit(
                f"PHASE_METRICS phase={phase_name} samples={len(values)} "
                f"avg_cpu={sum(phase_cpu) / len(phase_cpu):.2f}% "
                f"max_cpu={max(phase_cpu):.2f}% "
                f"avg_gpu={sum(phase_gpu) / len(phase_gpu):.2f}% "
                f"max_gpu={max(phase_gpu)}%"
            )
        self.emit(f"EXIT benchmark_status={return_code} stop_reason={stop_reason}")
        if stop_reason is not None:
            return 2
        return return_code

    def run(self) -> int:
        settings = self.settings
        settings.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = settings.log_path.open(
            "w", encoding="utf-8", buffering=1
        )
        self.emit(
            f"GATES baseline_memory={settings.minimum_baseline_memory}% "
            f"baseline_swap_range="
            f"{settings.maximum_baseline_swap_range_mib:.0f}MiB "
            f"baseline_swap_free="
            f"{settings.minimum_baseline_swap_free_mib:.0f}MiB "
            f"running_memory={settings.minimum_running_memory}% "
            f"running_swap_growth="
            f"{settings.maximum_running_swap_growth_mib:.0f}MiB "
            f"running_swap_free="
            f"{settings.minimum_running_swap_free_mib:.0f}MiB "
            f"poll={settings.poll_interval:.1f}s"
        )
        if settings.sample_only:
            sample = system_sample()
            gpu = gpu_sample()
            self.emit(
                f"SAMPLE_ONLY memory_free={sample.memory_free_percent}% "
                f"swap_used={sample.swap_used_mib:.2f}MiB "
                f"swap_free={sample.swap_free_mib:.2f}MiB "
                f"gpu={gpu.device_percent}%"
            )
            return 0
        if not settings.harness.is_file() or not os.access(
            settings.harness, os.X_OK
        ):
            raise RuntimeError(
                f"benchmark is missing or not executable: {settings.harness}"
            )
        if not settings.model_root.is_dir():
            raise RuntimeError(f"model root is missing: {settings.model_root}")
        self.cleanup_wave_artifacts(include_final=False)
        if settings.wave_path.exists():
            raise RuntimeError(
                f"refusing to replace existing wave file: {settings.wave_path}"
            )
        self.wave_cleanup_allowed = True
        baseline = self.baseline()
        if baseline is None or self.external_stop:
            return 3
        return self.launch(baseline)

    def close(self, successful: bool) -> None:
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)
            try:
                self.child.wait(timeout=self.settings.grace_seconds)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
        if self.output_thread is not None:
            self.output_thread.join()
            self.output_thread = None
        self.cleanup_wave_artifacts(
            include_final=self.wave_cleanup_allowed
            and (self.settings.delete_wave or not successful)
        )
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None


def parse_arguments() -> Settings:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the public Music3 benchmark while logging memory, swap, CPU, "
            "GPU, and phase timing. The monitor cancels before configured "
            "resource limits are crossed."
        )
    )
    parser.add_argument(
        "--harness", type=Path, default=root / "build/music3-benchmark"
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=root / "models/MiniMax-Music3-bd348",
    )
    parser.add_argument(
        "--wave", type=Path, default=Path("/private/tmp/m3_music3_debug.wav")
    )
    parser.add_argument(
        "--log", type=Path, default=Path("/private/tmp/m3_music3_debug.log")
    )
    parser.add_argument("--maximum-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--sequence", type=int, default=0)
    parser.add_argument("--caption", default="A gentle piano melody.")
    parser.add_argument("--lyrics", default="[Verse]\nHello world.")
    parser.add_argument("--baseline-samples", type=int, default=3)
    parser.add_argument("--baseline-interval", type=float, default=5.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--minimum-baseline-memory", type=int, default=85)
    parser.add_argument(
        "--maximum-baseline-swap-range-mib", type=float, default=32.0
    )
    parser.add_argument(
        "--minimum-baseline-swap-free-mib", type=float, default=512.0
    )
    parser.add_argument("--minimum-running-memory", type=int, default=20)
    parser.add_argument(
        "--maximum-running-swap-growth-mib", type=float, default=4096.0
    )
    parser.add_argument(
        "--minimum-running-swap-free-mib", type=float, default=128.0
    )
    parser.add_argument("--maximum-running-rss-mib", type=float, default=24576.0)
    parser.add_argument("--grace-seconds", type=float, default=60.0)
    parser.add_argument("--delete-wave", action="store_true")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    positive_values = (
        arguments.maximum_frames,
        arguments.baseline_samples,
        arguments.baseline_interval,
        arguments.poll_interval,
        arguments.grace_seconds,
    )
    if any(value <= 0 for value in positive_values):
        parser.error("frame, sample, interval, poll, and grace values must be positive")
    if not 0 <= arguments.minimum_baseline_memory <= 100:
        parser.error("baseline memory percentage must be in [0, 100]")
    if not 0 <= arguments.minimum_running_memory <= 100:
        parser.error("running memory percentage must be in [0, 100]")
    return Settings(
        harness=arguments.harness.resolve(),
        model_root=arguments.model_root.resolve(),
        wave_path=arguments.wave.resolve(),
        log_path=arguments.log.resolve(),
        maximum_frames=arguments.maximum_frames,
        seed=arguments.seed,
        sequence=arguments.sequence,
        caption=arguments.caption,
        lyrics=arguments.lyrics,
        baseline_samples=arguments.baseline_samples,
        baseline_interval=arguments.baseline_interval,
        poll_interval=arguments.poll_interval,
        minimum_baseline_memory=arguments.minimum_baseline_memory,
        maximum_baseline_swap_range_mib=(
            arguments.maximum_baseline_swap_range_mib
        ),
        minimum_baseline_swap_free_mib=arguments.minimum_baseline_swap_free_mib,
        minimum_running_memory=arguments.minimum_running_memory,
        maximum_running_swap_growth_mib=(
            arguments.maximum_running_swap_growth_mib
        ),
        minimum_running_swap_free_mib=arguments.minimum_running_swap_free_mib,
        maximum_running_rss_mib=arguments.maximum_running_rss_mib,
        grace_seconds=arguments.grace_seconds,
        delete_wave=arguments.delete_wave,
        sample_only=arguments.sample_only,
        quiet=arguments.quiet,
    )


def main() -> int:
    settings = parse_arguments()
    monitor = Monitor(settings)
    signal.signal(signal.SIGINT, monitor.forward_signal)
    signal.signal(signal.SIGTERM, monitor.forward_signal)
    status = 4
    try:
        status = monitor.run()
        return status
    except Exception as error:
        monitor.emit(f"FATAL {type(error).__name__}: {error}")
        return 4
    finally:
        monitor.close(successful=status == 0)


if __name__ == "__main__":
    raise SystemExit(main())
