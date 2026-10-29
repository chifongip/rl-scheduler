from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import pynvml

from models import GpuStatus

logger = logging.getLogger("hardware")

FAN_CURVE = ((0, 40), (50, 60), (65, 75), (75, 90), (80, 100))
FAN_HYSTERESIS_C = 5
FAN_COOLDOWN_SECONDS = 30
FAN_POLL_SECONDS = 5


@dataclass
class AutoFanState:
    band: int
    cooling_since: float | None = None


class GpuManager:
    def __init__(self, managed_gpu_ids: list[int] | None = None):
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if managed_gpu_ids is None:
            managed_gpu_ids = list(range(device_count))
        for gid in managed_gpu_ids:
            if gid >= device_count:
                raise ValueError(f"GPU {gid} not found (system has {device_count} GPUs)")
        # Build PCI Bus-Id map and sort by Bus-Id to match nvidia-smi order
        self.bus_id_map: dict[int, str] = {}
        nvml_bus_int: dict[int, int] = {}  # NVML index → bus number as int
        for gid in managed_gpu_ids:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gid)
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            bus_id = pci.busId if isinstance(pci.busId, str) else pci.busId.decode()
            self.bus_id_map[gid] = bus_id
            # Extract bus number from "00000000:47:00.0" → 0x47
            bus_hex = bus_id.split(":")[1]
            nvml_bus_int[gid] = int(bus_hex, 16)
        managed_gpu_ids = sorted(managed_gpu_ids, key=lambda gid: nvml_bus_int[gid])
        self.managed_gpu_ids = managed_gpu_ids

        # task_id -> gpu_id mapping for active jobs
        self.active_tasks: dict[str, int] = {}
        self.active_process_groups: dict[str, int] = {}
        self.disabled_gpu_ids: set[int] = set()
        self._lock = threading.RLock()
        # fan capability cache: gpu_id -> {num_fans, min_speed, max_speed} or None
        self._fan_support: dict[int, dict | None] = {}
        self._auto_fans: dict[int, AutoFanState] = {}
        self._pending_fan_restores: set[int] = set()
        self._fan_stop = threading.Event()
        self._fan_thread: threading.Thread | None = None
        logger.info("GpuManager initialized: managing GPUs %s (bus IDs: %s)",
                     self.managed_gpu_ids,
                     [self.bus_id_map[gid] for gid in self.managed_gpu_ids])

    def get_gpu_status(self, gpu_id: int) -> GpuStatus:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_total_mb = mem_info.total // (1024 * 1024)

        mem_used_mb = mem_info.used // (1024 * 1024)
        mem_pct = round(mem_info.used / mem_info.total * 100, 1) if mem_info.total > 0 else 0.0
        active_task_id = None
        with self._lock:
            for tid, gid in self.active_tasks.items():
                if gid == gpu_id:
                    active_task_id = tid
                    break
        external_count = self._count_external_compute_procs(gpu_id)

        # Fan info (None if unsupported)
        fan_speed = None
        fan_mode = None
        num_fans = None
        try:
            fan_info = self._probe_fan_support(gpu_id)
            if fan_info is not None:
                num_fans = fan_info["num_fans"]
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                fan_speed = pynvml.nvmlDeviceGetFanSpeed_v2(handle, 0)
                fan_mode = self.get_fan_status(gpu_id)["fan_mode"]
        except pynvml.NVMLError:
            pass

        return GpuStatus(
            gpu_id=gpu_id,
            name=name,
            temperature_c=temp,
            memory_used_mb=mem_used_mb,
            memory_total_mb=mem_total_mb,
            memory_utilization_pct=mem_pct,
            active_task_id=active_task_id,
            external_process_count=external_count,
            scheduling_enabled=self.is_scheduling_enabled(gpu_id),
            fan_speed_pct=fan_speed,
            fan_mode=fan_mode,
            num_fans=num_fans,
        )

    def get_all_gpu_status(self) -> list[GpuStatus]:
        statuses = []
        for gid in self.managed_gpu_ids:
            try:
                statuses.append(self.get_gpu_status(gid))
            except pynvml.NVMLError as e:
                logger.warning("NVML error reading GPU %d: %s", gid, e)
        return statuses

    def _count_external_compute_procs(self, gpu_id: int) -> int:
        """Count non-MPS compute processes on a GPU (processes not managed by this scheduler)."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            with self._lock:
                managed_groups = set(self.active_process_groups.values())
            count = 0
            for p in compute_procs:
                try:
                    if os.getpgid(p.pid) in managed_groups:
                        continue
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                try:
                    name = pynvml.nvmlSystemGetProcessName(p.pid)
                    if isinstance(name, bytes):
                        name = name.decode()
                    if "mps" not in name.lower():
                        count += 1
                except pynvml.NVMLError:
                    count += 1  # can't identify, assume it's real
            return count
        except pynvml.NVMLError:
            return 0

    def is_gpu_available(self, gpu_id: int) -> bool:
        if gpu_id not in self.managed_gpu_ids:
            return False
        with self._lock:
            if gpu_id in self.disabled_gpu_ids:
                logger.debug("GPU %d unavailable: scheduling is disabled", gpu_id)
                return False
            if gpu_id in self.active_tasks.values():
                logger.debug("GPU %d unavailable: has scheduler-managed task", gpu_id)
                return False
        ext = self._count_external_compute_procs(gpu_id)
        if ext:
            logger.info("GPU %d unavailable: %d compute process(es) running", gpu_id, ext)
            return False
        return True

    def find_available_gpu(self) -> int | None:
        for gid in self.managed_gpu_ids:
            if self.is_gpu_available(gid):
                return gid
        return None

    def get_available_gpu_ids(self) -> list[int]:
        """Return one availability snapshot for a scheduler dispatch cycle."""
        return [gid for gid in self.managed_gpu_ids if self.is_gpu_available(gid)]

    def is_scheduling_enabled(self, gpu_id: int) -> bool:
        with self._lock:
            return gpu_id in self.managed_gpu_ids and gpu_id not in self.disabled_gpu_ids

    def set_scheduling_enabled(self, gpu_id: int, enabled: bool) -> None:
        if gpu_id not in self.managed_gpu_ids:
            raise ValueError(f"GPU {gpu_id} is not managed")
        with self._lock:
            if enabled:
                self.disabled_gpu_ids.discard(gpu_id)
            else:
                self.disabled_gpu_ids.add(gpu_id)

    def register_task(self, task_id: str, gpu_id: int, process_group_id: int | None = None) -> None:
        with self._lock:
            self.active_tasks[task_id] = gpu_id
            if process_group_id is not None:
                self.active_process_groups[task_id] = process_group_id

    def unregister_task(self, task_id: str) -> None:
        with self._lock:
            self.active_tasks.pop(task_id, None)
            self.active_process_groups.pop(task_id, None)

    def _probe_fan_support(self, gpu_id: int) -> dict | None:
        if gpu_id in self._fan_support:
            return self._fan_support[gpu_id]
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            num_fans = pynvml.nvmlDeviceGetNumFans(handle)
            if num_fans < 1:
                self._fan_support[gpu_id] = None
                return None
            min_speed, max_speed = pynvml.nvmlDeviceGetMinMaxFanSpeed(handle)
            info = {"num_fans": num_fans, "min_speed": min_speed, "max_speed": max_speed}
            self._fan_support[gpu_id] = info
            return info
        except (pynvml.NVMLError, AttributeError, OSError):
            self._fan_support[gpu_id] = None
            return None

    def get_fan_status(self, gpu_id: int) -> dict:
        with self._lock:
            return self._get_fan_status(gpu_id)

    def _get_fan_status(self, gpu_id: int) -> dict:
        fan_info = self._probe_fan_support(gpu_id)
        if fan_info is None:
            return {"fan_supported": False}
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        try:
            speed = pynvml.nvmlDeviceGetFanSpeed_v2(handle, 0)
        except pynvml.NVMLError:
            speed = None
        try:
            policy = pynvml.nvmlDeviceGetFanControlPolicy_v2(handle, 0)
            mode = "manual" if policy == pynvml.NVML_FAN_POLICY_MANUAL else "auto"
        except pynvml.NVMLError:
            mode = "auto"
        if gpu_id in self._auto_fans:
            mode = "auto"
        return {
            "fan_supported": True,
            "fan_speed_pct": speed,
            "fan_mode": mode,
            "num_fans": fan_info["num_fans"],
            "min_speed": fan_info["min_speed"],
            "max_speed": fan_info["max_speed"],
        }

    def set_fan_speed(self, gpu_id: int, speed: int) -> dict:
        with self._lock:
            self._write_fan_speed(gpu_id, speed)
            self._auto_fans.pop(gpu_id, None)
            self._pending_fan_restores.discard(gpu_id)
            return self.get_fan_status(gpu_id)

    def _write_fan_speed(self, gpu_id: int, speed: int) -> None:
        fan_info = self._probe_fan_support(gpu_id)
        if fan_info is None:
            raise ValueError(f"GPU {gpu_id} does not support fan control")
        if speed < fan_info["min_speed"] or speed > fan_info["max_speed"]:
            raise ValueError(f"Fan speed {speed} out of range [{fan_info['min_speed']}, {fan_info['max_speed']}]")
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        try:
            for fan in range(fan_info["num_fans"]):
                pynvml.nvmlDeviceSetFanControlPolicy(handle, fan, pynvml.NVML_FAN_POLICY_MANUAL)
                pynvml.nvmlDeviceSetFanSpeed_v2(handle, fan, speed)
        except pynvml.NVMLError as e:
            self._fallback_fan_auto(gpu_id)
            raise ValueError(f"Failed to set fan speed on GPU {gpu_id}: {e}")
        logger.info("GPU %d fan speed set to %d%%", gpu_id, speed)

    def set_fan_auto(self, gpu_id: int) -> dict:
        with self._lock:
            fan_info = self._probe_fan_support(gpu_id)
            if fan_info is None:
                raise ValueError(f"GPU {gpu_id} does not support fan control")
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError as e:
                raise ValueError(f"Failed to read GPU {gpu_id} temperature: {e}")
            band = self._fan_band(temperature)
            self._write_auto_fan_speed(gpu_id, band)
            self._pending_fan_restores.discard(gpu_id)
            self._auto_fans[gpu_id] = AutoFanState(band)
            self._start_fan_thread()
            return self.get_fan_status(gpu_id)

    @staticmethod
    def _fan_band(temperature: int) -> int:
        return max((i for i, (threshold, _) in enumerate(FAN_CURVE) if temperature >= threshold), default=0)

    def _write_auto_fan_speed(self, gpu_id: int, band: int) -> None:
        info = self._probe_fan_support(gpu_id)
        speed = max(info["min_speed"], min(info["max_speed"], FAN_CURVE[band][1]))
        self._write_fan_speed(gpu_id, speed)

    def _update_auto_fans(self) -> None:
        # Serialize mode changes and hardware writes so a stale tick cannot undo Manual.
        with self._lock:
            for gpu_id in list(self._pending_fan_restores):
                if self._restore_fan_auto(gpu_id):
                    self._pending_fan_restores.discard(gpu_id)
            for gpu_id, state in list(self._auto_fans.items()):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    target = self._fan_band(temperature)
                    now = time.monotonic()
                    if target > state.band:
                        self._write_auto_fan_speed(gpu_id, target)
                        state.band = target
                        state.cooling_since = None
                    elif state.band > 0 and temperature <= FAN_CURVE[state.band][0] - FAN_HYSTERESIS_C:
                        if state.cooling_since is None:
                            state.cooling_since = now
                        elif now - state.cooling_since >= FAN_COOLDOWN_SECONDS:
                            # Each downward step must earn its own cooling delay.
                            target = state.band - 1
                            self._write_auto_fan_speed(gpu_id, target)
                            state.band = target
                            state.cooling_since = None
                    else:
                        state.cooling_since = None
                except (pynvml.NVMLError, ValueError) as e:
                    logger.warning("Auto fan control failed on GPU %d; restoring driver control: %s", gpu_id, e)
                    if gpu_id in self._auto_fans:
                        self._fallback_fan_auto(gpu_id)

    def _start_fan_thread(self) -> None:
        if self._fan_thread is None:
            self._fan_thread = threading.Thread(target=self._fan_loop, name="gpu-fans", daemon=True)
            self._fan_thread.start()

    def _fallback_fan_auto(self, gpu_id: int) -> None:
        self._auto_fans.pop(gpu_id, None)
        self._pending_fan_restores.add(gpu_id)
        if self._restore_fan_auto(gpu_id):
            self._pending_fan_restores.discard(gpu_id)
        else:
            # Initial Auto/Manual writes may fail before a worker exists.
            self._start_fan_thread()

    def _fan_loop(self) -> None:
        while not self._fan_stop.wait(FAN_POLL_SECONDS):
            self._update_auto_fans()

    def _restore_fan_auto(self, gpu_id: int) -> bool:
        fan_info = self._probe_fan_support(gpu_id)
        if fan_info is None:
            return False
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        except pynvml.NVMLError as e:
            logger.warning("Could not restore driver fan control on GPU %d: %s", gpu_id, e)
            return False
        restored = True
        for fan in range(fan_info["num_fans"]):
            try:
                pynvml.nvmlDeviceSetDefaultFanSpeed_v2(handle, fan)
            except pynvml.NVMLError as e:
                restored = False
                logger.warning("Could not reset fan %d on GPU %d: %s", fan, gpu_id, e)
            try:
                pynvml.nvmlDeviceSetFanControlPolicy(handle, fan, pynvml.NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW)
            except pynvml.NVMLError as e:
                restored = False
                logger.warning("Could not restore fan %d policy on GPU %d: %s", fan, gpu_id, e)
        return restored

    def shutdown(self) -> None:
        self._fan_stop.set()
        if self._fan_thread is not None:
            self._fan_thread.join()
        try:
            self._auto_fans.clear()
            for gid in self.managed_gpu_ids:
                fan_info = self._probe_fan_support(gid)
                if fan_info is not None:
                    try:
                        if self._restore_fan_auto(gid):
                            self._pending_fan_restores.discard(gid)
                            logger.info("GPU %d fan reset to automatic on shutdown", gid)
                        else:
                            logger.warning("Could not fully restore driver fan control on GPU %d during shutdown", gid)
                    except (pynvml.NVMLError, OSError) as e:
                        logger.warning("Could not reset fan on GPU %d during shutdown: %s", gid, e)
        finally:
            pynvml.nvmlShutdown()
            logger.info("GpuManager shut down")
