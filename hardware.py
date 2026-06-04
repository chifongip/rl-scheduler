from __future__ import annotations

import logging

import pynvml

from models import GpuStatus

logger = logging.getLogger("hardware")


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
        # fan capability cache: gpu_id -> {num_fans, min_speed, max_speed} or None
        self._fan_support: dict[int, dict | None] = {}
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

        # Sum per-process memory to match nvidia-smi's "Used" (excludes driver overhead)
        try:
            compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            proc_mem = sum(p.usedGpuMemory or 0 for p in compute_procs + graphics_procs)
        except pynvml.NVMLError:
            proc_mem = mem_info.used  # fallback to v1 if process query fails

        mem_used_mb = proc_mem // (1024 * 1024)
        mem_pct = round(proc_mem / mem_info.total * 100, 1) if mem_info.total > 0 else 0.0
        active_task_id = None
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
                try:
                    policy = pynvml.nvmlDeviceGetFanControlPolicy_v2(handle, 0)
                    fan_mode = "manual" if policy == pynvml.NVML_FAN_POLICY_MANUAL else "auto"
                except pynvml.NVMLError:
                    fan_mode = "auto"
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
            count = 0
            for p in compute_procs:
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

    def register_task(self, task_id: str, gpu_id: int) -> None:
        self.active_tasks[task_id] = gpu_id

    def unregister_task(self, task_id: str) -> None:
        self.active_tasks.pop(task_id, None)

    def _probe_fan_support(self, gpu_id: int) -> dict | None:
        if gpu_id in self._fan_support:
            return self._fan_support[gpu_id]
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            num_fans = pynvml.nvmlDeviceGetNumFans(handle)
            min_speed, max_speed = pynvml.nvmlDeviceGetMinMaxFanSpeed(handle)
            info = {"num_fans": num_fans, "min_speed": min_speed, "max_speed": max_speed}
            self._fan_support[gpu_id] = info
            return info
        except pynvml.NVMLError:
            self._fan_support[gpu_id] = None
            return None

    def get_fan_status(self, gpu_id: int) -> dict:
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
        return {
            "fan_supported": True,
            "fan_speed_pct": speed,
            "fan_mode": mode,
            "num_fans": fan_info["num_fans"],
            "min_speed": fan_info["min_speed"],
            "max_speed": fan_info["max_speed"],
        }

    def set_fan_speed(self, gpu_id: int, speed: int) -> dict:
        fan_info = self._probe_fan_support(gpu_id)
        if fan_info is None:
            raise ValueError(f"GPU {gpu_id} does not support fan control")
        if speed < fan_info["min_speed"] or speed > fan_info["max_speed"]:
            raise ValueError(f"Fan speed {speed} out of range [{fan_info['min_speed']}, {fan_info['max_speed']}]")
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        # Set manual mode first
        try:
            pynvml.nvmlDeviceSetFanControlPolicy(handle, 0, pynvml.NVML_FAN_POLICY_MANUAL)
        except pynvml.NVMLError as e:
            logger.warning("Could not set fan policy to manual on GPU %d: %s", gpu_id, e)
        pynvml.nvmlDeviceSetFanSpeed_v2(handle, 0, speed)
        logger.info("GPU %d fan speed set to %d%%", gpu_id, speed)
        return self.get_fan_status(gpu_id)

    def set_fan_auto(self, gpu_id: int) -> dict:
        fan_info = self._probe_fan_support(gpu_id)
        if fan_info is None:
            raise ValueError(f"GPU {gpu_id} does not support fan control")
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        try:
            pynvml.nvmlDeviceSetDefaultFanSpeed_v2(handle, 0)
        except pynvml.NVMLError as e:
            logger.warning("Could not reset fan speed on GPU %d: %s", gpu_id, e)
        try:
            pynvml.nvmlDeviceSetFanControlPolicy(handle, 0, pynvml.NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW)
        except pynvml.NVMLError as e:
            logger.warning("Could not restore fan policy on GPU %d: %s", gpu_id, e)
        logger.info("GPU %d fan reset to automatic", gpu_id)
        return self.get_fan_status(gpu_id)

    def shutdown(self) -> None:
        # Reset all GPUs to automatic fan control before shutting down
        for gid in self.managed_gpu_ids:
            fan_info = self._probe_fan_support(gid)
            if fan_info is not None:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gid)
                    pynvml.nvmlDeviceSetDefaultFanSpeed_v2(handle, 0)
                    pynvml.nvmlDeviceSetFanControlPolicy(handle, 0, pynvml.NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW)
                    logger.info("GPU %d fan reset to automatic on shutdown", gid)
                except pynvml.NVMLError as e:
                    logger.warning("Could not reset fan on GPU %d during shutdown: %s", gid, e)
        pynvml.nvmlShutdown()
        logger.info("GpuManager shut down")
