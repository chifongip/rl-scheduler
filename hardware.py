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
        return GpuStatus(
            gpu_id=gpu_id,
            name=name,
            temperature_c=temp,
            memory_used_mb=mem_used_mb,
            memory_total_mb=mem_total_mb,
            memory_utilization_pct=mem_pct,
            active_task_id=active_task_id,
            external_process_count=external_count,
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

    def shutdown(self) -> None:
        pynvml.nvmlShutdown()
        logger.info("GpuManager shut down")
