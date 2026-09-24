"""Read process and host resource counters without evaluating model work."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import json
import os
import platform
import subprocess
import sys

from moespresso.runtime.diagnostic_environment import diagnostic_tool_environment


def host_snapshot():
    """Read host state without creating a Metal device in the supervisor."""
    import psutil

    vm, swap, battery = psutil.virtual_memory(), psutil.swap_memory(), psutil.sensors_battery()
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "memory_total_bytes": vm.total, "memory_available_bytes": vm.available,
        "swap_used_bytes": swap.used, "paging_in_bytes": swap.sin,
        "paging_out_bytes": swap.sout,
        "paging_counter_kind": "vm_pageins_pageouts" if sys.platform == "darwin" else "psutil_sin_sout",
        "power": None if battery is None else {
            "percent": battery.percent, "plugged": battery.power_plugged,
        },
        "load_average": list(os.getloadavg()),
    }


def gpu_inventory():
    """Read GPU core counts without creating a Metal device."""
    if sys.platform != "darwin":
        return None
    try:
        proc = subprocess.run(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, timeout=3, check=True, env=diagnostic_tool_environment(),
        )
        inventory = []
        for row in json.loads(proc.stdout).get("SPDisplaysDataType", []):
            try:
                core_count = int(row["sppci_cores"])
            except (KeyError, TypeError, ValueError):
                inventory.append({})
            else:
                inventory.append({"core_count": core_count})
        return inventory
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class _RusageV2(ctypes.Structure):
    """Darwin sys/resource.h rusage_info_v2 ABI."""

    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
            "pageins", "wired_size", "resident_size", "phys_footprint",
            "proc_start_abstime", "proc_exit_abstime", "child_user_time",
            "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
            "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
        )
    ]


@lru_cache(maxsize=1)
def _reader():
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


def process_resources():
    """Return Darwin process counters, or None when the OS reader is unavailable.

    Disk counters cover all IO charged to this process. They are not isolated
    expert reads, swap traffic, device bandwidth or time spent waiting for IO.
    """
    if sys.platform != "darwin":
        return None
    try:
        value = _RusageV2()
        if _reader()(os.getpid(), 2, ctypes.byref(value)):
            return None
        return {
            "counter_kind": "darwin_proc_pid_rusage_v2",
            "disk_read_bytes": int(value.diskio_bytesread),
            "disk_write_bytes": int(value.diskio_byteswritten),
            "physical_footprint_bytes": int(value.phys_footprint),
        }
    except (OSError, AttributeError):
        return None
