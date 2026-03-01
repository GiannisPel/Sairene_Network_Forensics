import platform
import shutil
import subprocess
from datetime import datetime
from colorama import Back,Fore, Style, init
import psutil
import random

init(autoreset=True)

COLOR_POOL = [
    Fore.CYAN,
    Fore.MAGENTA,
    Fore.BLUE,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.RED
]

BLOCK_BACKS = [
    Back.BLACK,
    Back.RED,
    Back.GREEN,
    Back.YELLOW,
    Back.BLUE,
    Back.MAGENTA,
    Back.CYAN,
    Back.WHITE,
]

def rand_color():
    return random.choice(COLOR_POOL)

def c_rand(line_or_label: str, value: str) -> str:
    """
    If value is provided -> color the label only:  c_rand("OS:", "Windows 11")
    If value is None     -> color the whole line: c_rand("OS: Windows 11")
    """
    color = rand_color()
    if value is None:
        return f"{color}{line_or_label}{Style.RESET_ALL}"
    return f"{color}{line_or_label}{Style.RESET_ALL} {value}"

def make_color_blocks(blocks=None, block_width: int = 4, gap: str = "") -> str:
    if blocks is None:
        blocks = BLOCK_BACKS
    unit = " " * block_width
    return gap.join(f"{b}{unit}{Style.RESET_ALL}" for b in blocks)

def _bytes_to_human(n: int) -> str:
    #Human-readable formatter
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} B"


def get_disk_usage(path: str = "/") -> dict:
    #Windows accepts drive roots like "C:\\"
    total, used, free = shutil.disk_usage(path)
    return {
        "path": path,
        "total": _bytes_to_human(total),
        "used": _bytes_to_human(used),
        "free": _bytes_to_human(free),
        "percent_used": round((used / total) * 100, 1) if total else 0.0,
    }


def get_ram_usage() -> dict:
    vm = psutil.virtual_memory()
    return {
        "total": _bytes_to_human(vm.total),
        "used": _bytes_to_human(vm.used),
        "available": _bytes_to_human(vm.available),
        "percent_used": vm.percent,
    }


def get_os_info() -> dict:
    return {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def get_gpu_info_best_effort() -> str:
    """
    Best-effort GPU info:
    - If NVIDIA drivers/tools are present: nvidia-smi
    - Else: WMIC (older but often works)
    - Else: 'Unknown'
    """
    # 1) NVIDIA
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        out = (p.stdout or "").strip()
        if out:
            #Could be multiple GPUs
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            return " | ".join(lines)
    except Exception:
        pass

    # 2)WMIC (works on many Windows setups)
    try:
        p = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:list"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        out = (p.stdout or "").strip()
        if out:
            #Parse blocks separated by blank lines
            blocks = [b.strip() for b in out.split("\n\n") if b.strip()]
            gpus = []
            for b in blocks:
                name = None
                ram = None
                for line in b.splitlines():
                    if line.startswith("Name="):
                        name = line.split("=", 1)[1].strip()
                    elif line.startswith("AdapterRAM="):
                        val = line.split("=", 1)[1].strip()
                        if val.isdigit():
                            ram = _bytes_to_human(int(val))
                if name:
                    gpus.append(f"{name}" + (f" ({ram})" if ram else ""))
            if gpus:
                return " | ".join(gpus)
    except Exception:
        pass

    return "Unknown"


def format_neofetch(
    chat_model: str,
    disk_path: str,
    memory_db_size: str | None = None,
    wikipedia_size: str | None = None,
) -> str:
    os_info = get_os_info()
    disk = get_disk_usage(disk_path)
    ram = get_ram_usage()
    gpu = get_gpu_info_best_effort()

    lines = []

    lines.append(c_rand("Time:", datetime.now().isoformat(timespec="seconds")))
    lines.append(c_rand("OS:", f"{os_info['os']} {os_info['release']} ({os_info['machine']})"))
    lines.append(c_rand("OS Version:", os_info["version"]))
    lines.append(c_rand("Python:", os_info["python"]))

    lines.append("")
    
    lines.append(
        c_rand(
            f"Disk ({disk['path']}):",
            f"{disk['used']} / {disk['total']} used ({disk['percent_used']}%), Free: {disk['free']}"
        )
    )
    
    lines.append(
        c_rand(
            "RAM:",
            f"{ram['used']} / {ram['total']} used ({ram['percent_used']}%), Available: {ram['available']}"
        )
    )
    
    lines.append(c_rand("GPU:", gpu))
    lines.append(c_rand("CHAT_MODEL:", chat_model))
    
    if memory_db_size is not None:
        lines.append(c_rand("Memory DB size:", memory_db_size))
    
    if wikipedia_size is not None:
        lines.append(c_rand("Wikipedia corpus size:", wikipedia_size))
    
    lines.append("")  #spacing
    lines.append(make_color_blocks(block_width=4))
    lines.append(make_color_blocks(block_width=4))

    return "\n".join(lines)