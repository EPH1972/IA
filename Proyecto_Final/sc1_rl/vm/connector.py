"""
SEC 2 — Conector de VM
Gestiona el ciclo de vida de la máquina virtual vía VBoxManage CLI.
"""
import subprocess
import time


class VMConnector:
    """Controla el estado (encendido/apagado) de la VM VirtualBox."""

    VBOXMANAGE = "VBoxManage"

    def __init__(self, vm_name: str):
        self.vm_name = vm_name

    # ── Interno ───────────────────────────────────────────────────────────────

    def _run(self, *args: str, timeout: float = 30.0) -> tuple[int, str, str]:
        try:
            r = subprocess.run(
                [self.VBOXMANAGE, *args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "timeout"
        except FileNotFoundError:
            return -2, "", "VBoxManage no encontrado en PATH"

    # ── Estado ────────────────────────────────────────────────────────────────

    def get_state(self) -> str:
        """Devuelve el estado de la VM: 'running', 'poweroff', 'saved', 'unknown'."""
        code, out, _ = self._run("showvminfo", self.vm_name, "--machinereadable")
        if code != 0:
            return "unknown"
        for line in out.splitlines():
            if line.startswith("VMState="):
                return line.split("=", 1)[1].strip('"')
        return "unknown"

    def is_running(self) -> bool:
        return self.get_state() == "running"

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self, headless: bool = False) -> bool:
        """Arranca la VM. Devuelve True si el comando tuvo éxito."""
        mode = "headless" if headless else "gui"
        code, _, _ = self._run("startvm", self.vm_name, "--type", mode, timeout=60.0)
        return code == 0

    def stop(self, force: bool = False) -> bool:
        """Apaga la VM. force=True corta el suministro; False envía ACPI power."""
        cmd = "poweroff" if force else "acpipowerbutton"
        code, _, _ = self._run("controlvm", self.vm_name, cmd)
        return code == 0

    def wait_until_running(
        self, timeout: float = 120.0, poll_interval: float = 5.0
    ) -> bool:
        """Bloquea hasta que la VM esté en estado 'running' o expire el tiempo."""
        elapsed = 0.0
        while elapsed < timeout:
            if self.is_running():
                return True
            time.sleep(poll_interval)
            elapsed += poll_interval
        return False
