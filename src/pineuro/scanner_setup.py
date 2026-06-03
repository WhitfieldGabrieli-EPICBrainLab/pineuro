"""Scanner connection setup and testing for real-time fMRI neurofeedback.

Utilities for testing scanner network connectivity, validating the DICOM
watch directory, and generating platform-specific SMB share setup instructions.

Supported scanners (Siemens, Philips) push files to an SMB share on the
analysis computer. GE scanners lack a built-in push mechanism and are
not currently supported for real-time acquisition.
"""

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConnectionTestResult:
    """Result of a scanner connection test."""

    ping_ok: bool | None = None
    ping_latency_ms: float | None = None
    ping_detail: str = ""
    dir_exists: bool = False
    dir_writable: bool = False
    dir_empty: bool = True
    dir_file_count: int = 0
    dir_detail: str = ""

    @property
    def overall_ok(self) -> bool:
        """True if all critical checks passed (ping passed/skipped AND dir exists AND writable)."""
        ping_passed = self.ping_ok is not False
        return ping_passed and self.dir_exists and self.dir_writable


def ping_scanner(ip_address: str, timeout: float = 3.0) -> tuple[bool, float | None, str]:
    """Ping the scanner IP address to check network connectivity.

    Parameters
    ----------
    ip_address : str
        Scanner IP address (e.g., "192.168.2.1").
    timeout : float
        Ping timeout in seconds.

    Returns
    -------
    success : bool
    latency_ms : float or None
    detail : str
    """
    if not ip_address or not ip_address.strip():
        return False, None, "No scanner IP address configured"

    ip_address = ip_address.strip()

    if sys.platform == "win32":
        cmd = ["ping", "-n", "1", "-w", str(int(timeout * 1000)), ip_address]
    else:
        cmd = ["ping", "-c", "1", "-W", str(int(timeout)), ip_address]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        if result.returncode == 0:
            latency = _parse_ping_latency(result.stdout)
            if latency is not None:
                detail = f"Scanner at {ip_address} responded ({latency:.1f} ms)"
            else:
                detail = f"Scanner at {ip_address} responded"
            return True, latency, detail
        else:
            return False, None, f"Scanner at {ip_address} did not respond"
    except subprocess.TimeoutExpired:
        return False, None, f"Ping to {ip_address} timed out after {timeout}s"
    except FileNotFoundError:
        return False, None, "ping command not found"
    except Exception as e:
        return False, None, f"Ping error: {e}"


def _parse_ping_latency(output: str) -> float | None:
    """Parse ping latency from platform-specific output."""
    # Windows emits e.g. time=1ms or time<1ms; macOS/Linux emit e.g.
    # time=0.5 ms or time=1.23 ms.
    match = re.search(r"time[=<](\d+\.?\d*)\s*ms", output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def check_watch_directory(watch_dir: str | Path) -> tuple[bool, bool, bool, int, str]:
    """Check the DICOM watch directory status.

    Parameters
    ----------
    watch_dir : str or Path
        Path to the scanner DICOM watch directory.

    Returns
    -------
    exists : bool
    writable : bool
    is_empty : bool
    file_count : int
    detail : str
    """
    watch_dir = Path(watch_dir)

    if not watch_dir.exists():
        return False, False, True, 0, f"Directory does not exist: {watch_dir}"

    if not watch_dir.is_dir():
        return False, False, True, 0, f"Path is not a directory: {watch_dir}"

    writable = os.access(watch_dir, os.W_OK)

    try:
        files = [f for f in watch_dir.iterdir() if f.is_file()]
        file_count = len(files)
    except PermissionError:
        return True, False, True, 0, f"Cannot list directory (permission denied): {watch_dir}"

    is_empty = file_count == 0

    # Check for subfolders (Siemens scanners create per-sequence subfolders)
    try:
        subdirs = [d for d in watch_dir.iterdir() if d.is_dir()]
    except PermissionError:
        subdirs = []

    parts = [f"Directory exists: {watch_dir}"]
    if writable:
        parts.append("Writable: yes")
    else:
        parts.append("Writable: NO (check permissions)")
    if is_empty and not subdirs:
        parts.append("Empty: yes (ready for new session)")
    elif subdirs:
        parts.append(f"{len(subdirs)} subfolder(s) from previous sequences")
        if file_count > 0:
            parts.append(f"Also contains {file_count} top-level file(s)")
    else:
        parts.append(f"Contains {file_count} file(s) (may need clearing)")

    return True, writable, is_empty, file_count, "; ".join(parts)


def test_scanner_connection(
    scanner_ip: str | None,
    watch_dir: str | Path,
    ping_timeout: float = 3.0,
) -> ConnectionTestResult:
    """Run full scanner connection test.

    Parameters
    ----------
    scanner_ip : str or None
        Scanner IP address. If None or empty, ping test is skipped.
    watch_dir : str or Path
        Path to DICOM watch directory.
    ping_timeout : float
        Ping timeout in seconds.

    Returns
    -------
    ConnectionTestResult
    """
    result = ConnectionTestResult()

    if scanner_ip and scanner_ip.strip():
        result.ping_ok, result.ping_latency_ms, result.ping_detail = ping_scanner(scanner_ip, timeout=ping_timeout)
    else:
        result.ping_ok = None
        result.ping_detail = "Scanner IP not configured (ping skipped)"

    (
        result.dir_exists,
        result.dir_writable,
        result.dir_empty,
        result.dir_file_count,
        result.dir_detail,
    ) = check_watch_directory(watch_dir)

    return result


test_scanner_connection.__test__ = False  # type: ignore[attr-defined]  # prevent pytest collection


def get_setup_instructions(watch_dir: str, scanner_ip: str = "") -> dict[str, str]:
    """Generate platform-specific SMB share setup instructions.

    Parameters
    ----------
    watch_dir : str
        DICOM watch directory path (e.g., "/BOLD_DICOM" or "C:\\BOLD_DICOM").
    scanner_ip : str
        Scanner IP address for reference in instructions.

    Returns
    -------
    dict
        Keys: "windows", "macos", "linux", "scanner_notes"
    """
    share_name = Path(watch_dir).name or "BOLD_DICOM"

    instructions = {}

    instructions["scanner_notes"] = (
        "Siemens BOLD Dot Add-In Configurator\n"
        "====================================\n"
        "Open via Dot Cockpit > protocol program.\n\n"
        "Configuration fields:\n"
        f"  Enable Auto Export:  checked\n"
        f"  Target Host:         \\\\<analysis-computer-ip>\n"
        f"  Target Directory:    {share_name}\n"
        f"  User Name:           WORKGROUP\\<username>\n"
        f"  Password:            <password>\n\n"
        "IMPORTANT — run pineuro as the same user:\n"
        "  The scanner writes each DICOM to the share owned by the\n"
        "  User Name configured above, granting read/write/execute to\n"
        "  that account only. pineuro must therefore be launched while\n"
        "  logged in as that same dedicated user (on every platform).\n"
        "  If it runs under any other account it cannot read the\n"
        "  incoming DICOMs.\n\n"
        "macOS only — additional scanner-side step:\n"
        "  Before running BOLD sequences, open a Command Prompt\n"
        "  on the scanner console (Ctrl+Esc > Advanced User >\n"
        "  Command Prompt) and run:\n"
        f"    net use Z: \\\\<analysis-computer-ip>\\{share_name}"
        " /user:<username> <password>\n"
        "  Run this once per scanner session. To disconnect:\n"
        "    net use Z: /delete\n"
        "  This step is NOT needed for Windows or Linux.\n\n"
        f"Philips: Scanner export destination > same share path\n"
        f"GE: Not supported (no built-in push-to-share)\n\n"
        "Requirements:\n"
        f"- Wired Ethernet between scanner and analysis computer\n"
        f"- Scanner IP: {scanner_ip or '<check your site>'}\n\n"
        "Note: The examples below use 192.168.2.x as a sample subnet\n"
        "(scanner at .1, analysis computer at .5). Your site will have\n"
        "its own subnet — ask your scanner IT for the correct addresses."
    )

    instructions["windows"] = (
        f"# 1. Create a dedicated local user account\n"
        f"# The scanner cannot authenticate against Microsoft or\n"
        f"# domain accounts on the isolated scanner network.\n"
        f"# Settings > Accounts > Other users > Add account\n"
        f"#   > 'I don't have this person's sign-in information'\n"
        f"#   > 'Add a user without a Microsoft account'\n"
        f"#   > Username: <username>, Password: <password>\n\n"
        f"# 2. Create and share the DICOM folder\n"
        f"# Create the folder: {watch_dir}\n"
        f"# Right-click the folder > Properties > Sharing tab\n"
        f"#   > Advanced Sharing... > check 'Share this folder'\n"
        f"#   > Share name: {share_name}\n"
        f"#   > Permissions > Add > Everyone > Full Control > OK\n\n"
        f"# 3. Set a static IP on the scanner network interface\n"
        f"# Settings > Network & Internet > Ethernet > Edit\n"
        f"#   > IP assignment: Manual > toggle IPv4 on\n"
        f"#   > IP address: 192.168.2.5, Subnet mask: 255.255.255.0\n"
        f"# (Replace 192.168.2.5 with the IP assigned to this computer\n"
        f"# on your scanner network — ask scanner IT)\n\n"
        f"# 4. Firewall\n"
        f"# Settings > search 'Windows Defender Firewall'\n"
        f"#   > Allow an app through firewall\n"
        f"#   > check 'File and Printer Sharing' for Private and Public\n"
        f"# Or turn off the firewall entirely."
    )

    instructions["macos"] = (
        f"# 1. Create a dedicated local user account\n"
        f"# The scanner cannot authenticate against AD/domain accounts\n"
        f"# on the isolated scanner network. Create a local account:\n"
        f"# System Settings > Users & Groups > Add User...\n"
        f"#   New Account: Standard (not Admin)\n"
        f"#   Full Name / Account Name: <username>\n"
        f"#   Password: <password>\n"
        f"# You MUST log into this account and run pineuro from it —\n"
        f"# the scanner writes DICOMs readable only by this user.\n\n"
        f"# 2. Enable SMB file sharing\n"
        f"# System Settings > General > Sharing\n"
        f"#   a. Turn on File Sharing (green toggle)\n"
        f"#   b. Click the (i) button next to File Sharing\n"
        f"#   c. Click Options...\n"
        f"#   d. Ensure 'Share files and folders using SMB' is checked\n"
        f"#   e. Check the box next to <username> in the user list\n"
        f"#      (you will be prompted to enter <username>'s password)\n"
        f"#   f. Click Done\n\n"
        f"# 3. Set up the shared folder\n"
        f"mkdir -p {watch_dir}\n"
        f"chmod 777 {watch_dir}\n"
        f"# In the File Sharing info panel:\n"
        f"#   a. Under Shared Folders, click + and add {watch_dir}\n"
        f"#   b. Under Users, click + and add <username>\n"
        f"#   c. Set permissions for <username> to Read & Write\n\n"
        f"# 4. Set a static IP on the scanner network interface\n"
        f"# System Settings > Network > Ethernet > Details > TCP/IP\n"
        f"#   Configure IPv4: Manually\n"
        f"#   IP: 192.168.2.5, Subnet: 255.255.255.0\n"
        f"# (Replace 192.168.2.5 with the IP assigned to this computer\n"
        f"# on your scanner network — ask scanner IT)\n\n"
        f"# 5. Firewall\n"
        f"# System Settings > Network > Firewall\n"
        f"# Turn the firewall OFF, or ensure file sharing is allowed\n\n"
        f"# 6. Verify the share works locally\n"
        f"smbutil view //<username>@127.0.0.1\n"
        f"# Enter <username>'s password when prompted.\n"
        f"# You should see {share_name} in the share list."
    )

    instructions["linux"] = (
        f"# 1. Install Samba\n"
        f"sudo apt-get install --upgrade samba\n\n"
        f"# 2. Create a Samba password for the user\n"
        f"sudo smbpasswd -a <username>\n"
        f"# Three prompts: (1) sudo password, (2) samba password,\n"
        f"# (3) confirm samba password.\n\n"
        f"# 3. Create the DICOM watch directory\n"
        f"sudo mkdir -p {watch_dir}\n"
        f"sudo chown <username> {watch_dir}\n"
        f"sudo chmod -R 773 {watch_dir}\n\n"
        f"# 4. Configure Samba\n"
        f"# Back up existing config, then edit /etc/samba/smb.conf:\n"
        f"sudo cp /etc/samba/smb.conf /etc/samba/smb.conf.backup\n\n"
        f"# Add to the end of /etc/samba/smb.conf:\n"
        f"[{share_name}]\n"
        f"   path = {watch_dir}\n"
        f"   browsable = yes\n"
        f"   writable = yes\n"
        f"   valid users = <username>\n"
        f"   comment = Samba Share for BOLD dot-addin dicoms\n\n"
        f"# 5. Set a static IP on the scanner network interface\n"
        f"sudo ifconfig <interface> 192.168.2.5\n"
        f"# Replace <interface> with your ethernet interface (run ifconfig)\n"
        f"# Replace 192.168.2.5 with the IP assigned to this computer\n"
        f"# on your scanner network — ask scanner IT\n\n"
        f"# 6. Start/restart Samba\n"
        f"sudo systemctl restart smbd nmbd\n\n"
        f"# 7. Verify the share is visible\n"
        f"smbclient -L localhost -U <username>\n"
        f"# Should show {share_name} as a Disk share"
    )

    return instructions


def get_current_platform_key() -> str:
    """Get the current platform key for instructions lookup.

    Returns
    -------
    str
        One of: "windows", "macos", "linux"
    """
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "macos"
    else:
        return "linux"
