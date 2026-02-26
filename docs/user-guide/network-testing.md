# Network Testing with Two Machines

Test the full scanner-to-analysis-machine communication path before your pilot by using two laptops connected via ethernet — one running the simulator as a stand-in for the scanner, the other running the real pineuro pipeline.

This replicates the exact same path a real Siemens scanner uses: files pushed over SMB to a shared directory on the analysis machine, detected by pineuro's file watcher.

## What You Need

- **Analysis machine** (Windows or macOS) — the laptop that will run pineuro at the scanner
- **Sender machine** (any platform) — stands in for the scanner, runs the simulator
- An ethernet cable connecting the two machines directly (or both on the same network)
- Pre-collected DICOMs from a previous session (e.g., Rockland_NFB data, or DICOMs from a previous MURFI session)
- pineuro installed on both machines (`pip install -e .`)

## Step 1: Connect the Machines

Plug an ethernet cable directly between the two laptops. Assign static IPs on the ethernet interface so they can find each other. Use the same subnet convention your scanner network uses (commonly `192.168.2.x`).

### Windows (Settings UI)

1. Settings > Network & internet > Ethernet
2. Click "Edit" next to "IP assignment"
3. Switch to Manual > toggle IPv4 on
4. IP address: `192.168.2.5`, Subnet mask: `255.255.255.0`

### macOS (System Settings)

1. System Settings > Network > Ethernet (or USB Ethernet adapter) > Details > TCP/IP
2. Configure IPv4: Manually
3. IP: `192.168.2.5`, Subnet: `255.255.255.0`

Use `192.168.2.5` for the analysis machine and `192.168.2.1` for the sender (or any two addresses on the same subnet). Verify connectivity:

```bash
# From the sender machine, ping the analysis machine
ping 192.168.2.5
```

## Step 2: Set Up the SMB Share on the Analysis Machine

Create a shared directory where the simulator will push files — the same directory pineuro will watch.

### Windows (PowerShell as Admin)

```powershell
# Create the directory
New-Item -ItemType Directory -Force -Path "C:\BOLD_DICOM"

# Share it
New-SmbShare -Name "BOLD_DICOM" -Path "C:\BOLD_DICOM" -FullAccess "Everyone"

# Verify
Get-SmbShare | Where-Object Name -eq 'BOLD_DICOM'
```

Ensure Windows Firewall allows SMB (port 445):
- Settings > search "Firewall & network protection" > Allow an app through firewall
- Check "Private" for "File and Printer Sharing"

**Important:** Your Windows user account must have a password set. Accounts with blank passwords cannot authenticate over SMB.

### macOS

```bash
sudo mkdir -p /BOLD_DICOM
sudo chown $USER /BOLD_DICOM
sudo chmod -R 773 /BOLD_DICOM
```

Then enable SMB sharing:
1. System Settings > General > Sharing
2. Turn on File Sharing
3. Click (i) next to File Sharing
4. Click + under Shared Folders, add `/BOLD_DICOM`
5. Set permissions for your user: Read & Write
6. Click Options... > ensure "Share files using SMB" is checked
7. Check the box next to your username in the SMB user list

## Step 3: Mount the Share on the Sender Machine

From the sender machine, mount the analysis machine's share so the simulator can write to it.

### Windows Sender

```powershell
net use Z: \\192.168.2.5\BOLD_DICOM /user:192.168.2.5\<username> <password>
```

### macOS Sender

```bash
mkdir -p /Volumes/BOLD_DICOM
mount_smbfs //username:password@192.168.2.5/BOLD_DICOM /Volumes/BOLD_DICOM
```

### Linux Sender

```bash
sudo mkdir -p /mnt/BOLD_DICOM
sudo mount -t cifs //192.168.2.5/BOLD_DICOM /mnt/BOLD_DICOM -o username=<username>,password=<password>
```

Verify the mount by creating a test file:

```bash
echo "test" > /path/to/mounted/share/test.txt
# Check that it appears on the analysis machine in C:\BOLD_DICOM (or /BOLD_DICOM)
```

Delete the test file before proceeding.

## Step 4: Run the Test

Open a terminal on each machine.

### Analysis machine — start pineuro

Run a normal pineuro session (via `pineuro-gui` or programmatically) with the watch directory set to your SMB share path (`C:\BOLD_DICOM` on Windows, `/BOLD_DICOM` on macOS).

Alternatively, for a quicker check, use the simulator integration built into `collect_volumes` but point the watch directory at the share — then the sender machine writes to it externally.

### Sender machine — start the simulator

```bash
python -m pineuro.simulator \
    --source /path/to/your/dicoms \
    --target Z:\ \
    --speed 1.0 \
    --n 50
```

Replace `Z:\` with your mounted share path (`/Volumes/BOLD_DICOM` on macOS, `/mnt/BOLD_DICOM` on Linux).

Use `--speed 1.0` for real-time pacing (recommended for the first test). Use `--speed 10.0` for a faster dry run once you've confirmed it works.

## Step 5: What to Look For

In the pineuro logs on the analysis machine:

- **Volumes detected**: `Collected X/N volumes` messages appearing at TR intervals
- **No timeouts**: The collection should not time out waiting for files
- **Feedback computed**: If running a feedback session, activation values should appear in the log
- **Timing**: Volume processing should complete well within TR (check `proc:XXms` in log lines)

If volumes aren't being detected, check:
- Can you manually see files appearing in the share directory on the analysis machine?
- Is the firewall allowing SMB traffic (port 445)?
- Are the mount credentials correct?
- Try creating a file on the sender's mounted share and see if it appears on the analysis machine

## Testing Without Two Machines

For a quick sanity check on a single machine, run the simulator and pineuro in separate terminals targeting the same local directory:

```bash
# Terminal 1 — pineuro watches this directory
# (configure watch_dir to C:\BOLD_DICOM_TEST or /tmp/BOLD_DICOM_TEST)

# Terminal 2 — simulator writes to the same directory
python -m pineuro.simulator --source /path/to/dicoms --target C:\BOLD_DICOM_TEST --speed 1.0
```

This tests the file-watching pipeline but not the network path.

## Platform Notes

- **Windows**: File system events (`ReadDirectoryChangesW`) work reliably for local directories being written to via SMB. This is the most thoroughly tested path.
- **macOS**: File system events (`FSEvents`) work for local directories. pineuro also has a polling fallback (every 0.5s) that catches any missed events.
- **Linux**: `inotify` does not fire for files arriving over network mounts. pineuro's polling fallback handles this, but detection latency will be up to 0.5s instead of near-instant. Since the analysis machine hosts the share locally, this is only relevant if your analysis machine is Linux and you're watching a remote mount (unusual setup).
