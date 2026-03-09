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

Create a shared directory where the simulator will push files — the same directory pineuro will watch. The scanner cannot authenticate against Microsoft/Azure AD/domain accounts on the isolated scanner network, so each platform needs a dedicated local user account.

### Windows

1. **Create a dedicated local user**: Settings > Accounts > Other users > Add account > "I don't have this person's sign-in information" > "Add a user without a Microsoft account" > set username and password
2. **Create the folder**: e.g., `C:\BOLD_DICOM`
3. **Share the folder**: Right-click the folder > Properties > Sharing tab > Advanced Sharing... > check "Share this folder" > share name: `BOLD_DICOM` > Permissions > Add > Everyone > Full Control > OK
4. **Firewall**: Settings > search "Windows Defender Firewall" > Allow an app through firewall > check "File and Printer Sharing" for both Private and Public
5. **Static IP**: Settings > Network & Internet > Ethernet > Edit > Manual > IPv4 on > IP: `192.168.2.5`, Subnet: `255.255.255.0`

### macOS

1. **Create a dedicated local user**: System Settings > Users & Groups > Add User... > Standard account (not Admin) > set username and password. You do not need to log into this account.
2. **Enable SMB sharing**: System Settings > General > Sharing > File Sharing ON > click (i) > Options... > ensure "Share files and folders using SMB" is checked > check the box next to the dedicated user (enter their password when prompted) > Done
3. **Create and share the folder**:

```bash
mkdir -p /BOLD_DICOM
chmod 777 /BOLD_DICOM
```

In the File Sharing info panel: click + under Shared Folders and add `/BOLD_DICOM`, then click + under Users and add the dedicated user with Read & Write permissions.

4. **Firewall**: System Settings > Network > Firewall > turn OFF, or ensure file sharing is allowed
5. **Static IP**: System Settings > Network > Ethernet > Details > TCP/IP > Configure IPv4: Manually > IP: `192.168.2.5`, Subnet: `255.255.255.0`
6. **Verify**: `smbutil view //<username>@127.0.0.1` — should show `BOLD_DICOM` in the share list

### Linux

```bash
# 1. Install Samba
sudo apt-get install --upgrade samba

# 2. Create a Samba password for the user
sudo smbpasswd -a <username>

# 3. Create the DICOM watch directory
sudo mkdir -p /BOLD_DICOM
sudo chown <username> /BOLD_DICOM
sudo chmod -R 773 /BOLD_DICOM

# 4. Add the share to /etc/samba/smb.conf:
#    [BOLD_DICOM]
#       path = /BOLD_DICOM
#       browsable = yes
#       writable = yes
#       valid users = <username>
#       comment = Samba Share for BOLD dot-addin dicoms

# 5. Restart Samba
sudo systemctl restart smbd nmbd

# 6. Verify the share
smbclient -L localhost -U <username>
```

### Scanner-Side Configuration (Siemens BOLD Dot Add-In)

After setting up the SMB share, configure the scanner to export DICOMs. Open the Dot Add-In Configurator via Dot Cockpit > protocol program and set:

- **Enable Auto Export**: checked
- **Target Host**: `\\192.168.2.5` (the analysis computer's IP)
- **Target Directory**: `BOLD_DICOM`
- **User Name**: `WORKGROUP\<username>`
- **Password**: the dedicated user's password

**macOS only**: Before running BOLD sequences, open a Command Prompt on the scanner console (Ctrl+Esc > Advanced User > Command Prompt) and run:

```
net use Z: \\192.168.2.5\BOLD_DICOM /user:<username> <password>
```

Run this once per scanner session. This step is not needed for Windows or Linux. To disconnect when done: `net use Z: /delete`

## Step 3: Mount the Share on the Sender Machine

From the sender machine, mount the analysis machine's share so the simulator can write to it.

### Windows Sender

```powershell
net use Z: \\192.168.2.5\BOLD_DICOM /user:<username> *
```

The `*` prompts for the password interactively, which avoids issues with special characters (e.g., `&`) in passwords.

### macOS Sender

The easiest approach is to mount via Finder, which handles authentication reliably:

1. Finder > Go > Connect to Server (Cmd+K)
2. Enter: `smb://username@192.168.2.5/BOLD_DICOM`
3. Enter the analysis machine's password when prompted

Or from the command line (uses the same Finder mechanism):

```bash
open "smb://username@192.168.2.5/BOLD_DICOM"
```

The share will be mounted at `/Volumes/BOLD_DICOM`. Use that path as the simulator target.

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

Run a normal pineuro session (via `pineuro-gui` or programmatically) with the watch directory set to your SMB share path (`C:\BOLD_DICOM` on Windows, `/BOLD_DICOM` on macOS or Linux).

**Trigger:** For localizer and feedback runs, PsychoPy will wait for a trigger key before starting the task. Since the simulator on the other machine can't send a keystroke, you need to manually press the trigger key on the analysis machine's keyboard when you start the simulator. This matches the real scanner workflow where a TTL pulse or key event triggers the task.

### Sender machine — start the simulator

```bash
python -m pineuro.simulator \
    --source /path/to/your/dicoms \
    --target Z:/ \
    --speed 1.0 \
    --n 50
```

Replace `Z:/` with your mounted share path (`/Volumes/BOLD_DICOM` on macOS, `/mnt/BOLD_DICOM` on Linux). Use forward slashes — backslashes are interpreted as escape characters in bash.

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

## After Testing: Reset the Ethernet IP

If the laptop you used as the "sender" (scanner role) will later be used as the analysis machine at the scanner site, remember to change its ethernet static IP back to the analysis address (e.g., `192.168.2.5`) before going to the scanner. During testing the sender uses a different IP (e.g., `192.168.2.1`), and the scanner's BOLD dot-addin expects to reach the analysis machine at its configured address.

## Platform Notes

- **Windows**: File system events (`ReadDirectoryChangesW`) work reliably for local directories being written to via SMB. This is the most thoroughly tested path.
- **macOS**: File system events (`FSEvents`) work for local directories. pineuro also has a polling fallback (every 0.5s) that catches any missed events.
- **Linux**: `inotify` does not fire for files arriving over network mounts. pineuro's polling fallback handles this, but detection latency will be up to 0.5s instead of near-instant. Since the analysis machine hosts the share locally, this is only relevant if your analysis machine is Linux and you're watching a remote mount (unusual setup).
