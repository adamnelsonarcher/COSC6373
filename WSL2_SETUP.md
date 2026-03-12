# WSL2 + Cursor + TensorFlow GPU — Setup Guide

Use this so your COSC6373 project runs in WSL2 with Cursor and TensorFlow using your GPU.

**Drive note:** This guide puts WSL (and all Linux files, venvs, and packages) on a drive other than C:, so the bulk of the space stays off your system drive.

---

## Part 0: Put WSL on another drive (D:, E:, etc.)

WSL stores each distro in a virtual disk (VHDX). By default it goes on C:. To use another drive (e.g. **D:**), install once on C:, then **move** the distro. After the move, all Linux files, `apt` packages, Python venvs, and TensorFlow live on that drive.

Replace **D:** in the examples with your drive letter (e.g. **E:**, **F:**).

### 0.0 Find where your distro is now (already have WSL?)

The path `\\wsl.localhost\Ubuntu` is just the way Windows exposes the Linux filesystem—it does **not** tell you which drive the distro’s data lives on. To see the real location (and thus the drive):

In **PowerShell** (no admin needed):

```powershell
Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss | ForEach-Object { Get-ItemProperty $_.PSPath } | Select-Object DistributionName, BasePath
```

You’ll get a table like:

| DistributionName | BasePath |
|------------------|----------|
| Ubuntu           | C:\Users\PC\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu...\LocalState |

- **BasePath** is the folder that contains the distro’s `ext4.vhdx`. The **drive** is the first part of that path (e.g. **C:** or **D:**).
- If BasePath starts with `C:\`, the distro is on **C:**. To move it to another drive, follow **0.1–0.3** below and use your distro’s **exact** name (e.g. `Ubuntu` or `Ubuntu-24.04`) in the `wsl --manage` / `wsl --export` / `wsl --import` commands.

### 0.1 Create the target folder

In **PowerShell (Admin)** (use your drive letter):

```powershell
# Example: D: drive. Use E:\, F:\, etc. if you prefer.
New-Item -ItemType Directory -Path "D:\WSL" -Force
```

### 0.2 Install WSL and Ubuntu (initially on C:)

In **PowerShell (Admin)**:

```powershell
wsl --install
```

Restart if prompted, then:

```powershell
wsl --set-default-version 2
wsl --install -d Ubuntu-24.04
```

When Ubuntu opens, create your Linux username and password. You can do minimal setup; we’ll move it next.

### 0.3 Move the distro to your drive

Use the **exact** distro name from the table in 0.0 (e.g. `Ubuntu` or `Ubuntu-24.04`) in place of `Ubuntu` below.

**Option A — Move (Windows 10 22H2+ / Windows 11)**  
Simplest: move the distro in one step. In **PowerShell (Admin)**:

```powershell
# Shut down the distro first
wsl --shutdown
wsl --terminate Ubuntu

# Move to D:\WSL (replace D: and "Ubuntu" with your drive and distro name)
wsl --manage Ubuntu --move "D:\WSL\Ubuntu"
```

If `--manage --move` isn’t available (older Windows), use Option B.

**Option B — Export then import (all WSL2 versions)**  
In **PowerShell (Admin)** (replace `Ubuntu` with your distro name):

```powershell
wsl --shutdown
# Export to a tar on your target drive (can delete after import)
wsl --export Ubuntu "D:\WSL\ubuntu-backup.tar"

# Remove the distro from C: (frees space)
wsl --unregister Ubuntu

# Import onto your drive (distro name, install folder, backup file)
wsl --import Ubuntu "D:\WSL\Ubuntu" "D:\WSL\ubuntu-backup.tar" --version 2
```

After import, the default user is root. To use your normal user when you run `wsl`:

1. In PowerShell: `wsl -d Ubuntu` (you’ll be root; use your distro name if different).
2. In WSL, create a file:  
   `echo -e "[user]\ndefault=YOUR_LINUX_USERNAME" | sudo tee /etc/wsl.conf`
3. Replace `YOUR_LINUX_USERNAME` with the username you created during install. If you used Option B and never created a user, create one first:  
   `adduser yourusername` then add to `wsl.conf`.
4. In PowerShell: `wsl --shutdown`, then open WSL again; you should be your normal user.

Set as default distro:

```powershell
wsl --set-default Ubuntu
```

### 0.4 Verify location

In PowerShell:

```powershell
# Distro should run from D:\WSL\Ubuntu (or the path you used)
dir "D:\WSL\Ubuntu"
```

You should see a `.vhdx` file there. Run the 0.0 command again; **BasePath** should now be on your chosen drive.

---

## Part 1: WSL2 and Ubuntu (confirm / already moved)

### 1.1 Confirm WSL2 is running

In PowerShell:

```powershell
wsl --list --verbose
```

You should see your distro (e.g. Ubuntu-24.04) with **VERSION 2**. If you moved it in Part 0, it’s now on your other drive (e.g. D:\WSL).

### 1.2 (Optional) Confirm distro location

If you used the export/import or move method, the distro’s files are under the folder you chose (e.g. `D:\WSL\Ubuntu-24.04`). That’s where the space is used, not on C:.

---

## Part 2: Project location — same files from Windows and WSL (recommended)

You can keep **one** project folder and use it from both Windows (regular Cursor, Windows Python) and WSL (Linux, TensorFlow GPU). No copying; same repo, same files.

### Option 1: Shared folder — project on Windows (or E:), WSL mounts it

Keep your repo where it is (e.g. `C:\Users\PC\Desktop\...\COSC6373` or `E:\...\COSC6373`). Both sides use that folder.

| You're in | Path to project |
|-----------|------------------|
| **Windows** (Cursor, Explorer, Windows Python) | `C:\Users\PC\Desktop\projects_and_code\School\COSC6373` or `E:\...\COSC6373` |
| **WSL** (terminal, Cursor connected to WSL) | `/mnt/c/Users/PC/Desktop/projects_and_code/School/COSC6373` or `/mnt/e/.../COSC6373` |

- **Windows:** Open the folder in Cursor as usual; use your Windows Python/kernel for “regular” work.
- **WSL:** `cd /mnt/c/Users/PC/Desktop/projects_and_code/School/COSC6373` (or `/mnt/e/...`). Create a **Linux** venv there (e.g. `python3 -m venv .venv`) and use it for GPU/TensorFlow. The `.venv` folder will live next to your code; Windows ignores it (different Python), WSL uses it.
- **Same files:** Edits in either place are the same files. Git, notebooks, and data are shared.

**Caveat:** WSL access to `/mnt/c` or `/mnt/e` is a bit slower than native Linux disk and a few tools (e.g. heavy file watchers) can be fussy. For notebooks, TensorFlow, and typical dev work it’s usually fine.

### Option 2: Shared folder — project inside WSL, Windows opens it

Keep the project in WSL (e.g. `~/COSC6373`). Windows accesses it via the WSL network path.

| You're in | Path to project |
|-----------|------------------|
| **WSL** | `~/COSC6373` or `/home/yourusername/COSC6373` (fast) |
| **Windows** (Cursor, Explorer) | `\\wsl.localhost\Ubuntu-24.04\home\yourusername\COSC6373` |

- **WSL:** Best performance; create venv in the project, use for GPU.
- **Windows:** In Cursor use **File → Open Folder** and paste `\\wsl.localhost\Ubuntu-24.04\home\yourusername\COSC6373`, or in Explorer go to **\\wsl.localhost\Ubuntu-24.04** and browse to the folder. Use your Windows Python when running from Windows.
- **Same files:** One copy; edits from either side are the same files.

### Option 3: WSL-only (copy or clone into WSL)

If you only care about WSL and want maximum WSL performance, copy or clone the repo into the Linux filesystem (e.g. `~/COSC6373`). Then you have two copies unless you stop using the Windows one. Use this only if you’re fully switching to WSL for this project.

**Summary:** For “sometimes Windows, sometimes WSL, same files,” use **Option 1** (project on C: or E:, WSL uses `/mnt/c/...` or `/mnt/e/...`) or **Option 2** (project in WSL, Windows uses `\\wsl.localhost\...`). One venv in WSL for GPU; use your existing Windows Python when on Windows.

---

## Part 3: Cursor + WSL

### 3.1 Install WSL support in Cursor

1. Open Cursor.
2. Extensions: `Ctrl+Shift+X` → search **"WSL"**.
3. Install **"WSL"** by Microsoft (or the one Cursor suggests for WSL).

### 3.2 Open your project in WSL

**Method 1 — From Cursor (recommended)**  
1. Click the **green/blue button** in the bottom-left (Remote / connection).  
2. Choose **"Connect to WSL"** (or "New WSL Window") and pick your distro (e.g. Ubuntu-24.04).  
3. A new Cursor window opens in WSL. Use **File → Open Folder** and choose your project, e.g. `/home/yourusername/COSC6373`.  
4. If prompted, install the Cursor server in WSL and reload. Extensions may need to be installed in WSL.

**Method 2 — From WSL terminal**  
In WSL:

```bash
# Add Cursor to PATH once (adjust path if your Cursor install differs)
export PATH="$PATH:/mnt/c/Users/PC/AppData/Local/Programs/cursor/resources/app/bin"
# Open project
cursor ~/COSC6373
```

(You can add the `export` line to `~/.bashrc` so it’s permanent.)

**Important:** Open the folder that’s **inside** WSL (e.g. `~/COSC6373`), not a path like `\\wsl$\...\COSC6373` from the Windows side, so the integrated terminal and tools run in Linux.

---

## Part 4: GPU driver (Windows only)

WSL2 uses the **Windows** NVIDIA driver. You do **not** install the display driver inside Linux.

1. On Windows, install the latest [NVIDIA Game Ready or Studio driver](https://www.nvidia.com/Download/index.aspx) for your GPU.  
2. In **PowerShell** (or CMD) on Windows, run:

   ```powershell
   nvidia-smi
   ```

   You should see your GPU and driver version. If this works, WSL2 can use the GPU.

---

## Part 5: Python and TensorFlow GPU inside WSL

All of this is in the **WSL terminal** (or Cursor’s terminal with the folder opened in WSL). Python, venvs, and pip packages (including TensorFlow) are stored inside the WSL virtual disk—so if WSL is on D: (or another drive), the space is used there, not on C:.

If you’re using the **shared folder** (Option 1 in Part 2), `cd` to the project via its `/mnt/c/...` or `/mnt/e/...` path instead of `~/COSC6373`; the venv will still be created inside the project folder and only used when you’re in WSL.

### 5.1 System packages

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
```

### 5.2 (Recommended) Use a venv for the project

```bash
# If project is in WSL home:
cd ~/COSC6373

# If project is shared on Windows (e.g. C: or E:), use the mount path:
# cd /mnt/c/Users/PC/Desktop/projects_and_code/School/COSC6373
# cd /mnt/e/path/to/COSC6373

python3 -m venv .venv
source .venv/bin/activate
```

### 5.3 Install TensorFlow with GPU support

On Linux, `tensorflow[and-cuda]` works (no NCCL-on-Windows issue):

```bash
pip install --upgrade pip
pip install "tensorflow[and-cuda]"
pip install matplotlib numpy pandas scikit-learn kagglehub
```

### 5.4 Jupyter for notebooks

```bash
pip install jupyter
```

Run from the project directory:

```bash
jupyter notebook
# or
jupyter lab
```

Use the URL shown in the terminal (or open from Cursor’s “Simple Browser” or your Windows browser). The kernel will use the venv’s Python (with TensorFlow GPU).

### 5.5 Verify GPU in WSL

In the same WSL environment (with venv activated):

```bash
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('TensorFlow', tf.__version__); print('GPUs:', gpus if gpus else 'None')"
```

You should see at least one GPU listed.

---

## Part 6: Use the notebook

**When you want GPU (WSL):**  
1. In Cursor, connect to WSL (bottom-left → Connect to WSL → Ubuntu-24.04) and open the project folder (e.g. `~/COSC6373` or `/mnt/c/.../COSC6373` if using the shared folder).  
2. Open the notebook (e.g. `HW8/Part_B/...ipynb`).  
3. Select kernel → “Python Environments” → the WSL venv (e.g. `.../COSC6373/.venv`).  
4. Run the GPU check cell and the rest; TensorFlow will use the GPU.

**When you’re on your regular Windows setup:**  
Open the same project folder from Windows (no WSL). Choose your Windows Python/kernel. Same files, no GPU (unless you have TensorFlow GPU working on Windows).

If you prefer Jupyter from the terminal: in WSL, `cd` to the project, `source .venv/bin/activate`, then `jupyter notebook`.

---

## Quick reference

| Step                 | Where        | What to do                                                                 |
|----------------------|-------------|-----------------------------------------------------------------------------|
| WSL on other drive   | PowerShell  | Install WSL + Ubuntu, then `wsl --manage Ubuntu-24.04 --move "D:\WSL\..."` or export/import to D:\WSL |
| Confirm WSL2         | PowerShell  | `wsl --list --verbose`                                                      |
| Project in WSL        | WSL         | Copy or clone to `~/COSC6373` (lives on your WSL drive’s VHDX)              |
| Cursor + WSL          | Cursor      | Install WSL extension → Connect to WSL → Open `~/COSC6373`                  |
| GPU driver            | Windows     | Install latest NVIDIA driver, run `nvidia-smi`                              |
| Python + TF GPU       | WSL         | venv in project, `pip install "tensorflow[and-cuda]"` (space on WSL drive) |
| Check GPU             | WSL         | `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` |

---

## Troubleshooting

- **WSL / move to another drive**  
  - Ensure the target drive has enough free space (tens of GB for the VHDX and growth).  
  - Use an empty folder for the install path (e.g. `D:\WSL\Ubuntu-24.04`).  
  - If `--manage --move` fails, use the export → unregister → import steps in Part 0.  
  - After import, set default user in `/etc/wsl.conf` and run `wsl --shutdown` once.

- **“Failed to attach disk … ext4.vhdx … Access is denied” (E_ACCESSDENIED) when connecting Cursor to WSL**  
  Usually the VHDX on another drive (e.g. E:) is owned by SYSTEM/Administrators; your user must own it. Try in order:  
  1. **PowerShell (Admin):** `wsl --shutdown`, wait ~10 seconds.  
  2. **Set your user as owner of the VHDX** (WSL often won’t attach until the file is owned by your account). Replace `desktop\pc` with the output of `whoami` if different:  
     ```powershell
     $path = "E:\WSL\Ubuntu-24.04"
     $user = "desktop\pc"
     $Account = New-Object System.Security.Principal.NTAccount $user
     $Acl = Get-Acl "$path\ext4.vhdx"
     $Acl.SetOwner($Account)
     Set-Acl "$path\ext4.vhdx" -AclObject $Acl
     icacls "$path\ext4.vhdx" /grant "${user}:(F)"
     icacls "$path" /grant "${user}:(OI)(CI)(F)"
     ```  
     Then run `wsl` or connect from Cursor again (do **not** run Cursor as admin).  
  3. If it still fails: **move the distro back to C:** so WSL works at least (see “Move distro back to C:” below).  
  4. Restart the PC and retry (clears stuck file handles).

- **Move distro back to C: (so WSL works again)**  
  If E: (or another drive) keeps giving E_ACCESSDENIED, move the distro to a folder on C: (PowerShell as Admin):  
  `New-Item -ItemType Directory -Path "C:\WSL\Ubuntu-24.04" -Force`  
  `wsl --shutdown`  
  `wsl --manage Ubuntu-24.04 --move "C:\WSL\Ubuntu-24.04"`  
  Then run `wsl`. You can try moving to E: again later (e.g. after a Windows update or with a fresh export/import).

- **“Failed to patch code.sh launcher: ENOENT … open '…\bin\code'”**  
  The WSL extension expects a `code` launcher in Cursor’s bin folder. If you removed it (e.g. to stop Cursor overriding the `code` command), recreate it: copy `cursor` to `code` and `cursor.cmd` to `code.cmd` in `%LocalAppData%\Programs\cursor\resources\app\bin\`. Cursor updates may overwrite this; re-copy if the error returns.

- **Cursor can’t connect to WSL**  
  Ensure WSL is running: in PowerShell run `wsl`. Install/reinstall the WSL extension and try “Connect to WSL” again.

- **No GPU in TensorFlow**  
  - Run `nvidia-smi` on **Windows**; if it fails, update the NVIDIA driver.  
  - In WSL, ensure you’re in the venv where you installed `tensorflow[and-cuda]` and run the GPU check again.

- **Kernel / Python mismatch**  
  In Cursor, select the kernel that points to the WSL venv Python (e.g. `~/COSC6373/.venv/bin/python`).

- **Slow when project is on /mnt/c**  
  Move or clone the project into the Linux home directory (e.g. `~/COSC6373`) and open that folder in Cursor. With WSL on D: (or your drive), that home directory lives on that drive’s VHDX.
