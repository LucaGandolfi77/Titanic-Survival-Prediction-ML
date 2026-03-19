# QEMU VM Manager

A complete Python CLI + TUI tool for managing QEMU virtual machines from a
headless SSH terminal.  Start, stop, pause, monitor, and interact with VMs
entirely from the command line — VMs persist in the background even after
the SSH session disconnects.

---

## System prerequisites

| Dependency | Required | Notes |
|------------|----------|-------|
| **Python 3.10+** | Yes | Type hints use `X \| Y` syntax |
| **QEMU** | Yes | `qemu-system-x86_64` (or other arch) |
| **PyYAML** | Yes | `pip install pyyaml` |
| **tmux** or **screen** | Recommended | Used to detach VMs from the terminal; falls back to POSIX daemon if neither is available |
| **socat** | Optional | Cleaner serial console attachment |
| **qemu-img** | Optional | Required only for `vm disk` subcommands |

### Installing QEMU

```bash
# Debian / Ubuntu
sudo apt-get install qemu-system qemu-utils

# Arch Linux
sudo pacman -S qemu-full

# Alpine Linux
sudo apk add qemu-system-x86_64 qemu-img
```

---

## Installation

```bash
cd qemu-vm-manager
pip install -r requirements.txt
```

No build step is needed — all modules are plain Python files.

---

## Quickstart

### 1. Create a VM config

```bash
mkdir -p ~/.vms
cp examples/debian-minimal.yaml ~/.vms/my-vm.yaml
# Edit ~/.vms/my-vm.yaml — at minimum set disk_image to a real .qcow2 path.
```

### 2. Download a minimal test image

```bash
# Alpine Linux — tiny and boots fast
wget https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/x86_64/alpine-virt-3.19.1-x86_64.iso \
     -O /var/vms/alpine.iso

# Create a disk for it
python main.py disk create /var/vms/alpine.qcow2 2

# Update your YAML:
#   disk_image: "/var/vms/alpine.qcow2"
#   extra_args: ["-cdrom", "/var/vms/alpine.iso", "-boot", "d"]
```

### 3. Start the VM

```bash
python main.py start my-vm
```

### 4. Attach to the serial console

```bash
python main.py console my-vm
# Ctrl+] to detach
```

### 5. Open the live dashboard

```bash
python main.py dashboard
```

---

## CLI reference

| Command | Description |
|---------|-------------|
| `vm start <name_or_config>` | Start a VM (detached, survives SSH disconnect) |
| `vm stop <name> [-f]` | Graceful shutdown (or force with `-f`) |
| `vm pause <name>` | Suspend execution (SIGSTOP) |
| `vm resume <name>` | Resume execution (SIGCONT) |
| `vm restart <name_or_config>` | Stop + start |
| `vm status [name]` | Show status (all VMs if name omitted) |
| `vm list` | List all managed VMs with state |
| `vm console <name>` | Attach to serial console (Ctrl+] to detach) |
| `vm monitor <name>` | Open interactive QEMU monitor REPL |
| `vm qmp <name> <command> [json]` | Send a single QMP command |
| `vm disk create <path> <size_gb>` | Create a qcow2 disk image |
| `vm disk info <path>` | Show disk image metadata |
| `vm disk snapshot <name> <tag>` | Create internal QEMU snapshot |
| `vm disk resize <path> <size>` | Resize a disk image |
| `vm network <name>` | Show port-forward rules and port availability |
| `vm dashboard` | Launch live curses TUI |
| `vm config validate <path>` | Validate a YAML config file |

### Global flags

| Flag | Description |
|------|-------------|
| `--state-dir PATH` | State/log/socket directory (default `~/.vms`) |
| `--config-dir PATH` | YAML config directory (default `~/.vms`) |
| `-v / --verbose` | Enable debug logging |

---

## VM config YAML schema

```yaml
name: "vm-debian"
arch: "x86_64"                   # x86_64, aarch64, riscv64, …
cpu: "host"                      # QEMU -cpu value
cores: 2
ram_mb: 512
disk_image: "/var/vms/debian.qcow2"
kernel: null                     # optional
initrd: null                     # optional
kernel_args: "console=ttyS0"
network:
  type: "user"                   # user / tap / none
  hostfwd:
    - "tcp::2222-:22"
    - "tcp::8080-:80"
serial: "stdio"
monitor: "telnet::4444,server,nowait"
extra_args: []                   # raw QEMU flags
```

---

## TUI dashboard keyboard shortcuts

| Key | Action |
|-----|--------|
| ↑ / ↓ | Select VM |
| `s` | Start selected VM |
| `k` | Stop (kill) selected VM |
| `p` | Pause / resume |
| `c` | Attach to serial console |
| `m` | Open QEMU monitor REPL |
| `q` | Quit dashboard |

---

## Project structure

```
qemu-vm-manager/
├── main.py              # Entry point
├── cli.py               # argparse CLI + dispatch
├── config.py            # VMConfig dataclass, YAML loading, validation
├── qemu_builder.py      # Assemble QEMU command from config
├── process_manager.py   # Start/stop/pause, state file, PID tracking
├── monitor.py           # QMP client, human monitor, console attach
├── disk_manager.py      # qemu-img wrappers
├── tui.py               # Curses live dashboard
├── logging_utils.py     # Per-VM logs, event log, Python logging setup
├── requirements.txt
├── examples/
│   └── debian-minimal.yaml
├── tests/
│   ├── test_config.py
│   ├── test_qemu_builder.py
│   └── test_process_manager.py
└── README.md
```

---

## Running tests

```bash
pip install pytest
cd qemu-vm-manager
pytest tests/ -v
```

---

## Troubleshooting

### VM not starting

1. Check that `qemu-system-<arch>` is on your PATH: `which qemu-system-x86_64`
2. Validate your config: `python main.py config validate ~/.vms/my-vm.yaml`
3. Check the log: `cat ~/.vms/logs/<name>.log`

### Port already in use

The CLI warns you at `vm start` time.  Check with:

```bash
python main.py network <name>
```

To find the occupying process: `lsof -i :<port>` or `ss -tlnp | grep <port>`.

### Console not attaching

- Ensure the VM is running: `python main.py status <name>`
- The serial socket must exist: `ls ~/.vms/sockets/<name>-serial.sock`
- Install `socat` for best results: `sudo apt-get install socat`
- Fall back to the QEMU monitor: `python main.py monitor <name>`

### Tap networking requires root

User-mode networking (`type: "user"`) works without privileges.  Tap
networking (`type: "tap"`) typically requires root or membership in a
`netdev` group.  Stick with `user` unless you need bridged networking.

---

## License

MIT
