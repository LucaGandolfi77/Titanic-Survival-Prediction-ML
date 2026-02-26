# GR740 On-Board Computer Flight Software

**Project:** GR740-OBC-FSW  
**Version:** 1.0.0  
**Standard:** ECSS-E-ST-40C / ECSS-Q-ST-80C / DO-178C (DAL B/C)  
**License:** ESA Public License v2.0  

---

## 1. Mission Overview

This flight software (FSW) runs on the GR740 LEON4FT quad-core radiation-hardened
processor, serving as the On-Board Computer (OBC) for a small/medium satellite mission.
The OBC is responsible for:

- **Command & Data Handling (C&DH):** Receive, validate, and execute telecommands
  from ground via CCSDS/PUS protocols.
- **Telemetry Generation:** Collect housekeeping data and science telemetry, frame
  into CCSDS TM transfer frames for downlink.
- **Satellite Mode Management:** Control mission phases (Safe, Nominal, Science,
  Eclipse, Detumbling) via a deterministic finite state machine.
- **Fault Detection, Isolation & Recovery (FDIR):** Monitor subsystem health and
  autonomously recover from anomalies.
- **Subsystem Coordination:** Interface with ADCS, EPS, Payload, and Propulsion
  subsystems via SpaceWire, CAN, I2C, and SPI buses.

---

## 2. Hardware Block Diagram

```
                            +-----------------------------------+
                            |       GR740 LEON4FT OBC           |
                            |  +-------+  +-------+  +-------+  |
                            |  |Core 0 |  |Core 1 |  |Core 2 |  |
                            |  +-------+  +-------+  +-------+  |
                            |  +-------+                         |
                            |  |Core 3 |    AMBA AHB/APB Bus     |
                            |  +-------+                         |
                            |                                    |
    +---------+   SpW x4    |  +------+  +------+  +-------+    |
    | Payload |<----------->|  |GRSPW2|  |GRCAN |  |APBUART|    |
    +---------+             |  +------+  +------+  +-------+    |
                            |      |         |         |         |
    +---------+   SpW       |  +------+  +------+  +-------+    |
    |  ADCS   |<----------->|  |GPTMR |  |SPICTRL| |I2CMST |   |
    +---------+             |  +------+  +------+  +-------+    |
                            |      |         |         |         |
    +---------+   CAN       |  +------+  +------+  +-------+    |
    |   EPS   |<----------->|  |GRGPIO|  | SRAM |  | MRAM  |   |
    +---------+             |  |      |  | 32MB |  | 8MB   |   |
                            |  +------+  +------+  +-------+    |
    +---------+   UART      |                                    |
    | Ground  |<----------->|  +------+                          |
    +---------+             |  |EEPROM|  4MB                     |
                            |  +------+                          |
                            +-----------------------------------+

    Clock: 50 MHz | Supply: 1.0V core / 3.3V I/O
    Radiation: Total Dose >100 krad(Si), SEL immune to LET 62 MeV·cm²/mg
```

---

## 3. GR740 Processor Features Used

| Feature               | Description                                      |
|-----------------------|--------------------------------------------------|
| LEON4FT Quad-Core     | SPARC V8, fault-tolerant with hardware ECC        |
| GRSPW2 x4             | SpaceWire links, DMA, RMAP support                |
| GRCAN x2              | CAN 2.0B controller with TX/RX FIFOs              |
| APBUART x6            | Debug + TM/TC UART channels                       |
| GPTIMER               | General-purpose timers, watchdog                  |
| SPICTRL               | SPI master for MRAM, sensors                      |
| I2CMST                | I2C master for power monitors, IMU                |
| GRGPIO                | 32 GPIO pins for discrete I/O                     |
| IRQAMP                | Multi-processor interrupt controller              |
| AMBA AHB/APB          | On-chip bus with plug-and-play device enumeration  |
| Hardware ECC          | EDAC on SRAM with scrubbing                       |

---

## 4. Interface Summary

| Interface   | Protocol       | Usage                  | Speed      |
|-------------|----------------|------------------------|------------|
| SpaceWire   | GRSPW2/RMAP    | Payload, ADCS, memory  | 100 Mbps   |
| CAN         | GRCAN/CANopen  | EPS, propulsion, misc  | 1 Mbps     |
| UART        | APBUART        | Debug console, TM/TC   | 115200 bps |
| SPI         | SPICTRL        | MRAM, sensors          | 10 MHz     |
| I2C         | I2CMST         | Power monitors, IMU    | 400 kHz    |
| GPIO        | GRGPIO         | Deployment, discrete   | N/A        |

---

## 5. Build Instructions

### Prerequisites

- **BCC2 Toolchain** (Cobham Gaisler) or `sparc-rtems5-gcc`
- **RTEMS 5.x** BSP for GR740 (`sparc/leon4`)
- **GRMON3** debugger for target load/debug

### Build

```bash
# Set toolchain path
export CROSS_COMPILE=sparc-rtems5-

# Full build
make all

# Clean build artifacts
make clean

# Build unit tests (host)
make tests
```

### Toolchain Installation (BCC2)

```bash
# Download BCC2 from Cobham Gaisler
tar xzf bcc-2.x.x-gcc.tar.gz -C /opt/
export PATH=/opt/bcc-2.x.x/bin:$PATH
export CROSS_COMPILE=sparc-gaisler-elf-
```

---

## 6. Flash / Run Instructions

### Load via GRMON3

```bash
# Connect to GR740 eval board via JTAG
grmon3 -u -digilent

# Inside GRMON3:
load gr740-obc-fsw.elf
verify gr740-obc-fsw.elf
run
```

### Flash to MRAM (Flight)

```bash
grmon3 -u -digilent
# Write to MRAM at 0x20000000
load -binary gr740-obc-fsw.bin 0x20000000
verify -binary gr740-obc-fsw.bin 0x20000000
```

---

## 7. PUS Services Summary

| Service | Subtype | Name                    | Description                                  |
|---------|---------|-------------------------|----------------------------------------------|
| ST[1]   | 1,2,7,8 | TC Verification         | Acceptance/execution success/failure reports |
| ST[3]   | 25      | Housekeeping            | Periodic & on-demand HK parameter reports    |
| ST[5]   | 1-4     | Event Reporting         | Info, low/med/high severity event reports    |
| ST[8]   | 1       | Function Management     | Direct function invocation via TC            |
| ST[9]   | 1,2     | Time Management         | Set/report OBC time (CUC 4+2 format)        |
| ST[11]  | 4       | Time-tagged Commands    | Schedule TC execution at future time         |
| ST[17]  | 1,2     | Test (Are-You-Alive)    | Connection test ping/pong                    |

---

## 8. FDIR Logic

### Error Taxonomy

| Level    | Severity  | Action                                    |
|----------|-----------|-------------------------------------------|
| Level 1  | Soft      | Log error, increment counter              |
| Level 2  | Hard      | Reset subsystem, send PUS ST[5] event     |
| Level 3  | Critical  | Transition to SAFE mode                   |

### Monitored Faults

- SpaceWire link disconnection / credit error
- CAN bus-off condition
- Memory EDAC single/multi-bit errors
- Task stack overflow detection
- Task deadline miss (watchdog timeout)
- Temperature out-of-range (OBC, battery)
- Voltage rail deviation (EPS)
- Communication timeout (subsystem heartbeat)

### Recovery Strategy

```
Fault Detected → FDIR Table Lookup → Check Inhibit Flag
  → If inhibited: log only
  → If not inhibited:
      Level 1: log + counter
      Level 2: reset subsystem + event report
      Level 3: SAFE mode + event report + persist to MRAM
```

---

## 9. Satellite Modes

```
         ┌─────────┐
    ──-->│  BOOT   │
         └────┬────┘
              │ init complete
              v
         ┌─────────┐    FDIR critical    ┌─────────┐
    ┌--->│ NOMINAL │------------------->│  SAFE   │<───┐
    │    └────┬────┘                     └────┬────┘    │
    │         │ TC cmd                        │ TC cmd  │
    │         v                               │         │
    │    ┌──────────┐                         │         │
    │    │ SCIENCE  │─── FDIR ────────────────┘         │
    │    └──────────┘                                   │
    │         │                                         │
    │    ┌──────────┐                                   │
    └────│DETUMBLE  │─── FDIR ──────────────────────────┘
         └──────────┘
              │
         ┌──────────┐
         │ ECLIPSE  │ (autonomous entry on power low)
         └──────────┘
```

---

## 10. Memory Map

| Region       | Start Address  | Size   | Usage                          |
|--------------|----------------|--------|--------------------------------|
| Boot PROM    | 0x00000000     | 256 KB | Boot loader                    |
| SRAM         | 0x40000000     | 32 MB  | .data, .bss, stack, heap       |
| MRAM         | 0x20000000     | 8 MB   | Parameters, mode, flight log   |
| EEPROM       | 0x30000000     | 4 MB   | Firmware backup                |
| I/O Regs     | 0x80000000     | 256 MB | AMBA APB peripherals           |
| .text (exec) | 0x40000000     | varies | Program code (in SRAM at run)  |

---

## 11. Coding Standards

- **Language:** C99 strict (ISO/IEC 9899:1999)
- **Compliance:** MISRA-C:2012 (documented deviations)
- **Safety:** DO-178C DAL B/C, ECSS-Q-ST-80C
- **No dynamic allocation** after initialization
- **No recursion** (static stack analysis)
- **No VLAs** (Variable Length Arrays)
- Fixed-width integer types (`uint8_t`, `uint16_t`, `uint32_t`)
- All hardware registers accessed via `volatile` pointers
- Maximum cyclomatic complexity: 15 per function

---

## 12. License

Copyright (C) 2026 — ESA Public License v2.0

This software is provided under the ESA Public License version 2.0.
See LICENSE file for the full license text.

---

*Document generated for GR740-OBC-FSW v1.0.0*
