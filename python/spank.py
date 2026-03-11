"""Spank detector: play a short sound when a slap-like gyroscope spike is detected.

Usage examples:
  python spank.py --simulate
  python spank.py --serial /dev/ttyUSB0
  python spank.py --sensehat

The script supports several input backends (Sense HAT, MPU6050 via i2c, generic
serial lines that stream gyro data) and a `--simulate` mode for testing on any
machine. It generates a short beep using sounddevice (preferred) or falls back
to writing a WAV and using playsound.

All messages and comments are in English.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from playsound import playsound
except Exception:
    playsound = None

try:
    from sense_hat import SenseHat
except Exception:
    SenseHat = None

try:
    # optional mpu6050 library (pip install mpu6050-raspberrypi)
    from mpu6050 import mpu6050
except Exception:
    mpu6050 = None

import serial


class GyroReader:
    def read_gyro(self) -> Tuple[float, float, float]:
        """Return angular velocity (deg/s) tuple (x, y, z)."""
        raise NotImplementedError()


def select_readable():
    # cross-platform non-blocking check for stdin
    import select

    r, _, _ = select.select([sys.stdin], [], [], 0)
    return r


class SimulatedGyro(GyroReader):
    def __init__(self):
        self.t0 = time.time()

    def read_gyro(self):
        # produce quiet noise and occasional slap spikes when pressing Enter
        # for demo: press Enter to simulate a slap
        if sys.stdin in select_readable():
            _ = sys.stdin.readline()
            return (0.0, 0.0, 800.0)
        return (0.0, 0.0, 0.0)


class SenseHatGyro(GyroReader):
    def __init__(self):
        if SenseHat is None:
            raise RuntimeError("Sense HAT library not available")
        self.s = SenseHat()

    def read_gyro(self):
        o = self.s.get_gyroscope()  # degrees per second
        return (o.get("x", 0.0), o.get("y", 0.0), o.get("z", 0.0))


class MPU6050Gyro(GyroReader):
    def __init__(self, addr=0x68):
        if mpu6050 is None:
            raise RuntimeError("mpu6050 library not available")
        self.sensor = mpu6050(addr)

    def read_gyro(self):
        g = self.sensor.get_gyro_data()
        return (g.get("x", 0.0), g.get("y", 0.0), g.get("z", 0.0))


class SerialGyro(GyroReader):
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.1):
        self.s = serial.Serial(port, baudrate=baud, timeout=timeout)

    def read_gyro(self):
        # expect lines like: GYRO: x,y,z or x,y,z
        line = self.s.readline().decode(errors="ignore").strip()
        if not line:
            return (0.0, 0.0, 0.0)
        if line.upper().startswith("GYRO"):
            _, vals = line.split(":", 1)
        else:
            vals = line
        try:
            parts = [float(p) for p in vals.split(",")]
            if len(parts) >= 3:
                return (parts[0], parts[1], parts[2])
        except Exception:
            pass
        return (0.0, 0.0, 0.0)


def play_beep(duration=0.12, freq=1000.0, volume=0.2):
    """Play a short beep. Uses sounddevice if available, otherwise plays a small WAV via playsound."""
    if sd is not None and np is not None:
        fs = 44100
        t = np.linspace(0, duration, int(fs * duration), False)
        tone = np.sin(freq * 2 * math.pi * t) * volume
        # window the tone to avoid clicks
        window = np.hanning(len(tone))
        tone = tone * window
        sd.play(tone, fs)
        sd.wait()
        return

    if playsound is not None:
        # generate a tiny wav file and play it
        try:
            import wave
            import struct

            fs = 44100
            n = int(fs * duration)
            with wave.open("/tmp/spank_beep.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(fs)
                for i in range(n):
                    value = int(32767.0 * volume * math.sin(2 * math.pi * freq * (i / fs)))
                    data = struct.pack("<h", value)
                    wf.writeframesraw(data)
            playsound("/tmp/spank_beep.wav")
            return
        except Exception:
            pass

    print("[INFO] Beep requested but no audio backend available")


def detect_spank(reader: GyroReader, threshold: float = 300.0, window: float = 0.05):
    """Main loop: detect sudden spike in gyro magnitude.

    threshold: deg/s change considered a slap
    window: seconds between samples used to compute delta
    """
    last = reader.read_gyro()
    try:
        while True:
            time.sleep(window)
            cur = reader.read_gyro()
            dx = cur[0] - last[0]
            dy = cur[1] - last[1]
            dz = cur[2] - last[2]
            delta = math.sqrt(dx * dx + dy * dy + dz * dz)
            # debug print
            print(f"gyro={cur}, delta={delta:.1f}")
            if delta >= threshold:
                print("Spank detected! Playing sound...")
                play_beep()
                # short cooldown to avoid repeated triggers
                time.sleep(0.5)
            last = cur
    except KeyboardInterrupt:
        print("Exiting")


def build_reader_from_args(args) -> GyroReader:
    if args.simulate:
        return SimulatedGyro()
    if args.sensehat:
        return SenseHatGyro()
    if args.mpu:
        return MPU6050Gyro()
    if args.serial:
        return SerialGyro(args.serial, baud=args.baud)
    raise RuntimeError("No input backend selected. Use --simulate or --serial etc.")


def main():
    p = argparse.ArgumentParser(description="Spank detector: play sound on gyro spike")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--simulate", action="store_true", help="simulate input (press Enter to simulate slap)")
    g.add_argument("--sensehat", action="store_true", help="use Raspberry Pi Sense HAT gyro")
    g.add_argument("--mpu", action="store_true", help="use MPU6050 via i2c")
    g.add_argument("--serial", type=str, help="read gyro lines from serial port")
    p.add_argument("--baud", type=int, default=115200, help="serial baud rate")
    p.add_argument("--threshold", type=float, default=300.0, help="delta threshold (deg/s) to trigger sound")
    p.add_argument("--window", type=float, default=0.05, help="sampling window in seconds")
    args = p.parse_args()

    reader = build_reader_from_args(args)
    print("Starting spank detector. Press Ctrl+C to quit.")
    detect_spank(reader, threshold=args.threshold, window=args.window)


if __name__ == "__main__":
    main()
