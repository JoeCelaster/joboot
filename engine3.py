#!/usr/bin/env python3
"""Ford GT engine: granular synth built from the real recording (sound/fordgt.mp3).

Every 5 ms slice of fordgt.mp3 was tagged with its measured RPM (from the 8th engine
order of the V8). While driving, the synth plays the real slices recorded at the current
RPM: on-throttle rev-ups when you hold UP, off-throttle rev-downs when you let go,
idle slices at idle. Above the recorded max (~3500 rpm) the top slices are pitched up.

    python3 engine3.py                 # UP = gas pedal (progressive), q = shut off
    python3 engine3.py --render "idle:2,gas:0.3,idle:2,gas:3,idle:3,off" out.mp3
"""
import argparse
import array
import bisect
import fcntl
import json
import math
import os
import random
import shutil
import subprocess
import sys
import termios
import time
import tty

from engine import Keys  # same UP-key detection (evdev / key-repeat fallback)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "fordgt")
META = json.load(open(os.path.join(DATA, "rpm_map.json")))
SR = META["sr"]
FR = META["frame"]              # samples per RPM-map frame (5 ms)
GRAIN = 1024                    # 32 ms grains, 50% overlap
HOP = GRAIN // 2
DT = HOP / SR
AHEAD = 0.06

IDLE_RPM = 950.0
LIMITER = 6800.0
REC_MAX = 3450.0                # highest RPM present in the recording
WINDOW = [0.5 - 0.5 * math.cos(2 * math.pi * i / GRAIN) for i in range(GRAIN)]


def decode(name):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(DATA, name), "-ac", "1",
                          "-ar", str(SR), "-f", "s16le", "-"], check=True, stdout=subprocess.PIPE).stdout
    pcm = array.array("h")
    pcm.frombytes(raw[: len(raw) // 2 * 2])
    return [s / 32768.0 for s in pcm]


class Bank:
    """All slices of one kind (idle / on / off), searchable by RPM."""

    def __init__(self):
        self.segs = []      # (samples, rpm[], gain[])
        self.index = []     # sorted (rpm, seg, frame)

    def add(self, samples, rpm, gain_db):
        s = len(self.segs)
        gain = [10 ** (g / 20) for g in gain_db]
        self.segs.append((samples, rpm, gain))
        last = (len(samples) - 2 * GRAIN) // FR   # leave room for a pitched-up grain
        for f in range(0, min(len(rpm), max(0, last))):
            self.index.append((rpm[f], s, f))
        self.index.sort()
        self.keys = [e[0] for e in self.index]

    def pick(self, rpm):
        i = bisect.bisect_left(self.keys, rpm)
        lo, hi = max(0, i - 4), min(len(self.index), i + 4)
        _, s, f = random.choice(self.index[lo:hi])
        return s, f * FR


class Voice:
    """One grain stream reading a bank; continues the recording naturally while it matches."""

    def __init__(self, bank):
        self.bank, self.seg, self.pos, self.ratio = bank, None, 0.0, 1.0

    def grain(self, rpm, level, acc):
        b = self.bank
        if self.seg is not None:
            samples, rmap, _ = b.segs[self.seg]
            p = self.pos + HOP * self.ratio
            f = int(p) // FR
            ok = f < len(rmap) and p + GRAIN * 2.3 < len(samples) and \
                abs(rmap[f] - min(rpm, REC_MAX)) < 0.04 * rpm
            if not ok:
                self.seg = None
            else:
                self.pos = p
        if self.seg is None:
            self.seg, self.pos = b.pick(min(rpm, REC_MAX))
        samples, rmap, gains = b.segs[self.seg]
        f = min(int(self.pos) // FR, len(rmap) - 1)
        r = self.ratio = max(0.5, min(2.3, rpm / rmap[f]))
        g = level * gains[f]
        p0 = self.pos
        for k in range(GRAIN):
            q = p0 + k * r
            i = int(q)
            a = samples[i]
            acc[k] += (a + (samples[i + 1] - a) * (q - i)) * WINDOW[k] * g


class Engine:
    def __init__(self):
        self.banks = {"idle": Bank(), "on": Bank(), "off": Bank()}
        for s in META["segments"]:
            self.banks[s["bank"]].add(decode(s["file"]), s["rpm"], s["gain_db"])
        self.voices = {k: Voice(b) for k, b in self.banks.items()}
        st, sh = META["start"], META["shutdown"]
        self.start_clip = [v * 10 ** (st["gain_db"] / 20) for v in decode(st["file"])]
        self.stop_clip = [v * 10 ** (sh["gain_db"] / 20) for v in decode(sh["file"])]
        self.fire_at = int(st["fire_at"] * SR)

        self.rpm = float(st["handoff_rpm"])
        self.pedal = 0.0            # progressive pedal travel 0..1
        self.cut = 0.0
        self.acc = [0.0] * (GRAIN + HOP)
        self.state = "starting"
        self.clip, self.cp = self.start_clip, 0
        self.level = 0.0            # granular engine level (for start/stop crossfades)
        self.fade = 0.0             # per-hop level change
        self.handoff = len(self.start_clip) - int(0.3 * SR)

    # ------------------------------------------------------------ physics
    def step(self, key):
        run = self.state == "running"
        if key and run:     # foot pushes the pedal down progressively: tap = small blip
            self.pedal = min(1.0, self.pedal + DT / 0.45)
        else:               # releases fast
            self.pedal = max(0.0, self.pedal - DT / 0.07)
        th = 0.0 if self.cut > 0 else self.pedal
        self.cut = max(0.0, self.cut - DT)
        r = self.rpm
        if th > 0.01:       # measured free-rev: ~5000 rpm/s around 2500 rpm
            r += th * 6800 * (1 - (r / 7400) ** 2) * DT
        # measured decay: 3500 -> 1600 rpm in ~1 s  (engine braking ~ 600 + 0.55*rpm)
        r -= (1 - th) * (600 + 0.55 * r) * DT * (1.0 if r > IDLE_RPM else 0.0)
        if r < IDLE_RPM:    # idle air control pulls it back up
            r += (IDLE_RPM - r) * min(1.0, 6 * DT)
        r += random.uniform(-6, 6)
        if self.state == "stopping":
            r = max(300.0, r - 2500 * DT)
        if r >= LIMITER:
            r, self.cut = LIMITER - 250, 0.06
        self.rpm = r

    # ------------------------------------------------------------ audio
    def render(self):
        acc = self.acc
        if self.state == "starting" and self.cp >= self.handoff:
            self.fade = 1.0 / (0.3 * SR / HOP)
        if self.fade:
            self.level = max(0.0, min(1.0, self.level + self.fade))
        if self.level > 0.001:
            r, th = self.rpm, (0.0 if self.cut > 0 else self.pedal)
            on = min(1.0, th * 2.5)
            idle = max(0.0, min(1.0, (1450 - r) / 450)) * (1 - on)
            off = (1 - on) * (1 - idle)
            boost = 1.0 + 0.6 * max(0.0, (r - REC_MAX) / (LIMITER - REC_MAX))
            lvl = self.level * boost
            for name, w in (("on", on * (0.55 + 0.45 * th)), ("off", off), ("idle", idle)):
                if w > 0.02:
                    self.voices[name].grain(r, w * lvl, acc)
                else:
                    self.voices[name].seg = None
        out = array.array("h", bytes(2 * HOP))
        clip, cp = self.clip, self.cp
        tanh = math.tanh
        for k in range(HOP):
            v = acc[k]
            if clip is not None and cp < len(clip):
                v += clip[cp]
                cp += 1
            out[k] = int(tanh(v * 1.1) * 31000)
        self.cp = cp
        self.acc = acc[HOP:] + [0.0] * HOP
        if clip is not None and cp >= len(clip):
            if self.state == "starting":
                self.state, self.clip, self.level, self.fade = "running", None, 1.0, 0.0
            elif self.state == "stopping":
                self.state = "off"
        return out

    def key_during_start(self):
        """Blipping the throttle right after it fires cuts the start clip short."""
        if self.state == "starting" and self.cp > self.fire_at and self.cp < self.handoff:
            self.handoff = self.cp
            self.clip = self.start_clip[: self.cp + int(0.3 * SR)]
            n = int(0.3 * SR)
            for i in range(n):
                j = self.cp + i
                if j < len(self.clip):
                    self.clip[j] *= 1 - i / n

    def shut_off(self):
        if self.state in ("running", "starting"):
            self.state = "stopping"
            self.clip, self.cp = self.stop_clip, 0
            self.fade = -1.0 / (0.25 * SR / HOP)


# ---------------------------------------------------------------- io
def open_player():
    for cmd in (["pw-cat", "--playback", "--raw", "--format=s16", f"--rate={SR}", "--channels=1",
                 "--latency=40ms", "-"],
                ["pacat", "--playback", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
                 "--latency-msec=40"],
                ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SR), "-c", "1", "--buffer-time=60000"]):
        if shutil.which(cmd[0]):
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            try:
                fcntl.fcntl(p.stdin.fileno(), 1031, 4096)
            except OSError:
                pass
            return p
    sys.exit("No audio player found (need pw-cat, pacat or aplay).")


def draw(eng, key):
    w = 44
    fill = int(w * min(1.0, eng.rpm / 7000))
    bar = "".join(("\x1b[31m" if i >= w * 0.85 else "\x1b[33m" if i >= w * 0.5 else "\x1b[32m") + "█"
                  if i < fill else "\x1b[90m·" for i in range(w)) + "\x1b[0m"
    ped = int(eng.pedal * 10)
    tag = {"starting": "\x1b[36mSTARTING…", "stopping": "\x1b[90mSHUTTING OFF"}.get(eng.state)
    if not tag:
        tag = ("\x1b[1;31mSCREAMING!!" if eng.rpm > 5500 else
               "\x1b[33mACCELERATING" if eng.pedal > 0 else
               "\x1b[35mREV-DOWN" if eng.rpm > 1300 else "\x1b[32mIDLE")
    sys.stdout.write(f"\r  [{bar}] {eng.rpm:5.0f} rpm  pedal[{'▮' * ped}{' ' * (10 - ped)}]  {tag}\x1b[0m\x1b[K")
    sys.stdout.flush()


def live(use_evdev):
    eng = Engine()
    keys = Keys(use_evdev)
    player = open_player()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    sys.stdout.write("\x1b[?25l")
    print("\n  🏁  Ford GT — hold ↑ UP for gas (tap = blip, hold = full throttle), q to shut off")
    print(f"      input: {keys.mode}\n")
    chirp = decode(META["chirp"])
    player.stdin.write(array.array("h", (int(v * 20000) for v in chirp)).tobytes())
    t0, written, last_draw = time.monotonic() + len(chirp) / SR, 0, 0.0
    try:
        while eng.state != "off":
            ahead = written / SR - (time.monotonic() - t0)
            if ahead > AHEAD:
                time.sleep(ahead - AHEAD)
            elif ahead < -0.1:
                t0 = time.monotonic() - written / SR
            now = time.monotonic()
            key = keys.poll(now)
            if keys.quit:
                eng.shut_off()
            if key:
                eng.key_during_start()
            eng.step(key)
            player.stdin.write(eng.render().tobytes())
            written += HOP
            if now - last_draw > 0.05:
                draw(eng, key)
                last_draw = now
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        termios.tcflush(fd, termios.TCIFLUSH)
        sys.stdout.write("\x1b[?25h\n\n")
        keys.close()
        try:
            player.stdin.close()
            player.wait(timeout=2)
        except Exception:
            player.kill()


def render(script, out_path):
    eng = Engine()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1",
                            "-i", "-", out_path], stdin=subprocess.PIPE)
    while eng.state == "starting":
        eng.step(False)
        enc.stdin.write(eng.render().tobytes())
    for part in script.split(","):
        name, _, secs = part.partition(":")
        if name == "off":
            eng.shut_off()
            while eng.state != "off":
                eng.step(False)
                enc.stdin.write(eng.render().tobytes())
            continue
        for _ in range(int(float(secs) / DT)):
            eng.step(name == "gas")
            enc.stdin.write(eng.render().tobytes())
    enc.stdin.close()
    enc.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-evdev", action="store_true", help="detect UP from terminal key-repeat instead")
    ap.add_argument("--render", nargs=2, metavar=("SCRIPT", "OUT"))
    args = ap.parse_args()
    if args.render:
        render(*args.render)
    elif not sys.stdin.isatty():
        sys.exit("Run this in a terminal.")
    else:
        live(not args.no_evdev)
