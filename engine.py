#!/usr/bin/env python3
"""Terminal car engine: crank it, let it idle, and use the UP arrow as the gas pedal.

    python3 engine.py                 # play live (UP = throttle, RIGHT/LEFT = shift up/down, q = shut off)
    python3 engine.py --render "idle:1,gas:0.3,idle:1.5,gas:4,idle:2" out.mp3
    python3 engine.py --render "up,gas:2,up,gas:2,up,gas:3,down,idle:2,off" drive.mp3

Starts in 1st with the clutch slipping: hold UP and the car pulls away. Shifting is fully
manual: RIGHT to shift up (or it'll bounce off the limiter), LEFT down to neutral (N) to free-rev. In gear, the rpm is tied to
the car's speed, so an upshift drops the revs and a downshift blips them back up (rev-matched).
Any downshift goes in: drop a gear too early and the wheels drag the engine past the limiter.

Sound: made entirely from the real Chevy (chevy.mp3). Its idle is cut into individual
firing pulses and re-fired faster as the revs rise, so it keeps that hard, lumpy V8 tone.

Only needs python3 + ffmpeg (for decoding) + one of pacat / pw-cat / aplay.
Start / idle / shutdown sounds come from split/ (made from chevy.mp3).
"""
import argparse
import array
import fcntl
import glob
import math
import os
import random
import re
import select
import shutil
import struct
import subprocess
import sys
import termios
import time
import tty

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(HERE, "split")
SR = 24000
CHUNK = 256                     # samples per synthesis step (~10 ms)
DT = CHUNK / SR
AHEAD = 0.06                    # seconds of audio queued ahead of the speaker

IDLE_RPM = 1200.0               # the Chevy's lumpy cammed idle (measured from the recording)
REDLINE = 6500.0
LIMITER = 7000.0                # rev limiter cuts fuel here -> "brap brap" at the top
OVERREV_MAX = 9000.0            # a too-early downshift can drag the engine this far

# The sound is built from the real Chevy: every firing pulse of the idle recording is cut
# out (pitch marks at the exhaust pulses) and re-fired closer together as the revs rise.
# Each pulse keeps its own shape, so the engine keeps its hard, lumpy Chevy tone instead
# of turning into a sped-up tape; only the firing rate (rpm / 15 Hz for a V8) goes up.
GRAIN_CYCLES = 10               # 8-pulse firing cycles taken from the idle loop
GRAIN_SPEED = 0.2               # pulses play back rpm_ratio**this faster (a bit brighter up high)
GRAIN_SHRINK = 0.5              # and get rpm_ratio**this shorter, so they stay distinct
DELAY = 512                     # samples of lookahead so a pulse can start before its peak

# gears: km/h at the limiter in each gear (index 0 = neutral)
GEAR_TOP_KMH = [None, 50, 85, 120, 155, 190, 230]
RPM_PER_KMH = [0.0] + [LIMITER / v for v in GEAR_TOP_KMH[1:]]
LAUNCH_RPM = 1400.0             # revs the slipping clutch holds above idle when pulling away
SHIFT_TIME = 0.15               # clutch in, off the gas (or blipping it, on a downshift)

KEY_UP = 103                    # linux/input-event-codes.h
KEY_LEFT = 105
KEY_RIGHT = 106
EV_KEY = 1
EVENT_FMT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FMT)


# --------------------------------------------------------------------------- audio io
def decode(path):
    """Decode any audio file to a list of mono floats at SR using ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        check=True, stdout=subprocess.PIPE).stdout
    pcm = array.array("h")
    pcm.frombytes(raw[: len(raw) // 2 * 2])
    return [s / 32768.0 for s in pcm]


def pitch_marks(x, period):
    """Sample index of every exhaust pulse: the loudest point about one period after the last."""
    w, env, acc = 36, [0.0] * len(x), 0.0
    for i, v in enumerate(x):
        acc += abs(v) - (abs(x[i - w]) if i >= w else 0.0)
        env[i] = acc
    p = period
    marks = [max(range(int(p)), key=env.__getitem__)]
    while marks[-1] + 1.35 * p < len(x):
        m = max(range(int(marks[-1] + 0.7 * p), int(marks[-1] + 1.35 * p)), key=env.__getitem__)
        p = min(1.2 * period, max(0.8 * period, 0.85 * p + 0.15 * (m - marks[-1])))
        marks.append(m)
    return marks


def open_player():
    cmds = [
        ["pw-cat", "--playback", "--raw", "--format=s16", f"--rate={SR}", "--channels=1",
         "--latency=40ms", "-"],
        ["pacat", "--playback", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
         "--latency-msec=40", "--client-name=engine.py"],
        ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SR), "-c", "1", "--buffer-time=60000"],
    ]
    for cmd in cmds:
        if shutil.which(cmd[0]):
            # unbuffered: bursty writes make the sound server underrun and re-buffer for seconds
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            try:  # shrink the pipe so we never queue up more than ~85 ms of sound
                fcntl.fcntl(p.stdin.fileno(), 1031, 4096)  # F_SETPIPE_SZ
            except OSError:
                pass
            return p
    sys.exit("No audio player found (need pacat, pw-cat or aplay).")


# --------------------------------------------------------------------------- input
class Keys:
    """Tracks whether UP is held. Uses /dev/input (real press/release) when readable,
    otherwise falls back to guessing from terminal key-repeat."""

    def __init__(self, use_evdev=True):
        self.evdev = self._open_keyboards() if use_evdev else []
        self.held = False
        self.last_up = -1.0
        self.repeating = False
        self.quit = False
        self.shifts = 0           # pending gear changes: +1 per RIGHT press, -1 per LEFT

    def take_shifts(self):
        n, self.shifts = self.shifts, 0
        return n

    @staticmethod
    def _open_keyboards():
        fds = []
        try:
            blocks = open("/proc/bus/input/devices").read().split("\n\n")
        except OSError:
            return fds
        for b in blocks:
            m = re.search(r"H: Handlers=.*\bkbd\b.*?\b(event\d+)", b)
            if not m:
                continue
            try:
                fds.append(os.open("/dev/input/" + m.group(1), os.O_RDONLY | os.O_NONBLOCK))
            except OSError:
                pass
        return fds

    @property
    def mode(self):
        return "evdev (true key up/down)" if self.evdev else "terminal key-repeat"

    def poll(self, now):
        # terminal input: q / Esc / Ctrl-C to quit, arrows for fallback mode
        while select.select([sys.stdin], [], [], 0)[0]:
            data = os.read(sys.stdin.fileno(), 1024).decode(errors="ignore")
            ups = data.count("\x1b[A") + data.count("\x1bOA")
            if not self.evdev:
                self.shifts += data.count("\x1b[C") + data.count("\x1bOC")
                self.shifts -= data.count("\x1b[D") + data.count("\x1bOD")
            rest = re.sub(r"\x1b[\[O][A-D]", "", data)
            if "q" in rest.lower() or rest == "\x1b" or "\x03" in rest:
                self.quit = True
            if ups and not self.evdev:
                self.repeating = now - self.last_up < 0.2
                self.last_up = now
        if self.evdev:
            for fd in self.evdev:
                try:
                    buf = os.read(fd, EVENT_SIZE * 64)
                except BlockingIOError:
                    continue
                for off in range(0, len(buf) - EVENT_SIZE + 1, EVENT_SIZE):
                    _, _, typ, code, val = struct.unpack_from(EVENT_FMT, buf, off)
                    if typ == EV_KEY and code == KEY_UP:
                        self.held = val != 0
                    elif typ == EV_KEY and val == 1 and code in (KEY_LEFT, KEY_RIGHT):
                        self.shifts += 1 if code == KEY_RIGHT else -1
        else:
            # first press waits out the OS repeat delay; once repeats flow, release is detected fast
            window = 0.12 if self.repeating else 0.6
            self.held = now - self.last_up < window
        return self.held

    def close(self):
        for fd in self.evdev:
            os.close(fd)


# --------------------------------------------------------------------------- engine
class Engine:
    def __init__(self):
        idle = decode(os.path.join(SPLIT, "03_idle_loop.mp3"))
        marks = pitch_marks(idle, 0.0122 * SR)[: GRAIN_CYCLES * 8 + 2]
        # grain k spans the pulse before to the pulse after (for overlap-add at native rate)
        self.grains = [(marks[k], marks[k] - marks[k - 1], marks[k + 1] - marks[k])
                       for k in range(1, len(marks) - 1)]
        self.src = idle + [0.0] * 2
        self.rec_rpm = 15 * SR * (len(marks) - 1) / (marks[-1] - marks[0])
        self.mean_gap = (marks[-1] - marks[0]) / (len(marks) - 1)
        self.hann = [0.5 - 0.5 * math.cos(math.pi * i / 1024) for i in range(1025)]
        self.start_clip = decode(os.path.join(SPLIT, "02_start.mp3"))
        self.stop_clip = decode(os.path.join(SPLIT, "04_shutdown.mp3"))

        self.rpm = IDLE_RPM
        self.throttle = 0.0
        self.cut = 0.0            # seconds of rev-limiter fuel cut remaining
        self.acc = [0.0] * (DELAY + 4 * CHUNK)   # overlap-add buffer
        self.next_t = 0.0         # samples until the next pulse fires
        self.gi = 0               # which recorded pulse comes next
        self.tone = 0.0
        self.gear = 1             # sitting in 1st with the clutch slipping: hold UP to drive off
        self.kmh = 0.0
        self.shift_t = 0.0        # seconds of clutch-in left in the current shift
        self.shift_dir = 0
        self.grip = 30.0          # how hard the clutch pulls the engine to wheel speed
        self.note, self.note_t = "", 0.0
        self.pops = False
        self.prev_rpm = self.rpm
        self.prev = self._params()

        self.state = "starting"
        self.level = 0.0          # engine-loop level, ramps during start / stop
        self.clip = self.start_clip
        self.clip_pos = 0
        self.xfade_at = len(self.start_clip) - int(0.45 * SR)

    # ---- physics
    def step(self, pedal):
        if self.state != "running":
            pedal = 0.0
        if self.shift_t > 0:      # clutch is in: lift off, or blip to rev-match a downshift
            self.shift_t -= DT
            pedal = 0.8 if self.shift_dir < 0 else 0.0
        self.note_t = max(0.0, self.note_t - DT)
        self.throttle += (pedal - self.throttle) * min(1.0, DT * 14)
        wobble = 25 * math.sin(time.monotonic() * 7)
        if self.cut > 0:
            self.cut -= DT
            fuel = 0.0
        else:
            fuel = self.throttle
        ratio = RPM_PER_KMH[self.gear]
        engaged = ratio and self.shift_t <= 0 and self.state == "running"
        overdriven = False
        if engaged:
            # clutch out: the engine turns with the wheels (or slips while pulling away)
            wheel = self.kmh * ratio
            if wheel > OVERREV_MAX:           # engine can't spin faster: the rear tyres skid
                self.kmh -= 35 * DT
                wheel = OVERREV_MAX
            overdriven = wheel > LIMITER + 250
            # the slipping clutch lets the revs keep rising as the car picks up speed
            launch = IDLE_RPM + wobble + LAUNCH_RPM * fuel
            target = (launch ** 4 + wheel ** 4) ** 0.25
            self.rpm += (target - self.rpm) * min(1.0, DT * self.grip)
            self.grip = min(30.0, self.grip + DT * 60)
            mult = ratio / RPM_PER_KMH[1]     # lower gears multiply torque more
            torque = max(0.35, 1 - ((self.rpm - 4500) / 4500) ** 2)
            accel = 30 * fuel * torque
            if wheel > IDLE_RPM:              # engine braking once the clutch is fully in
                accel -= 8 * (1 - fuel) * self.rpm / LIMITER
            if self.rpm > LIMITER:            # over-revved: it brakes the car hard
                accel -= 25 * (self.rpm - LIMITER) / 1000
            self.kmh += accel * mult * DT
        else:
            if fuel > 0.02:
                accel = 7000 * (1 - 0.6 * self.rpm / LIMITER)
                self.rpm += fuel * accel * DT
            drop = (900 + 0.9 * (self.rpm - IDLE_RPM)) * (1 - fuel)
            self.rpm -= drop * DT
        self.kmh = max(0.0, self.kmh - (0.5 + 0.00012 * self.kmh ** 2) * DT)
        if self.state == "stopping":
            self.rpm -= 4000 * DT
        self.rpm = max(self.rpm, IDLE_RPM + wobble if self.state != "stopping" else 300)
        if overdriven:                        # wheels spin the engine past the limiter: fuel stays cut
            self.cut = max(self.cut, 2 * DT)
            self.rpm = min(self.rpm, OVERREV_MAX)
        elif self.rpm >= LIMITER:
            self.rpm = LIMITER - 350
            self.cut = 0.07

    def shift(self, d):
        """d = +1 upshift, -1 downshift. Any downshift goes in, even one that over-revs the engine."""
        g = self.gear + d
        if self.state != "running" or not 0 <= g < len(RPM_PER_KMH):
            return
        self.gear, self.shift_t, self.shift_dir, self.grip = g, SHIFT_TIME, d, 6.0

    def _params(self):
        norm = max(0.0, (self.rpm - IDLE_RPM) / (LIMITER - IDLE_RPM))
        load = self.throttle * (0.0 if self.cut > 0 else 1.0)
        overrun = (1 - self.throttle) * min(1.0, max(0.0, (self.rpm - IDLE_RPM) / 1500))
        gain = (1 - 0.15 * load * norm) * (1 - 0.3 * overrun)
        tone = 1 - 0.45 * overrun                 # off the gas: softer, duller burble
        drive = 1 + 2.0 * load * norm ** 0.7       # on the gas: harder and grittier
        return gain, tone, drive

    def _fire(self, at, rpm):
        """Overlap-add the next recorded pulse at buffer index `at`; returns samples to the next."""
        center, l1, l2 = self.grains[self.gi]
        self.gi = (self.gi + 1) % len(self.grains)
        r = max(1.0, rpm / self.rec_rpm)
        q, sh = r ** GRAIN_SPEED, r ** GRAIN_SHRINK
        amp = (sh / r) ** 0.8                      # pulses overlap more as they crowd together
        if self.cut > 0:
            amp *= 0.25                            # limiter fuel cut: hardly any bang
        elif self.pops and random.random() < 0.004:
            amp *= 2.2                             # unburnt fuel popping in the exhaust
        h1, h2 = l1 / sh, l2 / sh
        k1, k2 = 1024 / h1, 1024 / h2
        src, acc, hann = self.src, self.acc, self.hann
        base = int(at)
        for j in range(-int(h1), int(h2)):
            p = center + j * q
            i = int(p)
            v = src[i] + (src[i + 1] - src[i]) * (p - i)
            w = hann[int((j + h1) * k1)] if j < 0 else hann[1024 - int(j * k2)]
            acc[base + j] += v * w * amp
        # keep the lumpy cam rhythm at idle; it evens out as the revs rise
        irregular = r ** -0.7
        return (self.mean_gap + (l2 - self.mean_gap) * irregular) / r

    # ---- synthesis
    def render(self, n=CHUNK):
        g0, t0, d0 = self.prev
        g1, t1, d1 = self.prev = self._params()
        m0, m1 = self.prev_rpm, self.rpm
        self.prev_rpm = m1
        self.pops = self.throttle < 0.1 and self.rpm > 2800 and self.state == "running"
        while self.next_t < n:
            self.next_t += self._fire(DELAY + self.next_t, m0 + (m1 - m0) * self.next_t / n)
        self.next_t -= n

        lvl0 = self.level
        if self.state == "starting" and self.clip_pos + n > self.xfade_at:
            self.level = min(1.0, self.level + n / (0.45 * SR))
        elif self.state == "stopping":
            self.level = max(0.0, self.level - n / (0.3 * SR))
        lvl1 = self.level

        out = array.array("h", bytes(2 * n))
        acc, tanh, tone = self.acc, math.tanh, self.tone
        clip, cp = self.clip, self.clip_pos
        th0, th1 = tanh(d0), tanh(d1)
        for k in range(n):
            f = k / n
            tone += (acc[k] - tone) * (t0 + (t1 - t0) * f)
            d = d0 + (d1 - d0) * f
            x = tone
            if d > 1.001:
                x += (tanh(x * d) / (th0 + (th1 - th0) * f) - x) * min(1.0, d - 1)
            x *= (g0 + (g1 - g0) * f) * (lvl0 + (lvl1 - lvl0) * f)
            if clip is not None and cp < len(clip):
                x += clip[cp]
                cp += 1
            if x > 0.75:                           # soft knee instead of hard clipping
                x = 0.75 + 0.25 * tanh((x - 0.75) * 4)
            elif x < -0.75:
                x = -0.75 + 0.25 * tanh((x + 0.75) * 4)
            out[k] = int(x * 32000)
        self.acc = acc[n:] + [0.0] * n
        self.tone, self.clip_pos = tone, cp

        if self.state == "starting" and cp >= len(clip):
            self.state, self.clip, self.level = "running", None, 1.0
        elif self.state == "stopping" and cp >= len(clip):
            self.state = "off"
        return out

    def shut_off(self):
        if self.state in ("running", "starting"):
            self.state = "stopping"
            self.clip, self.clip_pos = self.stop_clip, 0


# --------------------------------------------------------------------------- ui
def draw(eng, pedal, mode):
    width = 40
    fill = int(width * min(1.0, eng.rpm / LIMITER))
    red_from = int(width * REDLINE / LIMITER)
    bar = ""
    for i in range(width):
        if i < fill:
            bar += ("\x1b[31m" if i >= red_from else "\x1b[33m" if i >= width * 0.55 else "\x1b[32m") + "█"
        else:
            bar += "\x1b[90m·"
    bar += "\x1b[0m"
    if eng.note_t > 0:
        tag = f"\x1b[1;35m{eng.note.upper()}\x1b[0m"
    elif eng.state == "starting":
        tag = "\x1b[36mCRANKING…\x1b[0m"
    elif eng.state == "stopping":
        tag = "\x1b[90mSHUTTING OFF\x1b[0m"
    elif eng.rpm > REDLINE or (pedal and eng.rpm > REDLINE - 800):
        tag = "\x1b[1;31mSCREAMING!!\x1b[0m"
    elif eng.shift_t > 0:
        tag = "\x1b[36mSHIFTING\x1b[0m"
    elif pedal:
        tag = "\x1b[33mACCELERATING\x1b[0m"
    else:
        tag = "\x1b[32mIDLE\x1b[0m"
    gear = eng.gear or "N"
    sys.stdout.write(f"\r  [{bar}] {eng.rpm:5.0f} rpm  \x1b[1m{gear}\x1b[0m  {eng.kmh:3.0f} km/h"
                     f"  gas:{'▮' if pedal else '▯'}  {tag}\x1b[K")
    sys.stdout.flush()


def live(use_evdev):
    eng = Engine()
    keys = Keys(use_evdev)
    player = open_player()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    sys.stdout.write("\x1b[?25l")
    print("\n  🚗  Engine simulator — hold ↑ UP for gas, → / ← to shift up / down, q to shut off")
    print(f"      input: {keys.mode}\n")
    last_draw = 0.0
    t0 = time.monotonic()
    written = 0
    try:
        while eng.state != "off":
            # pace by the wall clock: players like pacat happily buffer seconds of audio,
            # which would make the pedal lag. Stay only AHEAD seconds in front of playback.
            ahead = written / SR - (time.monotonic() - t0)
            if ahead > AHEAD:
                time.sleep(ahead - AHEAD)
            elif ahead < -0.1:  # fell behind (e.g. suspended) - resync instead of rushing
                t0 = time.monotonic() - written / SR
            now = time.monotonic()
            pedal = keys.poll(now)
            if keys.quit:
                eng.shut_off()
            for _ in range(abs(n := keys.take_shifts())):
                eng.shift(1 if n > 0 else -1)
            eng.step(1.0 if pedal else 0.0)
            player.stdin.write(eng.render().tobytes())
            written += CHUNK
            if now - last_draw > 0.05:
                draw(eng, pedal, keys.mode)
                last_draw = now
        player.stdin.write(bytes(SR // 5))
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
    """Offline render, e.g. "idle:1,gas:0.3,up,gas:3,down,idle:2,off" -> out_path."""
    eng = Engine()
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1",
                            "-i", "-", out_path], stdin=subprocess.PIPE)
    while eng.state == "starting":
        eng.step(0)
        enc.stdin.write(eng.render().tobytes())
    for part in script.split(","):
        name, _, secs = part.partition(":")
        if name == "off":
            eng.shut_off()
            while eng.state != "off":
                eng.step(0)
                enc.stdin.write(eng.render().tobytes())
            continue
        if name in ("up", "down"):
            eng.shift(1 if name == "up" else -1)
            continue
        for _ in range(int(float(secs) / DT)):
            eng.step(1.0 if name == "gas" else 0.0)
            enc.stdin.write(eng.render().tobytes())
    enc.stdin.close()
    enc.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-evdev", action="store_true",
                    help="don't read /dev/input; detect the UP key from terminal key-repeat instead")
    ap.add_argument("--render", nargs=2, metavar=("SCRIPT", "OUT"),
                    help='render offline, e.g. --render "idle:1,up,gas:3,up,gas:3,down,idle:2,off" demo.mp3')
    args = ap.parse_args()
    if args.render:
        render(*args.render)
    else:
        if not sys.stdin.isatty():
            sys.exit("Run this in a terminal.")
        live(not args.no_evdev)
