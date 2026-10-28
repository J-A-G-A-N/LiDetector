#!/usr/bin/env python3
"""
Live audio spectrogram: ffmpeg (capture) -> numpy (FFT) -> Flask (SSE) -> browser canvas.

Install:   pip install flask numpy      (and install ffmpeg itself)

Run examples:
  Linux (PulseAudio/PipeWire): python spectrogram_app.py --format pulse --device default
  Linux (ALSA):                python spectrogram_app.py --format alsa --device hw:0,0
  macOS:                       python spectrogram_app.py --format avfoundation --device ":0"
  Windows:                     python spectrogram_app.py --format dshow --device "audio=Microphone (Realtek Audio)"

List devices:
  Linux:   pactl list short sources      |  arecord -l
  macOS:   ffmpeg -f avfoundation -list_devices true -i ""
  Windows: ffmpeg -list_devices true -f dshow -i dummy

Then open http://localhost:5000
"""
import argparse
import base64
import platform
import queue
import subprocess
import threading

import numpy as np
from flask import Flask, Response, render_template_string

# ----------------------------- configuration ------------------------------
FFT_SIZE = 1024          # window length (samples) -> FFT_SIZE//2 + 1 frequency bins
HOP = 512                # new samples per spectrogram column
DB_MIN, DB_MAX = -100.0, -10.0   # dB range mapped to colour scale (re. full scale)

app = Flask(__name__)
clients = []                     # one queue per connected browser
clients_lock = threading.Lock()
config = {"rate": 44100}


# ------------------------------ audio capture -----------------------------
def build_ffmpeg_cmd(fmt, device, rate):
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", fmt, "-i", device,
        "-ac", "1",              # mono
        "-ar", str(rate),        # resample
        "-f", "s16le",           # raw 16-bit little-endian PCM
        "-",                     # to stdout
    ]


def broadcast(msg):
    with clients_lock:
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:   # slow client: drop frame
                pass


def capture_loop(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    window = np.hanning(FFT_SIZE).astype(np.float32)
    norm = window.sum() / 2.0    # so a full-scale sine ~ 0 dB
    buf = np.zeros(FFT_SIZE, dtype=np.float32)
    bytes_per_hop = HOP * 2

    while True:
        raw = proc.stdout.read(bytes_per_hop)
        if len(raw) < bytes_per_hop:
            err = proc.stderr.read().decode(errors="replace")
            print("ffmpeg stopped.\n" + err)
            broadcast("error")
            return

        new = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        buf = np.concatenate((buf[HOP:], new))

        mag = np.abs(np.fft.rfft(buf * window)) / norm
        db = 20.0 * np.log10(mag + 1e-10)
        col = np.clip((db - DB_MIN) / (DB_MAX - DB_MIN), 0, 1)
        col = (col * 255).astype(np.uint8)
        broadcast(base64.b64encode(col.tobytes()).decode())


# --------------------------------- routes ---------------------------------
@app.route("/")
def index():
    return render_template_string(
        PAGE, bins=FFT_SIZE // 2 + 1, rate=config["rate"]
    )


@app.route("/stream")
def stream():
    q = queue.Queue(maxsize=200)
    with clients_lock:
        clients.append(q)

    def gen():
        try:
            while True:
                try:
                    yield f"data: {q.get(timeout=15)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with clients_lock:
                if q in clients:
                    clients.remove(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Live Spectrogram</title>
<style>
  body { margin:0; background:#111; color:#ddd; font-family:system-ui,sans-serif; }
  header { padding:12px 20px; display:flex; gap:20px; align-items:center; }
  h1 { font-size:18px; margin:0; }
  #status { font-size:13px; opacity:.7; }
  #wrap { position:relative; margin:0 20px 20px; }
  canvas { width:100%; height:70vh; background:#000; image-rendering:pixelated; display:block; }
  #axis { position:absolute; left:4px; top:0; bottom:0; display:flex;
          flex-direction:column; justify-content:space-between;
          font-size:11px; color:#fff; text-shadow:0 0 3px #000; pointer-events:none; }
</style>
</head>
<body>
<header>
  <h1>Live Spectrogram</h1>
  <span id="status">connecting…</span>
</header>
<div id="wrap">
  <canvas id="c" width="1000" height="{{ bins }}"></canvas>
  <div id="axis"></div>
</div>

<script>
const BINS = {{ bins }}, RATE = {{ rate }};
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const W = canvas.width, H = canvas.height;
ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);

// y-axis labels (linear frequency, top = Nyquist)
const axis = document.getElementById("axis");
for (let i = 4; i >= 0; i--) {
  const s = document.createElement("span");
  s.textContent = (RATE / 2 * i / 4 / 1000).toFixed(1) + " kHz";
  axis.appendChild(s);
}

// colour map (black -> purple -> red -> orange -> yellow -> white)
const stops = [[0,0,0],[40,10,90],[150,30,100],[230,80,30],[255,200,40],[255,255,220]];
const lut = new Uint8ClampedArray(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255 * (stops.length - 1), k = Math.min(Math.floor(t), stops.length - 2), f = t - k;
  for (let c = 0; c < 3; c++) lut[i*3+c] = stops[k][c] + (stops[k+1][c] - stops[k][c]) * f;
}

const colImg = ctx.createImageData(1, H);

function drawColumn(bytes) {
  // scroll left by 1 px, then paint new column at right edge
  ctx.drawImage(canvas, -1, 0);
  for (let y = 0; y < H; y++) {
    const v = bytes[H - 1 - y];            // low freq at bottom
    colImg.data[y*4]   = lut[v*3];
    colImg.data[y*4+1] = lut[v*3+1];
    colImg.data[y*4+2] = lut[v*3+2];
    colImg.data[y*4+3] = 255;
  }
  ctx.putImageData(colImg, W - 1, 0);
}

const status = document.getElementById("status");
const es = new EventSource("/stream");
es.onopen = () => status.textContent = "live";
es.onerror = () => status.textContent = "disconnected";
es.onmessage = (e) => {
  if (e.data === "error") { status.textContent = "ffmpeg stopped (see server log)"; return; }
  const bin = atob(e.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  drawColumn(bytes);
};
</script>
</body>
</html>
"""


# ---------------------------------- main ----------------------------------
def default_input():
    system = platform.system()
    if system == "Linux":
        return "pulse", "default"
    if system == "Darwin":
        return "avfoundation", ":0"
    return "dshow", None


if __name__ == "__main__":
    d_fmt, d_dev = default_input()
    p = argparse.ArgumentParser(description="Live spectrogram web app")
    p.add_argument("--format", default=d_fmt, help="ffmpeg input format (pulse, alsa, avfoundation, dshow)")
    p.add_argument("--device", default=d_dev, help="audio device / port name")
    p.add_argument("--rate", type=int, default=44100, help="sample rate")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    if not args.device:
        p.error('--device is required on Windows, e.g. --device "audio=Microphone (Realtek Audio)"')

    config["rate"] = args.rate
    cmd = build_ffmpeg_cmd(args.format, args.device, args.rate)
    print("Running:", " ".join(cmd))

    threading.Thread(target=capture_loop, args=(cmd,), daemon=True).start()
    app.run(host=args.host, port=args.port, threaded=True)
