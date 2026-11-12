import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

cv2.setNumThreads(2)  # tiny images: fewer OpenCV threads means less scheduling overhead

# --- Dark Theme Palette ---
BG_COLOR = "#0f1117"
CARD_BG = "#161b22"
INPUT_BG = "#0d1117"
BORDER_COLOR = "#30363d"
TEXT_MAIN = "#f0f6fc"
TEXT_MUTED = "#8b949e"
ACCENT_GREEN = "#238636"
ACCENT_GREEN_HOVER = "#2ea043"
ACCENT_RED = "#da3633"
ACCENT_RED_HOVER = "#f85149"
ACCENT_BLUE = "#1f6feb"
ACCENT_BLUE_HOVER = "#388bfd"
BLUE_TEXT = "#58a6ff"
GREEN_TEXT = "#7ee787"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
LEGACY_SETTINGS_FILE = os.path.join(BASE_DIR, "hit_stats.json")  # old versions kept the ntfy topic here
LOG_FILE = os.path.join(BASE_DIR, "error_log.txt")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")

DEFAULT_SETTINGS = {"ntfy_server": "https://ntfy.sh", "ntfy_topic": ""}

# --- Detection tuning ---
CAM_W, CAM_H, CAM_FPS = 640, 480, 30
PROC_FPS = 15                     # frames are decoded/processed at this rate; the rest are only grabbed
ZONE = (0.35, 0.30, 0.65, 0.55)   # x1, y1, x2, y2 as fractions of the frame
PROC_SIZE = (160, 120)
MOTION_THRESHOLD = 350            # foreground pixels that count as motion
MIN_HIT_SECONDS = 0.5             # motion must last this long to count as a hit
GRACE_SECONDS = 0.25              # motion must stop this long to end an episode
WARMUP_FRAMES = 20                # processed frames used to settle the background model
RECONNECT_AFTER_FAILS = 10        # consecutive failed grabs before reconnecting the camera
RECONNECT_TRIES = 5
RECONNECT_DELAY = 2.0

# --- Clip settings ---
CLIP_FPS = PROC_FPS
PC_PRE, PC_POST = 3.0, 7.0        # 10 s clip saved on this PC
PHONE_PRE, PHONE_POST = 2.0, 3.0  # 5 s clip pushed to the phone
PHONE_WIDTH = 480                 # phone clip is downscaled to keep uploads small
MAX_ATTACH_BYTES = 14 * 1024 * 1024  # ntfy.sh caps attachments at 15 MB
MAX_SAVED_CLIPS = 40
CODECS = ("avc1", "H264", "mp4v")  # first one this OpenCV build can write wins

# --- UI tuning ---
UI_INTERVAL_MS = 40               # ~25 fps redraw while running
IDLE_INTERVAL_MS = 250            # slow poll while stopped
DISPLAY_W, DISPLAY_H = 760, 360
MAX_LOG_LINES = 200


def log_error(msg):
  try:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
      f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
  except Exception:
    pass


def load_settings():
  settings = dict(DEFAULT_SETTINGS)
  for path in (SETTINGS_FILE, LEGACY_SETTINGS_FILE):
    try:
      with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
      if isinstance(data, dict):
        for key in settings:
          if data.get(key):
            settings[key] = str(data[key])
        return settings
    except FileNotFoundError:
      continue
    except Exception as e:
      log_error(f"Load Settings Error ({os.path.basename(path)}): {e}")
  return settings


def save_settings(settings):
  try:
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
      json.dump({k: settings[k] for k in DEFAULT_SETTINGS}, f, indent=4)
    os.replace(tmp, SETTINGS_FILE)  # atomic: no corrupt file if killed mid-write
  except Exception as e:
    log_error(f"Save Settings Error: {e}")


def send_ntfy(server, topic, title, message, priority="default", tags="dash", attach=None):
  """Blocking push to ntfy. Returns (ok, detail). Call from a thread."""
  topic = topic.strip().strip("/")
  if not topic:
    return False, "No ntfy topic set."
  url = f"{server.rstrip('/')}/{urllib.parse.quote(topic, safe='')}"
  headers = {"Title": title, "Priority": priority, "Tags": tags}  # headers must stay ASCII
  if attach:
    with open(attach, "rb") as f:
      data = f.read()
    headers.update({
        "Filename": os.path.basename(attach),
        "Message": message,
        "Content-Type": "video/mp4",
    })
    method, timeout = "PUT", 90
  else:
    data, method, timeout = message.encode("utf-8"), "POST", 8
  req = urllib.request.Request(url, data=data, method=method, headers=headers)
  try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return (200 <= resp.status < 300), f"HTTP {resp.status}"
  except urllib.error.HTTPError as e:
    return False, f"HTTP {e.code}"
  except Exception as e:
    return False, str(e)


def open_camera():
  backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
  for backend in backends:
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
      cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # less USB bandwidth / CPU
      cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
      cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
      cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
      cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
      return cap
    cap.release()
  return None


def open_writer(path, fps, size):
  for code in CODECS:
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), fps, size)
    if w.isOpened():
      return w
    w.release()
  return None


class TimedWriter:
  """VideoWriter wrapper that keeps real-time duration even if the camera runs slower/faster."""

  def __init__(self, writer, fps, size=None):
    self.writer, self.fps, self.size = writer, fps, size
    self.t0 = None
    self.count = 0

  def write(self, t, frame):
    if self.t0 is None:
      self.t0 = t
    target = int((t - self.t0) * self.fps) + 1
    n = target - self.count
    if n <= 0:
      return
    if self.size:
      frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
    for _ in range(n):
      self.writer.write(frame)
    self.count = target

  def close(self):
    try:
      self.writer.release()
    except Exception:
      pass


class ClipJob(threading.Thread):
  """Encodes the 10 s PC clip and the 5 s phone clip off the detection thread."""

  def __init__(self, pre_frames, trigger_t, cfg, msgs):
    super().__init__()  # non-daemon so a clip in progress is finalized on exit
    self.pre = pre_frames
    self.trigger_t = trigger_t
    self.cfg, self.msgs = cfg, msgs
    self.q = queue.Queue()
    self.pc = self.phone = None
    self.phone_closed = False
    self.opened = False
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.pc_path = os.path.join(CLIPS_DIR, f"hit_{stamp}.mp4")
    self.phone_path = os.path.join(CLIPS_DIR, f"phone_{stamp}.mp4")

  def add(self, t, frame):
    self.q.put((t, frame))

  def finish(self):
    self.q.put(None)

  def log(self, text):
    self.msgs.put(("log", text))

  def _open(self, frame):
    self.opened = True
    os.makedirs(CLIPS_DIR, exist_ok=True)
    h, w = frame.shape[:2]
    pw = PHONE_WIDTH - PHONE_WIDTH % 2
    ph = int(round(pw * h / w / 2)) * 2
    pc_w = open_writer(self.pc_path, CLIP_FPS, (w, h))
    phone_w = open_writer(self.phone_path, CLIP_FPS, (pw, ph))
    self.pc = TimedWriter(pc_w, CLIP_FPS) if pc_w else None
    self.phone = TimedWriter(phone_w, CLIP_FPS, (pw, ph)) if phone_w else None
    if not self.pc and not self.phone:
      self.log("No video codec available - clips disabled.")
    if not self.phone:
      self._push_text("Hit detected (video unavailable).")

  def _process(self, t, frame):
    if not self.opened:
      self._open(frame)
    if self.pc:
      self.pc.write(t, frame)
    if self.phone and not self.phone_closed:
      if t > self.trigger_t + PHONE_POST:
        self._close_phone()
      elif t >= self.trigger_t - PHONE_PRE:
        self.phone.write(t, frame)

  def _close_phone(self):
    self.phone_closed = True
    self.phone.close()
    threading.Thread(target=self._upload, daemon=True).start()

  def _push_text(self, message):
    ok, detail = send_ntfy(self.cfg["ntfy_server"], self.cfg["ntfy_topic"], "Motion Tracker",
                           message, priority="high")
    self.log("Phone alert sent." if ok else f"Phone alert FAILED: {detail}")

  def _upload(self):
    try:
      if not self.cfg["ntfy_topic"].strip():
        self.log("ntfy: no topic set, phone clip skipped.")
        return
      stamp = f"{datetime.now():%H:%M:%S}"
      if os.path.getsize(self.phone_path) > MAX_ATTACH_BYTES:
        self._push_text(f"Hit detected at {stamp} (clip too large to send).")
        return
      ok, detail = send_ntfy(self.cfg["ntfy_server"], self.cfg["ntfy_topic"], "Motion Tracker",
                             f"Hit detected at {stamp}", priority="high", attach=self.phone_path)
      self.log("5s clip sent to phone." if ok else f"Phone clip FAILED: {detail}")
      if not ok:
        log_error(f"ntfy clip error: {detail}")
    except Exception as e:
      log_error(f"Upload Error: {e}")
      self.log(f"Phone clip FAILED: {e}")
    finally:
      try:
        os.remove(self.phone_path)
      except OSError:
        pass

  def run(self):
    try:
      for t, frame in self.pre:
        self._process(t, frame)
      while True:
        item = self.q.get()
        if item is None:
          break
        self._process(*item)
      if self.phone and not self.phone_closed:
        self._close_phone()
      if self.pc:
        self.pc.close()
        self.log(f"10s clip saved: clips/{os.path.basename(self.pc_path)}")
        self._prune()
    except Exception as e:
      log_error(f"ClipJob Error: {e}")
      self.log(f"Clip error: {e}")

  @staticmethod
  def _prune():
    try:
      files = sorted(
          (os.path.join(CLIPS_DIR, f) for f in os.listdir(CLIPS_DIR)
           if f.startswith("hit_") and f.endswith(".mp4")),
          key=os.path.getmtime, reverse=True)
      for old in files[MAX_SAVED_CLIPS:]:
        os.remove(old)
    except OSError:
      pass


class MotionWorker(threading.Thread):
  """Grabs frames, detects motion, builds previews and triggers clips.

  Frames outside the PROC_FPS cadence are only grabbed (no decode), which saves most of the CPU
  on MJPG cameras. Previews are skipped entirely while the window is minimized.
  """

  def __init__(self, cap, msgs, cfg):
    super().__init__(daemon=True)
    self.cap, self.msgs, self.cfg = cap, msgs, cfg
    self.stop_evt = threading.Event()
    self.show_preview = threading.Event()
    self.show_preview.set()
    self.lock = threading.Lock()
    self.latest = None  # (preview_id, rgb_image)

  def stop(self):
    self.stop_evt.set()

  def get_latest(self):
    with self.lock:
      return self.latest

  def _reconnect(self):
    self.msgs.put(("log", "Camera lost - reconnecting..."))
    try:
      self.cap.release()
    except Exception:
      pass
    for _ in range(RECONNECT_TRIES):
      if self.stop_evt.wait(RECONNECT_DELAY):
        return False
      cap = open_camera()
      if cap is not None:
        self.cap = cap
        self.msgs.put(("log", "Camera reconnected. Recalibrating..."))
        return True
    self.msgs.put(("error", "Camera lost and could not be reconnected."))
    return False

  @staticmethod
  def _new_bg():
    return cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=30, detectShadows=False)

  def run(self):
    bg = self._new_bg()
    ring = deque(maxlen=int(PC_PRE * PROC_FPS))
    gap = 1.0 / PROC_FPS - 0.004
    last_proc = 0.0
    in_hit = triggered = False
    start = last_motion = 0.0
    job = None
    job_start = 0.0
    n = 0          # processed frames
    fails = 0
    scale = None
    try:
      while not self.stop_evt.is_set():
        if not self.cap.grab():
          fails += 1
          if fails >= RECONNECT_AFTER_FAILS:
            if not self._reconnect():
              break
            bg, n, fails, in_hit, triggered = self._new_bg(), 0, 0, False, False
          else:
            time.sleep(0.05)
          continue
        fails = 0
        now = time.monotonic()
        if now - last_proc < gap:
          continue  # grabbed only; skip decoding
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
          continue
        last_proc = now
        n += 1

        # Clip feed: rolling pre-roll buffer, plus the active job if any
        ring.append((now, frame))
        if job:
          job.add(now, frame)

        h, w = frame.shape[:2]
        x1, y1 = int(w * ZONE[0]), int(h * ZONE[1])
        x2, y2 = int(w * ZONE[2]), int(h * ZONE[3])
        roi = frame[y1:y2, x1:x2]

        if roi.size:
          small = cv2.resize(roi, PROC_SIZE, interpolation=cv2.INTER_AREA)
          gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
          motion = cv2.countNonZero(bg.apply(gray))  # mask is strictly 0/255 (no shadows)

          if n == WARMUP_FRAMES + 1:
            self.msgs.put(("live", None))

          if n > WARMUP_FRAMES:
            if motion > MOTION_THRESHOLD:
              last_motion = now
              if not in_hit:
                in_hit, triggered, start = True, False, now
                self.msgs.put(("log", "💨 Motion detected in zone..."))
              if not triggered and now - start >= MIN_HIT_SECONDS:
                triggered = True
                if job is None:
                  self.msgs.put(("log", "✅ Hit confirmed - recording clips..."))
                  job = ClipJob(list(ring), now, self.cfg, self.msgs)
                  job.start()
                  job_start = now
            elif in_hit and now - last_motion >= GRACE_SECONDS:
              in_hit = False

        if job and now - job_start >= PC_POST:
          job.finish()
          job = None

        # Preview is built here (not on the UI thread) and skipped when the window is hidden
        if self.show_preview.is_set():
          if scale is None:
            scale = min(DISPLAY_W / w, DISPLAY_H / h)
          img = cv2.resize(frame, (int(w * scale), int(h * scale)))
          if job:
            cv2.circle(img, (20, 20), 7, (54, 51, 218), -1)  # REC dot
          with self.lock:
            self.latest = (n, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    except Exception as e:
      log_error(f"Worker Error: {e}")
      self.msgs.put(("error", str(e)))
    finally:
      if job:
        job.finish()
      try:
        self.cap.release()
      except Exception:
        pass


class MotionTrackerApp:

  def __init__(self, root):
    self.root = root
    self.root.title("Motion Tracker")
    self.root.geometry("850x860")
    self.root.configure(bg=BG_COLOR)
    self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    self.worker = None
    self.msgs = queue.Queue()  # worker/clip/ntfy threads -> UI
    self.cfg = load_settings()
    self.is_running = False
    self.last_preview_id = -1
    self._photo = None  # reused PhotoImage: paste() avoids reallocating every frame
    self._tick_job = None

    self._build_ui()
    self.update_clock()
    self._tick()
    self.log_event("Application initialized successfully.")
    if not self.cfg["ntfy_topic"]:
      self.log_event("Set an ntfy topic above to get phone alerts.")

  # ---------- UI construction ----------
  def _card(self, parent, padx=15, pady=10, fill="x", expand=False, bottom=10):
    f = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER_COLOR,
                 highlightthickness=1, padx=padx, pady=pady)
    f.pack(fill=fill, expand=expand, pady=(0, bottom))
    return f

  def _build_ui(self):
    main = tk.Frame(self.root, bg=BG_COLOR)
    main.pack(fill="both", expand=True, padx=20, pady=20)

    # Clock bar
    header = self._card(main, pady=10)
    self.clock_var = tk.StringVar()
    tk.Label(header, textvariable=self.clock_var, bg=CARD_BG, fg=GREEN_TEXT,
             font=("Consolas", 10, "bold")).pack(side="right")

    # ntfy settings
    ntfy_card = self._card(main, pady=12)
    tk.Label(ntfy_card, text="NTFY TOPIC (PRESS ENTER TO SAVE):", bg=CARD_BG, fg=TEXT_MUTED,
             font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 5))
    row = tk.Frame(ntfy_card, bg=CARD_BG)
    row.pack(fill="x")
    self.ntfy_topic_var = tk.StringVar(value=self.cfg["ntfy_topic"])
    entry = tk.Entry(row, textvariable=self.ntfy_topic_var, bg=INPUT_BG, fg=BLUE_TEXT,
                     insertbackground=TEXT_MAIN, font=("Segoe UI", 12, "bold"), bd=0,
                     relief="flat", highlightbackground=BORDER_COLOR, highlightthickness=1)
    entry.pack(side="left", fill="x", expand=True, ipady=8, ipadx=10)
    entry.bind("<Return>", self.on_ntfy_submit)
    entry.bind("<FocusOut>", self.on_ntfy_submit)
    tk.Button(row, text="🔔 TEST", bg=ACCENT_BLUE, fg=TEXT_MAIN,
              font=("Segoe UI", 10, "bold"), activebackground=ACCENT_BLUE_HOVER,
              activeforeground=TEXT_MAIN, bd=0, padx=15, pady=6, cursor="hand2",
              command=self.test_ntfy).pack(side="right", padx=(10, 0))

    # Controls
    control = self._card(main)
    self.toggle_btn = tk.Button(
        control, text="▶  START DETECTION", bg=ACCENT_GREEN, fg=TEXT_MAIN,
        font=("Segoe UI", 10, "bold"), activebackground=ACCENT_GREEN_HOVER,
        activeforeground=TEXT_MAIN, bd=0, padx=15, pady=6, cursor="hand2",
        command=self.toggle_detection)
    self.toggle_btn.pack(side="left")

    # Camera feed
    monitor = self._card(main, padx=12, pady=12, fill="both", expand=True)
    mh = tk.Frame(monitor, bg=CARD_BG)
    mh.pack(fill="x", pady=(0, 6))
    tk.Label(mh, text="LIVE CAMERA FEED", bg=CARD_BG, fg=TEXT_MUTED,
             font=("Segoe UI", 9, "bold")).pack(side="left")
    self.status_var = tk.StringVar(value="STATUS: STANDBY")
    tk.Label(mh, textvariable=self.status_var, bg=CARD_BG, fg=ACCENT_BLUE,
             font=("Segoe UI", 9, "bold")).pack(side="right")
    holder = tk.Frame(monitor, bg="#000000", height=DISPLAY_H)
    holder.pack(fill="both", expand=True)
    holder.pack_propagate(False)
    self.video_label = tk.Label(holder, bg="#000000")
    self.video_label.pack(expand=True)

    # Log
    log_card = self._card(main, padx=10, pady=8, bottom=0)
    tk.Label(log_card, text="ACTIVITY EVENT LOG", bg=CARD_BG, fg=TEXT_MUTED,
             font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 4))
    self.log_box = tk.Text(log_card, height=5, bg=INPUT_BG, fg=GREEN_TEXT,
                           font=("Consolas", 8), bd=0, highlightthickness=0, state="disabled")
    self.log_box.pack(fill="x")

  # ---------- Logging / settings ----------
  def log_event(self, message):
    try:
      self.log_box.config(state="normal")
      self.log_box.insert(tk.END, f"[{datetime.now():%H:%M:%S}] {message}\n")
      lines = int(self.log_box.index("end-1c").split(".")[0])
      if lines > MAX_LOG_LINES:
        self.log_box.delete("1.0", f"{lines - MAX_LOG_LINES + 1}.0")
      self.log_box.see(tk.END)
      self.log_box.config(state="disabled")
    except Exception as e:
      log_error(f"Log Event Error: {e}")

  def _sync_settings(self):
    self.cfg["ntfy_topic"] = self.ntfy_topic_var.get().strip()
    save_settings(self.cfg)

  def on_ntfy_submit(self, event=None):
    self._sync_settings()
    if event is not None and event.keysym == "Return":
      self.log_event(f"ntfy topic saved: '{self.cfg['ntfy_topic']}'")

  def test_ntfy(self):
    self._sync_settings()
    if not self.cfg["ntfy_topic"]:
      self.log_event("ntfy: enter a topic first.")
      return
    cfg = dict(self.cfg)

    def _run():
      ok, detail = send_ntfy(cfg["ntfy_server"], cfg["ntfy_topic"], "Motion Tracker",
                             "Test alert - notifications are working.", tags="bell")
      self.msgs.put(("log", "ntfy test sent." if ok else f"ntfy test FAILED: {detail}"))

    threading.Thread(target=_run, daemon=True).start()

  def update_clock(self):
    self.clock_var.set(datetime.now().strftime("%A, %b %d %Y — %H:%M:%S"))
    self.root.after(1000, self.update_clock)

  # ---------- Detection control ----------
  def toggle_detection(self):
    if self.is_running:
      self.stop_detection()
    else:
      self.start_detection()

  def start_detection(self):
    try:
      cap = open_camera()
      if cap is None:
        raise RuntimeError("Could not open camera stream.")
    except Exception as e:
      log_error(f"Camera Start Error: {e}")
      messagebox.showerror("Camera Error", f"Failed to open camera:\n{e}")
      return
    self._sync_settings()
    self.worker = MotionWorker(cap, self.msgs, self.cfg)
    self.worker.start()
    self.is_running = True
    self.last_preview_id = -1
    self.toggle_btn.config(text="⏹  STOP DETECTION", bg=ACCENT_RED,
                           activebackground=ACCENT_RED_HOVER)
    self.status_var.set("STATUS: CALIBRATING")
    self.log_event("Camera feed active. Calibrating background...")
    self._reschedule(0)

  def stop_detection(self, reason="Detection paused."):
    self.is_running = False
    if self.worker:
      self.worker.stop()
      self.worker.join(timeout=3)
      self.worker = None
    self.toggle_btn.config(text="▶  START DETECTION", bg=ACCENT_GREEN,
                           activebackground=ACCENT_GREEN_HOVER)
    self.status_var.set("STATUS: STANDBY")
    self.video_label.config(image="")
    self._photo = None
    self.log_event(reason)

  # ---------- UI loop (adaptive: fast while running, slow when idle) ----------
  def _reschedule(self, delay):
    if self._tick_job:
      self.root.after_cancel(self._tick_job)
    self._tick_job = self.root.after(delay, self._tick)

  def _tick(self):
    try:
      self._drain()
      if self.is_running and self.worker:
        visible = self.root.state() != "iconic"
        if visible:
          self.worker.show_preview.set()
          self._draw()
        else:
          self.worker.show_preview.clear()  # minimized: skip all preview work
    except Exception as e:
      log_error(f"UI Tick Error: {e}")
    self._tick_job = self.root.after(
        UI_INTERVAL_MS if self.is_running else IDLE_INTERVAL_MS, self._tick)

  def _drain(self):
    try:
      while True:
        kind, val = self.msgs.get_nowait()
        if kind == "log":
          self.log_event(val)
        elif kind == "live":
          self.status_var.set("STATUS: LIVE")
          self.log_event("Motion detection online.")
        elif kind == "error":
          if self.is_running:
            self.stop_detection("Camera error - detection stopped.")
            messagebox.showerror("Camera Error", str(val))
          return
    except queue.Empty:
      pass

  def _draw(self):
    latest = self.worker.get_latest()
    if latest is None or latest[0] == self.last_preview_id:
      return
    self.last_preview_id = latest[0]
    img = Image.fromarray(latest[1])
    if self._photo is None or (self._photo.width(), self._photo.height()) != img.size:
      self._photo = ImageTk.PhotoImage(img)
      self.video_label.config(image=self._photo)
    else:
      self._photo.paste(img)

  # ---------- Shutdown ----------
  def on_closing(self):
    try:
      if self.is_running:
        self.stop_detection("Shutting down.")
      self._sync_settings()
    finally:
      self.root.destroy()


def main():
  root = tk.Tk()
  MotionTrackerApp(root)
  root.mainloop()


if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    log_error(f"Fatal: {e}")
    raise