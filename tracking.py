"""
tracking.py - Real-time multi-model target tracking for a robot.

Examples (run inside S:\\MODEL\\Object):
    python tracking.py --list                    show trackable classes
    python tracking.py airpods                   your trained AirPods model
    python tracking.py bottle                    one COCO class
    python tracking.py "bottle,cup,phone"        several objects at once
    python tracking.py all                       AirPods + all 80 COCO classes
    python tracking.py "red mug"                 free text (YOLO-World)
    python tracking.py all --source clips/AirPods_4.mp4 --log test4.csv
    python tracking.py bottle --udp 127.0.0.1:9000

Keys: q quit | r release lock | n next object | s snapshot | + / - confidence
"""
import argparse, csv, json, socket, sys, threading, time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO, YOLOWorld

AIR_WORDS = {"airpods", "airpod", "airbuds", "earbuds", "air pods"}
ALL_WORDS = {"all", "any", "everything"}
ALIASES = {"phone": "cell phone", "mobile": "cell phone", "mobile phone": "cell phone",
           "smartphone": "cell phone", "monitor": "tv", "screen": "tv", "mug": "cup",
           "glass": "wine glass", "plant": "potted plant", "table": "dining table",
           "sofa": "couch", "ball": "sports ball", "teddy": "teddy bear"}
AIR_OFFSET, WORLD_OFFSET = 100000, 200000      # keep IDs of different models apart


# ======================================================================
# 1. ARGUMENTS
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Robot target tracker")
    p.add_argument("target", nargs="?", default="bottle",
                   help="airpods | all | COCO name | 'a,b,c' | free text")
    p.add_argument("--list", action="store_true", help="print trackable classes and exit")
    p.add_argument("--source", default="0", help="camera index or video file path")
    p.add_argument("--air-model", default="airpods_runs/v2/weights/best.pt")
    p.add_argument("--air-class", type=int, default=None,
                   help="AirPods class id (None for the 1-class model, 10 for multi_v1)")
    p.add_argument("--coco-model", default="yolo11s.pt")
    p.add_argument("--world-model", default="yolov8s-worldv2.pt")
    p.add_argument("--conf", type=float, default=None, help="override confidence")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--max-lost", type=int, default=30, help="frames to coast before unlocking")
    p.add_argument("--reacquire-radius", type=float, default=0.25)
    p.add_argument("--smooth", type=float, default=0.4, help="EMA 0..1 (higher = faster)")
    p.add_argument("--deadzone", type=float, default=0.05)
    p.add_argument("--udp", default=None, help="host:port to send JSON, e.g. 127.0.0.1:9000")
    p.add_argument("--log", default=None, help="CSV file to log every frame")
    p.add_argument("--no-show", action="store_true", help="run without a window")
    return p.parse_args()


def print_classes(args):
    coco = YOLO(args.coco_model)
    print(f"\nCOCO classes ({len(coco.names)}):")
    print(", ".join(coco.names.values()))
    if Path(args.air_model).exists():
        print("\nAirPods model classes:", YOLO(args.air_model).names)
    else:
        print(f"\nAirPods model NOT found at: {args.air_model}")
    print("\nSpecial words: all, airpods")
    print("Aliases:", ", ".join(f"{k}->{v}" for k, v in ALIASES.items()))
    print("Anything else is sent to YOLO-World as free text.\n")


# ======================================================================
# 2. FRAME SOURCE (threaded camera: always the newest frame)
# ======================================================================
class FrameSource:
    def __init__(self, source, width, height):
        self.is_camera = str(source).isdigit()
        src = int(source) if self.is_camera else source
        if self.is_camera and sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)
        if self.is_camera:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        self._lock = threading.Lock()
        self._frame, self._ts, self._count, self._seen = None, 0.0, 0, 0
        self._running = True
        if self.is_camera:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._ts = frame, time.time()
                self._count += 1

    def read(self, timeout=1.0):
        """Return (ok, frame, capture_timestamp)."""
        if not self.is_camera:
            ok, frame = self.cap.read()
            return ok, frame, time.time()
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._count != self._seen and self._frame is not None:
                    self._seen = self._count
                    return True, self._frame.copy(), self._ts
            time.sleep(0.001)
        return False, None, 0.0

    def release(self):
        self._running = False
        time.sleep(0.05)
        self.cap.release()


# ======================================================================
# 3. DETECTOR BRANCHES
# ======================================================================
@dataclass
class Branch:
    model: object
    classes: list
    conf: float
    name: str
    id_offset: int = 0
    fixed_label: str = None


def build_branches(args):
    raw = args.target.lower().strip()
    words = [ALIASES.get(w.strip(), w.strip()) for w in raw.split(",") if w.strip()]
    branches = []

    def air_branch():
        if not Path(args.air_model).exists():
            return None
        cls = [args.air_class] if args.air_class is not None else None
        return Branch(YOLO(args.air_model), cls, args.conf or 0.5,
                      "airpods", AIR_OFFSET, "AirPods")

    # ---- "all": every COCO class + AirPods ----
    if len(words) == 1 and words[0] in ALL_WORDS:
        branches.append(Branch(YOLO(args.coco_model), None, args.conf or 0.4, "coco-all"))
        b = air_branch()
        if b: branches.append(b)
        else: print(f"WARNING: AirPods model not found ({args.air_model}); COCO only.")
        return branches

    coco = None
    coco_ids, world_words, want_air = [], [], False
    for w in words:
        if w in AIR_WORDS:
            want_air = True
            continue
        if coco is None:
            coco = YOLO(args.coco_model)
        names = {v.lower(): k for k, v in coco.names.items()}
        if w in names:
            coco_ids.append(names[w])
        else:
            world_words.append(w)

    if want_air:
        b = air_branch()
        if b is None:
            raise FileNotFoundError(f"AirPods model not found: {args.air_model}\n"
                                    f"Find it with:  dir -Recurse -Filter best.pt")
        branches.append(b)
    if coco_ids:
        branches.append(Branch(coco, coco_ids, args.conf or 0.4, "coco"))
    if world_words:
        world = YOLOWorld(args.world_model)
        world.set_classes(world_words)
        branches.append(Branch(world, None, args.conf or 0.25, "yolo-world", WORLD_OFFSET))
    if not branches:
        raise ValueError("No valid target. Try: python tracking.py --list")
    return branches


# ======================================================================
# 4. DETECTIONS + TARGET SELECTION
# ======================================================================
@dataclass
class Det:
    tid: int
    conf: float
    cx: float
    cy: float
    w: float
    h: float
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""


def extract_detections(res, branch):
    dets = []
    if res.boxes is None or res.boxes.id is None:
        return dets
    ids = res.boxes.id.int().tolist()
    confs = res.boxes.conf.tolist()
    clss = res.boxes.cls.int().tolist()
    xywh, xyxy = res.boxes.xywh.tolist(), res.boxes.xyxy.tolist()
    for i, tid in enumerate(ids):
        label = branch.fixed_label or res.names[clss[i]]
        dets.append(Det(tid + branch.id_offset, confs[i], *xywh[i], *xyxy[i], label))
    return dets


class TargetSelector:
    """Follows ONE target. Coasts when lost, and re-acquires the nearest box
    with the same label after a tracker ID switch."""

    def __init__(self, max_lost, radius, fw, fh):
        self.max_lost, self.radius, self.fw, self.fh = max_lost, radius, fw, fh
        self.reset()

    def reset(self):
        self.locked_id, self.lost, self.last_xy, self.label = None, 0, None, None

    def _dist(self, d):
        dx = (d.cx - self.last_xy[0]) / self.fw
        dy = (d.cy - self.last_xy[1]) / self.fh
        return (dx * dx + dy * dy) ** 0.5

    def lock(self, d):
        self.locked_id, self.lost = d.tid, 0
        self.last_xy, self.label = (d.cx, d.cy), d.label

    def update(self, dets):
        if self.locked_id is not None:
            same = [d for d in dets if d.tid == self.locked_id]
            if same:
                self.lost, self.last_xy = 0, (same[0].cx, same[0].cy)
                return same[0], "TRACKING"
            self.lost += 1
            if self.lost <= self.max_lost:
                near = [d for d in dets
                        if d.label == self.label and self._dist(d) < self.radius]
                if near:
                    best = min(near, key=self._dist)
                    self.lock(best)
                    return best, "REACQUIRED"
                return None, "COASTING"
            self.reset()
        if dets:
            best = max(dets, key=lambda d: d.conf)
            self.lock(best)
            return best, "LOCKED"
        return None, "SEARCHING"


class EMA:
    def __init__(self, a):
        self.a, self.v = a, None

    def update(self, x):
        self.v = x if self.v is None else self.a * x + (1 - self.a) * self.v
        return self.v

    def reset(self):
        self.v = None


# ======================================================================
# 5. OUTPUT TO THE ROBOT
# ======================================================================
@dataclass
class TargetMsg:
    t_capture: float
    valid: bool          # False => robot must NOT act on pan/tilt (watchdog)
    status: str
    label: str
    track_id: int
    conf: float
    pan: float           # -1 left ... +1 right
    tilt: float          # -1 up   ... +1 down
    size: float          # box area / frame area (bigger = closer)
    latency_ms: float


class Publisher:
    """Replace publish() with a ROS 2 publisher later."""

    def __init__(self, udp):
        self.sock = self.addr = None
        if udp:
            host, port = udp.split(":")
            self.addr = (host, int(port))
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, msg):
        if self.sock:
            self.sock.sendto(json.dumps(asdict(msg)).encode(), self.addr)


# ======================================================================
# 6. DRAWING
# ======================================================================
def short_id(tid):
    return tid % 100000


def draw(frame, dets, target, msg, fps, args, branches):
    h, w = frame.shape[:2]
    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    dzx, dzy = int(args.deadzone * w / 2), int(args.deadzone * h / 2)
    cv2.rectangle(frame, (w // 2 - dzx, h // 2 - dzy), (w // 2 + dzx, h // 2 + dzy),
                  (120, 120, 120), 1)
    for d in dets:
        is_t = target is not None and d.tid == target.tid
        col = (0, 255, 0) if is_t else (150, 150, 150)
        cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                      col, 3 if is_t else 1)
        cv2.putText(frame, f"{d.label} #{short_id(d.tid)} {d.conf:.2f}",
                    (int(d.x1), max(15, int(d.y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    if target is not None:
        cv2.circle(frame, (int(target.cx), int(target.cy)), 6, (0, 0, 255), -1)
    color = (0, 255, 0) if msg.valid else (0, 0, 255)
    confs = "/".join(f"{b.conf:.2f}" for b in branches)
    lines = [
        f"{args.target} [{'+'.join(b.name for b in branches)}] conf>={confs}",
        f"{msg.status}  {msg.label}  ID {short_id(msg.track_id) if msg.valid else '-'}",
        f"pan {msg.pan:+.2f}  tilt {msg.tilt:+.2f}  size {msg.size:.2f}",
        f"FPS {fps:.0f}  latency {msg.latency_ms:.0f} ms",
    ]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ======================================================================
# 7. MAIN LOOP
# ======================================================================
def main():
    args = parse_args()
    if args.list:
        print_classes(args)
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name(0) if device == 0 else "CPU")

    branches = build_branches(args)
    for b in branches:
        print(f"Branch: {b.name} | conf {b.conf} | classes {b.classes}")

    src = FrameSource(args.source, args.width, args.height)
    ok, frame, _ = src.read()
    if not ok:
        raise RuntimeError("No frames from source")
    fh, fw = frame.shape[:2]

    selector = TargetSelector(args.max_lost, args.reacquire_radius, fw, fh)
    sm_pan, sm_tilt, sm_size = EMA(args.smooth), EMA(args.smooth), EMA(args.smooth)
    pub = Publisher(args.udp)

    log_f = log_w = None
    if args.log:
        log_f = open(args.log, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["t_capture", "status", "valid", "label", "id", "conf",
                        "pan", "tilt", "size", "latency_ms", "infer_ms"])

    snap_dir = Path("snapshots"); snap_dir.mkdir(exist_ok=True)
    t_start, n = time.time(), 0

    try:
        while True:
            ok, frame, t_cap = src.read()
            if not ok:
                print("Source ended or camera failed.")
                break
            h, w = frame.shape[:2]

            dets, infer_ms = [], 0.0
            for b in branches:
                res = b.model.track(frame, persist=True, tracker=args.tracker,
                                    classes=b.classes, conf=b.conf, imgsz=args.imgsz,
                                    device=device, verbose=False)[0]
                dets += extract_detections(res, b)
                infer_ms += res.speed["inference"]

            target, status = selector.update(dets)

            if target is not None:
                pan = sm_pan.update((target.cx - w / 2) / (w / 2))
                tilt = sm_tilt.update((target.cy - h / 2) / (h / 2))
                size = sm_size.update((target.w * target.h) / (w * h))
                pan_o = 0.0 if abs(pan) < args.deadzone else pan
                tilt_o = 0.0 if abs(tilt) < args.deadzone else tilt
                valid, tid, conf, label = True, target.tid, target.conf, target.label
            else:
                if status == "SEARCHING":
                    sm_pan.reset(); sm_tilt.reset(); sm_size.reset()
                pan_o = tilt_o = size = 0.0
                valid, tid, conf, label = False, -1, 0.0, ""

            latency = (time.time() - t_cap) * 1000
            msg = TargetMsg(t_cap, valid, status, label, tid, conf,
                            round(pan_o, 3), round(tilt_o, 3), round(size, 4),
                            round(latency, 1))
            pub.publish(msg)

            n += 1
            fps = n / (time.time() - t_start)
            if log_w:
                log_w.writerow([f"{t_cap:.3f}", status, int(valid), label, tid,
                                f"{conf:.3f}", msg.pan, msg.tilt, msg.size,
                                msg.latency_ms, f"{infer_ms:.1f}"])

            if not args.no_show:
                draw(frame, dets, target, msg, fps, args, branches)
                cv2.imshow("tracking (q quit)", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == ord("r"):
                    selector.reset(); print("Lock released")
                elif k == ord("n") and dets:
                    order = sorted(d.tid for d in dets)
                    cur = selector.locked_id
                    nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else order[0]
                    selector.lock(next(d for d in dets if d.tid == nxt))
                    print("Locked", selector.label, short_id(nxt))
                elif k == ord("s"):
                    p = snap_dir / f"snap_{int(time.time())}.jpg"
                    cv2.imwrite(str(p), frame); print("Saved", p)
                elif k in (ord("+"), ord("=")):
                    for b in branches: b.conf = min(b.conf + 0.05, 0.95)
                    print("confidence +0.05")
                elif k == ord("-"):
                    for b in branches: b.conf = max(b.conf - 0.05, 0.05)
                    print("confidence -0.05")
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        cv2.destroyAllWindows()
        if log_f:
            log_f.close()
        print(f"Done. {n} frames, avg FPS {n / max(time.time() - t_start, 1e-6):.1f}")


if __name__ == "__main__":
    main()