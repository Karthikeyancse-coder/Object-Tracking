import cv2, time
from ultralytics import YOLO

model = YOLO("airpods_runs/v2/weights/best.pt")   # or runs/detect/airpods_runs/v2/weights/best.pt
CONF = 0.6                                        # raise/lower with keys

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)          # DSHOW is faster on Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

t0, n = time.time(), 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                      conf=CONF, device=0, verbose=False)[0]
    out = res.plot()

    n += 1
    fps = n / (time.time() - t0)
    cv2.putText(out, f"FPS {fps:.0f}  conf>={CONF:.2f}  boxes {len(res.boxes)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("AirPods live (q quit, +/- conf)", out)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k in (ord("+"), ord("=")):
        CONF = min(CONF + 0.05, 0.95)
    elif k == ord("-"):
        CONF = max(CONF - 0.05, 0.05)

cap.release()
cv2.destroyAllWindows()