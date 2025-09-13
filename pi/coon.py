#!/usr/bin/env python3
"""
Refactored motion+YOLO+GPIO pipeline with:
 - Auto PiGPIOFactory (reduced servo jitter when available)
 - Alarm cooldown
 - Background-subtraction motion detection (MOG2)

Behavior:
- Initializes camera, YOLO, relay, and servo
- Captures a startup frame and runs a brief servo/relay test
- Main loop: background-subtraction motion detection; on motion, run YOLO
- Append concise top-5 summary; save "likelycat" triplets; raccoon => alarm
- Cooldown prevents back-to-back alarms
"""

from ultralytics import YOLO
import cv2
import time
import numpy as np
import datetime
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from gpiozero import DigitalOutputDevice, Servo

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "best.pt"
CAMERA_INDEX = 0

IMAGES_DIR = Path("images")
SUMMARY_PATH = Path("summary_new.txt")
ALARM_LOG_PATH = Path("alarms.txt")

# GPIO pins
SERVO_GPIO = 24
RELAY_GPIO = 18

# Servo pulse width bounds
MIN_PW = 0.0005
MAX_PW = 0.005

# Motion (background subtraction) parameters
# We compute motion_ratio = nonzero_fg_pixels / total_pixels on the FG mask
MOTION_RATIO_THRESHOLD = 0.15   # ~5% of pixels moving; tune for your scene
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 16
MOG2_DETECT_SHADOWS = False

# Classification thresholds
CAT_CONF_THRESHOLD = 0.30
RACCOON_CLASS_ID = 2
RACCOON_ALARM_CONF = 0.75

# Alarm behavior
ALARM_CYCLES = 10
ALARM_MIN_PAUSE = 0.3
ALARM_MAX_PAUSE = 0.7
ALARM_COOLDOWN_SECONDS = 30.0    # do not trigger another alarm within this window

# Startup relay/servo routine cycles
STARTUP_TOGGLES = 3

# Loop pacing
LOOP_SLEEP_SECONDS = 1.0


# ----------------------------
# Utilities
# ----------------------------
def ts_for_filename() -> str:
    """Filesystem-safe timestamp."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def safe_write(path: Path, text: str, append: bool = True) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        f.write(text)


def save_triplet(base_name: str, img0: np.ndarray, img1: np.ndarray, cam: cv2.VideoCapture) -> None:
    """
    Save three frames with suffixes: '', '_0', and '_1'
    img1 is the main detection frame, img0 is the prior capture,
    and a fresh frame is captured as '_1'.
    """
    p_main = IMAGES_DIR / f"{base_name}.jpg"
    p_0 = IMAGES_DIR / f"{base_name}_0.jpg"
    p_1 = IMAGES_DIR / f"{base_name}_1.jpg"

    cv2.imwrite(str(p_main), img1)
    #cv2.imwrite(str(p_0), img0)
    #ret, img2 = cam.read()
    #if ret:
        #cv2.imwrite(str(p_1), img2)


# ----------------------------
# Hardware wrappers
# ----------------------------
def build_servo() -> Servo:
    """
    Try to use PiGPIOFactory to reduce jitter; fall back to default if unavailable.
    """
    try:
        from gpiozero.pins.pigpio import PiGPIOFactory
        factory = PiGPIOFactory()
        return Servo(SERVO_GPIO, min_pulse_width=MIN_PW, max_pulse_width=MAX_PW, pin_factory=factory)
    except Exception:
        # pigpio not installed/running -> fallback
        return Servo(SERVO_GPIO, min_pulse_width=MIN_PW, max_pulse_width=MAX_PW)


class Hardware:
    def __init__(self) -> None:
        self.relay = DigitalOutputDevice(RELAY_GPIO, active_high=True, initial_value=False)
        self.servo = build_servo()
        self.servo.min()

    def cleanup(self) -> None:
        try:
            self.relay.off()
        except Exception:
            pass
        try:
            self.servo.min()
        except Exception:
            pass

    def pulse_servo_and_relay(self, pause: float, count: int) -> None:
        """One on/off relay pulse paired with servo sweep min->max."""
        for i in range(count):
            self.relay.on()
            self.servo.max()
            time.sleep(pause)
            self.relay.off()
            self.servo.min()
            time.sleep(pause)
        self.servo.min()
        self.servo.max()


    def startup_exercise(self, frame_to_save: np.ndarray) -> None:
        """Run on first loop: save a frame and do a few relay/servo toggles."""
        filename = IMAGES_DIR / f"{ts_for_filename()}.jpg"
        print(str(filename))
        cv2.imwrite(str(filename), frame_to_save)

        for _ in range(STARTUP_TOGGLES):
            self.pulse_servo_and_relay(0.5,3)
            


# ----------------------------
# YOLO handling
# ----------------------------
def load_model(path: str) -> YOLO:
    return YOLO(path)


def classify(model: YOLO, frame: np.ndarray) -> Dict:
    """
    Normalize YOLO classification output:
    {
      "top1_id": int,
      "top1_name": str,
      "top1_conf": float,
      "top5": List[Tuple[str, float]],   # [(name, conf), ...]
      "names": Dict[int, str],
      "raw": Result
    }
    """
    results = model(frame, stream=False)
    result = results[0]
    names_map = result.names

    top1_id = int(result.probs.top1)
    top1_conf = float(result.probs.top1conf.cpu())
    top5_ids = [int(i) for i in list(result.probs.top5)]
    top5_confs = [float(c.item()) for c in list(result.probs.top5conf)]

    top1_name = names_map[top1_id]
    top5_named = [(names_map[i], conf) for i, conf in zip(top5_ids, top5_confs)]

    return {
        "top1_id": top1_id,
        "top1_name": top1_name,
        "top1_conf": top1_conf,
        "top5": top5_named,
        "names": names_map,
        "raw": result
    }


def append_summary(now_iso: str, pick_name: str, top5: List[Tuple[str, float]]) -> None:
    """
    Append a single line to summary.txt:
    <timestamp>\t<pick_name>\t<class1\txx.xx%>\t<class2\txx.xx%>...\n
    """
    sorted_top5 = sorted([f"{name}\t{conf:.2%}" for name, conf in top5])
    line = now_iso + "\t" + pick_name + "\t" + "\t".join(sorted_top5) + "\n"
    safe_write(SUMMARY_PATH, line, append=True)


# ----------------------------
# Motion detection (Background Subtraction)
# ----------------------------
class MotionDetector:
    def __init__(self) -> None:
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS
        )

    def motion_ratio(self, frame: np.ndarray) -> float:
        """
        Returns the ratio of moving pixels to total pixels using FG mask.
        """
        fg = self.subtractor.apply(frame)
        # Binary mask; treat >0 as motion
        moving = np.count_nonzero(fg)
        total = fg.size
        return moving / float(total)


# ----------------------------
# Alarm routine
# ----------------------------
def run_alarm(h: Hardware) -> None:
    for _ in range(ALARM_CYCLES):
        pause = random.uniform(ALARM_MIN_PAUSE, ALARM_MAX_PAUSE)
        h.pulse_servo_and_relay(pause,ALARM_CYCLES)
        time.sleep(pause)
        


# ----------------------------
# Main loop
# ----------------------------
def main() -> None:
    ensure_dirs()
    hw = Hardware()

    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}")

    # optional warmup
    time.sleep(0.5)

    model = load_model(MODEL_PATH)
    motion = MotionDetector()

    startup_done = False
    last_alarm_time = 0.0  # epoch seconds

    try:
        while True:
            ret1, frame1 = cam.read()
            if not ret1:
                print("WARN: Failed to read frame1; retrying...")
                time.sleep(0.1)
                continue

            if not startup_done:
                hw.startup_exercise(frame1)
                startup_done = True

            time.sleep(LOOP_SLEEP_SECONDS)

            # Grab a "prior" frame for triplet saving
            ret0, frame0 = cam.read()
            if not ret0:
                frame0 = frame1

            # Background subtraction motion
            mratio = motion.motion_ratio(frame1)

            if mratio > MOTION_RATIO_THRESHOLD:
                now = datetime.datetime.now()
                now_iso = now.isoformat(timespec="seconds")
                print(f"Motion detected: ratio={mratio:.3%}")

                print("Start:", now_iso)
                cls = classify(model, frame1)
                print("End:", datetime.datetime.now().isoformat(timespec="seconds"))

                # Summary log
                append_summary(now_iso, cls["top1_name"], cls["top5"])

                # Convenience lookup for "cat" / "raccoon" names in top-5
                names_to_conf = dict(cls["top5"])
                cat_conf = names_to_conf.get("cat", 0.0)
                raccoon_conf = names_to_conf.get("raccoon", 0.0)

                print(now_iso)
                print("Coon Confidence:", f"{raccoon_conf:.2%}")
                print("Cat Confidence :", f"{cat_conf:.2%}")

                # High cat confidence: save triplet
                if cat_conf > CAT_CONF_THRESHOLD:
                    base = f"likelycat_{ts_for_filename()}"
                    save_triplet(base, frame0, frame1, cam)
                    print("High Cat Confidence -> saved images.")

                # Raccoon alarm with cooldown
                if cls["top1_id"] == RACCOON_CLASS_ID and cls["top1_conf"] > RACCOON_ALARM_CONF:
                    now_sec = time.time()
                    since_last = now_sec - last_alarm_time
                    if since_last >= ALARM_COOLDOWN_SECONDS:
                        base = f"raccoon_{ts_for_filename()}"
                        safe_write(ALARM_LOG_PATH, f"{now_iso}\n***ALARM***\n{IMAGES_DIR / (base + '.jpg')}\n", append=True)
                        save_triplet(base, frame0, frame1, cam)
                        print("***ALARM*** -> running deterrent pattern")
                        run_alarm(hw)
                        last_alarm_time = now_sec
                    else:
                        remaining = int(ALARM_COOLDOWN_SECONDS - since_last)
                        print(f"Alarm suppressed (cooldown {remaining}s remaining).")

    except KeyboardInterrupt:
        print("Shutting down (CTRL+C).")
    finally:
        try:
            cam.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        hw.cleanup()


if __name__ == "__main__":
    main()
