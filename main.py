import cv2
import numpy as np
import time
import winsound
import pyttsx3
from collections import deque
import os
import sys

# Determine video source from command-line arguments
if len(sys.argv) > 1:
    f_name = sys.argv[1]
    if os.path.exists(f_name):
        VIDEO_SOURCE = f_name
    elif f_name == "0":
        VIDEO_SOURCE = int(f_name)
    else:
        print(f"File '{f_name}' not found. Using default video.")
        VIDEO_SOURCE = "v1.mp4"
else:
    print("No video argument provided. Using default video.")
    VIDEO_SOURCE = "v1.mp4"


'''
Configuration
'''
MIN_CONTOUR_AREA = 800
ALERT_COOLDOWN = 3
VOICE_COOLDOWN = 10
LOG_FILE = "alerts.log"

# Motion thresholds
LOW_MOTION = 15_000
MEDIUM_MOTION = 40_000
HIGH_MOTION_SPIKE = 25_000
HIGH_DENSITY = 0.10

TEMPORAL_FRAMES_HIGH = 8
TEMPORAL_FRAMES_MEDIUM = 4

# AI Voice Engine
voice_engine = pyttsx3.init()
voice_engine.setProperty('rate', 160)

# Motion smoothing
motion_history = deque(maxlen=5)

last_voice_alert = 0


# Initialize video source and read the first frame
def init_video(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Video source not accessible")
        exit()

    ret, frame = cap.read()
    if not ret:
        print("Cannot read first frame")
        exit()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cap, gray


# Calculate and smooth crowd motion between frames
def calculate_motion(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    blur = cv2.GaussianBlur(diff, (7, 7), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    motion_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA:
            motion_area += area

    motion_history.append(motion_area)
    return int(sum(motion_history) / len(motion_history))


# Classify crowd risk using motion, acceleration, and density
def classify_risk(motion_area, motion_delta, density):
    score = 0
    reasons = []

    if motion_area > LOW_MOTION:
        score += min(30, 10 * motion_area / LOW_MOTION)
        reasons.append("noticeable movement")

    if motion_area > MEDIUM_MOTION:
        score += min(40, 10 * motion_area / MEDIUM_MOTION)
        reasons.append("heavy group movement")

    if motion_delta > HIGH_MOTION_SPIKE:
        score += min(40, 10 * motion_delta / HIGH_MOTION_SPIKE)
        reasons.append("sudden acceleration")

    if density > HIGH_DENSITY:
        score += 40
        reasons.append("high crowd density")

    score = int(score)

    if score >= 70:
        return "HIGH", ", ".join(reasons), score
    elif score >= 30:
        return "MEDIUM", ", ".join(reasons), score
    else:
        return "LOW", "normal movement", score


# Conflict Resolution (Decision under Uncertainty)
def resolve_conflicting_signals(risk, motion_area, motion_delta, density):
    """
    Prevent HIGH risk when only density is high.
    """

    danger_signals = 1
    
    if motion_area > MEDIUM_MOTION:
        danger_signals += 1
    if motion_delta > HIGH_MOTION_SPIKE:
        danger_signals += 1

    # High density alone should not cause HIGH risk
        if density > HIGH_DENSITY and danger_signals == 0:
            return "MEDIUM", "high density without panic indicators"
    return risk, None


# Confidence Estimation
def calculate_confidence(risk, abnormal_frames):
    base = min(1.0, abnormal_frames / 5)

    if risk == "HIGH":
        return round(min(1.0, base + 0.3), 2)
    elif risk == "MEDIUM":
        return round(min(1.0, base + 0.1), 2)
    else:
        return round(base * 0.5, 2)


# Generate alerts for sustained abnormal crowd behavior
def generate_alert(risk, reason, abnormal_frames, last_alert_time):
    global last_voice_alert
    current_time = time.time()
    timestamp = time.strftime("%H:%M:%S")

    required_frames = (
        TEMPORAL_FRAMES_HIGH if risk == "HIGH" else TEMPORAL_FRAMES_MEDIUM
    )

    if (
        abnormal_frames >= required_frames
        and risk != "LOW"
        and current_time - last_alert_time > ALERT_COOLDOWN
    ):
        alert_msg = (
            f"[{timestamp}] "
            f"Risk Level: {risk} | "
            f"Situation: {reason} | "
            f"Action: "
            f"{'Immediate attention required' if risk == 'HIGH' else 'Monitor the situation'}"
        )

        print("-"*20)
        print(f'''Time: {timestamp}
Risk Level: {risk}
Situation: {reason}
Action: {'Immediate attention required' if risk == 'HIGH' else 'Monitor the situation'}''')

        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(alert_msg + "\n")

        if risk == "HIGH" and current_time - last_voice_alert > VOICE_COOLDOWN:
            winsound.Beep(1200, 700)
            voice_engine.say(
                "Warning. High crowd risk detected. Immediate attention required."
            )
            voice_engine.runAndWait()
            last_voice_alert = current_time

        return current_time, 0

    return last_alert_time, abnormal_frames


# Display risk level and FPS on the video frame
def display_info(frame, risk, fps, time_text, score, confidence):
    COLORS = {
        "LOW": (0, 200, 0),
        "MEDIUM": (0, 200, 255),
        "HIGH": (0, 0, 255)
    }
    risk_color = COLORS[risk]

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (420, 260), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    x = 25
    y = 40
    line_gap = 35

    cv2.putText(frame, time_text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y += line_gap
    cv2.putText(frame, "RISK LEVEL", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

    y += line_gap
    cv2.putText(frame, risk, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, risk_color, 3)

    y += line_gap + 5
    cv2.putText(frame, "RISK SCORE", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

    bar_x = x
    bar_y = y + 15
    bar_width = 360
    bar_height = 14
    filled = int((score / 100) * bar_width)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height),
                  (80, 80, 80), 1)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + filled, bar_y + bar_height),
                  risk_color, -1)

    cv2.putText(frame, f"{score}/100",
                (bar_x + bar_width - 80, bar_y + bar_height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    y = bar_y + bar_height + 30
    cv2.putText(frame, f"FPS: {fps:.2f}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)


    y += 30
    cv2.putText(frame, f"CONFIDENCE: {confidence:.2f}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Crowd Safety Monitor", frame)


# Main processing loop for crowd safety monitoring
def main():
    cap, prev_gray = init_video(VIDEO_SOURCE)

    prev_motion = 0
    last_alert_time = 0
    abnormal_frames = 0

    start_time = time.time()
    frame_count = 0
    fps = 0

    print("Crowd Safety Monitoring Started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed

        elapsed_sec = int(elapsed)
        minutes = elapsed_sec // 60
        seconds = elapsed_sec % 60
        time_text = f"TIME: {minutes:02d}:{seconds:02d}"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion_area = calculate_motion(prev_gray, gray)
        frame_area = frame.shape[0] * frame.shape[1]
        density = min(1.0, motion_area / (frame_area * 0.6))
        motion_delta = abs(motion_area - prev_motion)

        risk, reason, score = classify_risk(
            motion_area, motion_delta, density
        )

        # Conflict resolution (NEW)
        risk, conflict_reason = resolve_conflicting_signals(
            risk, motion_area, motion_delta, density
        )
        if conflict_reason:
            reason = conflict_reason

        if risk in ["HIGH", "MEDIUM"]:
            abnormal_frames += 1
        else:
            abnormal_frames = 0

        confidence = calculate_confidence(risk, abnormal_frames)

        last_alert_time, abnormal_frames = generate_alert(
            risk, reason, abnormal_frames, last_alert_time
        )

        display_info(frame, risk, fps, time_text, score, confidence)

        prev_gray = gray.copy()
        prev_motion = motion_area

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring Stopped")
    print(f"Average FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
