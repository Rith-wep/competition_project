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
else:
    print("No video argument provided. Using default video.")
    VIDEO_SOURCE = "v1.mp4"   # 0 for webcam or "video_file_name"


'''
Configuration
'''
MIN_CONTOUR_AREA = 800
ALERT_COOLDOWN = 3
VOICE_COOLDOWN = 10
LOG_FILE = "alerts.log"

# Motion threholds
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

    a, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, a = cv2.findContours(
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
        score += 1
        reasons.append("noticeable movement")

    if motion_area > MEDIUM_MOTION:
        score += 2
        reasons.append("heavy group movement")

    if motion_delta > HIGH_MOTION_SPIKE:
        score += 2
        reasons.append("sudden acceleration")

    if density > HIGH_DENSITY:
        score += 2
        reasons.append("high crowd density")

    if score >= 5:
        return "HIGH", ", ".join(reasons)
    elif score >= 3:
        return "MEDIUM", ", ".join(reasons)
    else:
        return "LOW", "normal movement"

# Generate alerts for sustained abnormal crowd behavior
def generate_alert(risk, reason, abnormal_frames, last_alert_time):
    global last_voice_alert
    current_time = time.time()
    timestamp = time.strftime("%H:%M:%S")

    required_frames = (
        TEMPORAL_FRAMES_HIGH if risk == "HIGH" else TEMPORAL_FRAMES_MEDIUM
    )

    if (abnormal_frames >= required_frames and risk != "LOW" and current_time - last_alert_time > ALERT_COOLDOWN):
        alert_msg = (
            f"[{timestamp}] "
            f"Risk Level: {risk} | "
            f"Situation: {reason} | "
            f"Action: "
            f"{'Immediate attention required' if risk == 'HIGH' else 'Monitor the situation'}"
        )

        print(alert_msg)

        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(alert_msg + "\n")

        if (risk == "HIGH" and current_time - last_voice_alert > VOICE_COOLDOWN):
            winsound.Beep(1200, 700)
            voice_engine.say(
                "Warning. High crowd risk detected. Immediate attention required."
            )
            voice_engine.runAndWait()
            last_voice_alert = current_time

        return current_time, 0

    return last_alert_time, abnormal_frames


# Display risk level and FPS on the video frame
def display_info(frame, risk, fps, time_text):
    color = (0, 255, 0)
    if risk == "MEDIUM":
        color = (0, 255, 255)
    elif risk == "HIGH":
        color = (0, 0, 255)

    
    cv2.putText(frame, time_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 200), 2)

    cv2.putText(frame, f"RISK: {risk}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

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

        risk, reason = classify_risk(motion_area, motion_delta, density)

        if risk in ["HIGH", "MEDIUM"]:
            abnormal_frames += 1
        else:
            abnormal_frames = 0

        last_alert_time, abnormal_frames = generate_alert(
            risk, reason, abnormal_frames, last_alert_time
        )

        display_info(frame, risk, fps, time_text)

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
