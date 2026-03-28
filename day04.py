

import cv2
import mediapipe as mp
import time
import sys


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_TOP    = [159, 160, 161]
LEFT_EYE_BOTTOM = [145, 144, 153]
LEFT_EYE_LEFT   = 33
LEFT_EYE_RIGHT  = 133

RIGHT_EYE_TOP    = [386, 387, 388]
RIGHT_EYE_BOTTOM = [374, 373, 380]
RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263


def get_ear(landmarks, top_ids, bottom_ids, left_id, right_id):
    """
    Calculate Eye Aspect Ratio (EAR).
    EAR = (vertical distance) / (horizontal distance)
    Open eye: ~0.25-0.35  |  Closed eye: < 0.20
    """
    vertical = 0
    for t, b in zip(top_ids, bottom_ids):
        vertical += abs(landmarks[t].y - landmarks[b].y)
    vertical /= len(top_ids)

    horizontal = abs(landmarks[left_id].x - landmarks[right_id].x)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

#
EAR_THRESHOLD = 0.21         

WINK_OPEN_MARGIN = 0.05    

BLINK_TIME_WINDOW  = 2.0 
BLINKS_TO_LOCK     = 3    
MIN_BLINK_DURATION = 2        



STATE_IDLE     = "IDLE"
STATE_COUNTING = "COUNTING"
STATE_LOCKED   = "LOCKED"

state = STATE_IDLE

blink_count         = 0
counting_start_time = 0.0
eye_closed_frames   = 0

left_closed_frames  = 0
right_closed_frames = 0

wink_triggered = False 


def reset_blink_state():
    global blink_count, counting_start_time, eye_closed_frames
    blink_count         = 0
    counting_start_time = 0.0
    eye_closed_frames   = 0


def draw_centered_text(frame, text, y, scale=1.0, color=(255, 255, 255), thickness=2):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (w - tw) // 2
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


print("\nBlinkLock Enhanced is running!")
print(f"EAR threshold : {EAR_THRESHOLD}")
print(f"Lock trigger  : Blink {BLINKS_TO_LOCK}x within {BLINK_TIME_WINDOW}s")
print(f"Unlock        : Wink either eye | 'u' key\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    h, w      = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = face_mesh.process(rgb_frame)

    now = time.time()

    if state == STATE_LOCKED:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        draw_centered_text(frame, "LOCKED", h // 2 - 50,
                           scale=2.5, color=(0, 0, 255), thickness=5)
        draw_centered_text(frame, "WINK to unlock",
                           h // 2 + 10, scale=0.75, color=(180, 180, 180))
        draw_centered_text(frame, "'u' = emergency unlock",
                           h // 2 + 45, scale=0.55, color=(120, 120, 120))

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            left_ear  = get_ear(lm, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,
                                LEFT_EYE_LEFT,  LEFT_EYE_RIGHT)
            right_ear = get_ear(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)

            left_closed  = left_ear  < EAR_THRESHOLD
            right_closed = right_ear < EAR_THRESHOLD
            open_thresh  = EAR_THRESHOLD + WINK_OPEN_MARGIN

            left_closed_frames  = left_closed_frames  + 1 if left_closed  else 0
            right_closed_frames = right_closed_frames + 1 if right_closed else 0

            if (left_closed_frames >= MIN_BLINK_DURATION and
                    not right_closed and right_ear >= open_thresh):
                state = STATE_IDLE
                left_closed_frames = right_closed_frames = 0
                reset_blink_state()
                print("LEFT WINK → UNLOCKED")

            elif (right_closed_frames >= MIN_BLINK_DURATION and
                      not left_closed and left_ear >= open_thresh):
                state = STATE_IDLE
                left_closed_frames = right_closed_frames = 0
                reset_blink_state()
                print("RIGHT WINK → UNLOCKED")

        cv2.imshow("BlinkLock - Day 04", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('u'):
            state = STATE_IDLE
            reset_blink_state()
            print("Emergency UNLOCK")
        elif key == ord('q'):
            break
        continue

    current_ear  = 0.0
    left_ear_val = right_ear_val = 0.0

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        left_ear_val  = get_ear(lm, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,
                                LEFT_EYE_LEFT,  LEFT_EYE_RIGHT)
        right_ear_val = get_ear(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
        current_ear   = (left_ear_val + right_ear_val) / 2

        eye_is_closed = current_ear < EAR_THRESHOLD

        if eye_is_closed:
            eye_closed_frames += 1
        else:
            if eye_closed_frames >= MIN_BLINK_DURATION:
                if state == STATE_IDLE:
                    state               = STATE_COUNTING
                    blink_count         = 1
                    counting_start_time = now
                    print(f"Blink {blink_count}/{BLINKS_TO_LOCK}")
                elif state == STATE_COUNTING:
                    blink_count += 1
                    print(f"Blink {blink_count}/{BLINKS_TO_LOCK}")
                    if blink_count >= BLINKS_TO_LOCK:
                        state = STATE_LOCKED
                        reset_blink_state()
                        print("LOCKED! Wink to unlock.")
            eye_closed_frames = 0

        if state == STATE_COUNTING:
            elapsed = now - counting_start_time
            if elapsed > BLINK_TIME_WINDOW:
                print(f"Timeout. {blink_count}/{BLINKS_TO_LOCK} blinks. Reset.")
                state = STATE_IDLE
                reset_blink_state()

        for idx in (LEFT_EYE_TOP + LEFT_EYE_BOTTOM +
                    RIGHT_EYE_TOP + RIGHT_EYE_BOTTOM):
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    ear_color = (0, 0, 255) if current_ear < EAR_THRESHOLD else (0, 255, 0)
    cv2.putText(frame, f"EAR: {current_ear:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
    cv2.putText(frame, f"L: {left_ear_val:.3f}  R: {right_ear_val:.3f}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(frame, f"Threshold: {EAR_THRESHOLD}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    state_colors = {
        STATE_IDLE:     (200, 200, 200),
        STATE_COUNTING: (0, 200, 255),
    }
    cv2.putText(frame, f"State: {state}", (10, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                state_colors.get(state, (200, 200, 200)), 2)

    if state == STATE_COUNTING:
        elapsed   = now - counting_start_time
        remaining = max(0, BLINK_TIME_WINDOW - elapsed)
        cv2.putText(frame, f"Blinks: {blink_count}/{BLINKS_TO_LOCK}",
                    (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Time: {remaining:.1f}s",
                    (10, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        bar_w = int((remaining / BLINK_TIME_WINDOW) * 200)
        cv2.rectangle(frame, (10, 173), (10 + bar_w, 183), (0, 200, 255), -1)
        cv2.rectangle(frame, (10, 173), (210, 183), (100, 100, 100), 1)

    cv2.putText(frame,
                f"Blink {BLINKS_TO_LOCK}x=LOCK | Wink=UNLOCK | q=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)

    cv2.imshow("BlinkLock - Day 04", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u') and state == STATE_LOCKED:
        state = STATE_IDLE
        reset_blink_state()
        print("Emergency UNLOCK")

cap.release()
cv2.destroyAllWindows()
print("\nBlinkLock ended. See you tomorrow for Day 05!")
