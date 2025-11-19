# collect_gestures.py
import cv2, mediapipe as mp, time, csv, os
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

LABEL = input("Enter label name for this session (e.g. fist): ").strip()
OUT_CSV = "gesture_landmarks.csv"

cap = cv2.VideoCapture(0)
print("Press SPACE to capture sample, q to quit.")

def normalize_landmarks(landmarks):
    # landmarks: list of (x,y,z) normalized by image dims from MediaPipe
    data = np.array(landmarks)  # (21,3)
    # translate so wrist (0) is origin
    origin = data[0]
    data = data - origin
    # scale by max distance
    max_val = np.max(np.abs(data))
    if max_val != 0:
        data = data / max_val
    return data.flatten().tolist()  # length 63

with open(OUT_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    if os.stat(OUT_CSV).st_size == 0:
        # header: label + feat0..feat62
        writer.writerow(["label"] + [f"f{i}" for i in range(63)])
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
        cv2.putText(img, f"Label: {LABEL}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Collect", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == 32:  # SPACE capture
            if res.multi_hand_landmarks:
                l = res.multi_hand_landmarks[0].landmark
                landmarks = [(lm.x, lm.y, lm.z) for lm in l]
                features = normalize_landmarks(landmarks)
                writer.writerow([LABEL] + features)
                print("Captured sample for", LABEL)
            else:
                print("No hand detected — try again.")
cap.release()
cv2.destroyAllWindows()
