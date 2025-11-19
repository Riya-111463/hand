# 
import cv2, mediapipe as mp, joblib, numpy as np
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

model = joblib.load("gesture_rf.joblib")

def normalize_landmarks(landmarks):
    data = np.array(landmarks)
    origin = data[0]
    data = data - origin
    max_val = np.max(np.abs(data))
    if max_val != 0:
        data = data / max_val
    return data.flatten().reshape(1,-1)

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    label_text = ""
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        landmarks = [(p.x, p.y, p.z) for p in lm]
        feat = normalize_landmarks(landmarks)
        pred = model.predict(feat)[0]
        # optional: model.predict_proba to get confidence
        label_text = f"{pred}"
        mp_drawing.draw_landmarks(img, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
    cv2.putText(img, label_text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    cv2.imshow("Gesture", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
