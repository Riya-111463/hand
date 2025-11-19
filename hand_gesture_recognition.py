import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

tip_ids = [4, 8, 12, 16, 20]  # Thumb and fingertips

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    finger_count = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Count fingers
            if lm_list[tip_ids[0]][0] < lm_list[tip_ids[0] - 1][0]:
                finger_count += 1  # Thumb

            for id in range(1, 5):
                if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
                    finger_count += 1

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display finger count
    cv2.putText(frame, f'Fingers: {finger_count}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
