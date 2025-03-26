import mediapipe as mp
from controller import Controller
import cv2
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

print("Press 'q' to quit")

try:
    while cap.isOpened():
        start_time = time.time()
        success, img = cap.read()
        
        if not success:
            print("Failed to capture frame. Retrying...")
            time.sleep(1)  
            continue
        
        img = cv2.flip(img, 1)
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(imgRGB)
        
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
                
                Controller.hand_Landmarks = hand_landmarks
                Controller.update_fingers_status()
                Controller.cursor_moving()
                Controller.detect_scrolling()
                Controller.detect_zoomming()
                Controller.detect_clicking()
                Controller.detect_dragging()

        cv2.imshow('Hand Tracker', img)

        if cv2.waitKey(5) & 0xFF == 27: 
            print("Quitting program...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
    cap.release()
    cv2.destroyAllWindows()

print("Program ended successfully.")