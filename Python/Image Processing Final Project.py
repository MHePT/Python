import cv2
import numpy as np
import matplotlib.pyplot as plt

def capture_image():
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture an image or 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        cv2.imshow('Live Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

def preprocess_image(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Convert to binary and segmentation using Otsu's thresholding
    binary_image = cv2.threshold(smoothed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return gray_image, smoothed_image, binary_image

def detect_edges(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    #[ 1 2 1]
    #[ 0 0 0] y
    #[-1-2-1]
    #  
    #[ 1 0-1]
    #[ 2 0-2] x
    #[ 1 0-1]
    
    edge_image = cv2.magnitude(sobel_x, sobel_y) #sqrt(x2 + y2)
    edge_image = np.uint8(np.clip(edge_image, 0, 255)) #check range andchange to int 8bit

    # < 50 --> 0 , > 50 --> 255
    strong_edges = cv2.threshold(edge_image, 50, 255, cv2.THRESH_BINARY)[1]
    return strong_edges

def display_results(original, gray, edge):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 2)
    plt.title("Grayscale Image")
    plt.imshow(gray, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Edge Image")
    plt.imshow(edge, cmap='gray')

    plt.tight_layout()
    plt.show()

def Hand_Tracking():
    import mediapipe as mp

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands# create object
    #we need hands detection tool with 70% confidence
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    #object to drraw lines
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the image horizontally and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(rgb_frame)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



captured_image = capture_image()
if captured_image is None:
  print("No image captured. Exiting...")
  exit()

gray_img, smoothed_img, binary_img = preprocess_image(captured_image)
edge_img = detect_edges(gray_img)
display_results(captured_image, gray_img, edge_img)
Hand_Tracking()