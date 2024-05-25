import cv2

cap = cv2.VideoCapture(1)  # Try different indices like 1, 2 if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

cv2.imshow('frame', frame)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()