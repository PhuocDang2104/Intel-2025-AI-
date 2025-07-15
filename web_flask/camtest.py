import cv2

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Can't receive frame")
        break
    cv2.imshow('Test Camera', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()