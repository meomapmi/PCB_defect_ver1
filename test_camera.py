import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("camera", frame)

    # nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

# giải phóng camera
cap.release()

# đóng tất cả cửa sổ
cv2.destroyAllWindows()