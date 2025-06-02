import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BRIGHTNESS,-2)

while(True):
    ret, frame = cap.read()

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()