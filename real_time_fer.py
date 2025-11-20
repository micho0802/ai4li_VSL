import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = DeepFace.analyze(
        frame,
        actions=["emotion"],
        enforce_detection=False 
    )

    if isinstance(result, list):
        result = result[0]

    emotion = result.get("dominant_emotion", "unknown")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        emotion,
        (50, 50),
        font,
        3,
        (0, 0, 255),
        2,
        cv2.LINE_4
    )

    cv2.imshow("Original video", frame)

    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
