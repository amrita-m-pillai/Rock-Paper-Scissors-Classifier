import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("rps_model.h5")
classes = ['paper', 'rock', 'scissors']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    image = cv2.resize(roi, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    label = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Rock Paper Scissors Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
