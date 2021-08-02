import cv2
import tensorflow as tf
import numpy as np

face_detector = cv2.CascadeClassifier(
    '../material/Cascades/haarcascade_frontalface_default.xml')
with open('../material/Weights/network_emotions.json', 'r') as json_file:
    json_saved_model = json_file.read()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

video_capture = cv2.VideoCapture(0)
network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('../material/Weights/weights_emotions.hdf5')
network_loaded.compile(loss='categorical_crossentropy',
                       optimizer='Adam', metrics=['accuracy'])

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    faces = face_detector.detectMultiScale(
        frame, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        current_face = frame[y:y + h, x:x + w]
        current_face = cv2.resize(current_face, (48, 48))
        current_face = current_face / 255
        current_face = np.expand_dims(current_face, axis=0)
        prediction = network_loaded.predict(current_face)

        if prediction is not None:
            result = np.argmax(prediction)
            cv2.putText(frame, emotions[result], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
