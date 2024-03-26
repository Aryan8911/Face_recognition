import cv2
import numpy as np
import os

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = np.sqrt(np.sum((train - test) ** 2, axis=1))
    nearest_neighbors = np.argsort(dist)[:k]
    labels = train_labels[nearest_neighbors]
    output = np.unique(labels, return_counts=True)
    prediction = output[0][np.argmax(output[1])]
    return prediction

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

dataset_path = r"D:\program\.vscode\face rego\faces\\"  # Fix path issue
face_data = []
train_labels = []

class_id = 0
names = {}

# Load datasets and assign labels
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        print("Loaded " + fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)
        train_labels.extend([class_id] * data_item.shape[0])  # Assign unique labels
        class_id += 1

face_dataset = np.concatenate(face_data, axis=0)
train_labels = np.array(train_labels)[:, np.newaxis]

trainset = np.concatenate((face_dataset, train_labels), axis=1)

while True:
    ret, frame = cap.read()
    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for face in faces:
        x, y, w, h = face
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        out = knn(trainset[:, :-1], face_section.flatten())
        pred_name = names[out]
        cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("Face", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
