import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from config import *
from utils import *


onlyfiles = [f for f in listdir(FACE_PATH) if isfile(join(FACE_PATH, f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = FACE_PATH + onlyfiles[i]  # face/user1.jpg
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
print('[INFO] Model Started Training  !!!')
Labels = np.asarray(Labels, dtype=np.int32)

# Linear Binary Phase Histogram Classifier
model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))
print('[INFO] Model Training Complete !!!')

cap = cv2.VideoCapture(DEVICE)
count = 0

while True:
    ret, frame = cap.read()

    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'faces/user' + str(count) + '.jpg'

        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('FACE CROPPER ', face)
    else:
        print('face not found ')

    if cv2.waitKey(1) == 13 or count == 200:
        break

cap.release()
cv2.destroyAllWindows()
print('Collecting Samples Complete  !!!!')
