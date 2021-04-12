import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

IMG_DIR = "emotions/"

EMOTIONS = {
    '0': 'anger',
    '1': 'disgust',
    '2': 'fear',
    '3': 'happiness',
    '4': 'sadness',
    '5': 'surprise',
    '6': 'neutral',
}

df = pd.read_csv('fer2013.csv', encoding='utf-8')

df['emotion'] = df['emotion'].astype(str)

train = 0
test = 0

for index, row in tqdm(df.iterrows()):
    img = np.array([list(map(lambda x: int(x), row['pixels'].split()))]).reshape((48, 48))

    if row['Usage'] == 'Training':
        path = f"{IMG_DIR}train/{EMOTIONS[row['emotion']]}/{index}.jpg"
        train += 1
    else:
        path = f"{IMG_DIR}test/{EMOTIONS[row['emotion']]}/{index}.jpg"
        test += 1

    cv2.imwrite(path, img)

print("Train Images:", train)
print("Test Images:", test)
