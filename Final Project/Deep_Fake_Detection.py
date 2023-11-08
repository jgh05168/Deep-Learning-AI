# Deep Fake Detection

import glob
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization, Conv3D, Activation, Dense, GlobalAveragePooling3D, MaxPooling3D, ZeroPadding3D


labels_file = 'full.csv'        # 전처리 진행 전 FAKE / REAL 데이터에 대해 사전 labeling 진행
sequence_length = 30
num_classes = 2
num_epochs = 200
batch_size = 2

# Load labels
header_list = ["file", "label"]
labels = pd.read_csv(labels_file, names=header_list)


######## Load and preprocess video frames
# image_size : resize to 112 x 112
# using opencv
def preprocess_frame(image):
    resized_frame = cv2.resize(image, (112, 112)).astype(np.float32)
    min_val = np.min(resized_frame)
    max_val = np.max(resized_frame)
    preprocessed_frame = (resized_frame - min_val) / (max_val - min_val)
    return preprocessed_frame


# Stream video frames
# 각 frame에 대해 전처리 진행 & 일정 길이의 frame sequence 생성
def stream_video_frames(video_path, sequence_length):
    vidObj = cv2.VideoCapture(video_path)       # 각 프레임을 읽어오는 과정
    frames = []
    frame_count = 0
    while True:
        success, image = vidObj.read()
        if not success:
            break
        preprocessed_frame = preprocess_frame(image)
        frames.append(preprocessed_frame)
        frame_count += 1
        if frame_count == sequence_length:
            yield np.array(frames, dtype=np.float32)
            frames = []
            frame_count = 0
    if frames:
        # Pad frames if the video is shorter than the desired sequence length
        num_pad_frames = sequence_length - frame_count
        padded_frames = frames + [frames[-1]] * num_pad_frames
        yield np.array(padded_frames, dtype=np.float32)


def preprocess_video_data(video_files, labels, sequence_length):
    random.shuffle(video_files)
    print(len(video_files))
    data = []
    for video_path in video_files:
        frames_generator = stream_video_frames(video_path, sequence_length)
        frames_list = list(frames_generator)
        if len(frames_list) == 0:
            continue
        frames_array = frames_list[0]
        label = labels.loc[labels["file"] == video_path.split('/')[-1], "label"].values
        if len(label) == 0:
            continue
        label = label[0]
        if label == 'FAKE':
            label = 0
        elif label == 'REAL':
            label = 1
        data.append((frames_array, label))

    if len(data) == 0:
        print("No valid data found.")
        exit()

    X = np.array([item[0] for item in data])
    Y = to_categorical(np.array([item[1] for item in data]))
    return X, Y


########## Data preprocessing
# Train_data : 800
# Test_data : 200
video_files_train = glob.glob('FAKE_train/*.mp4')
video_files_train += glob.glob('REAL_train/*.mp4')
X_train, Y_train = preprocess_video_data(video_files_train, labels, sequence_length)

video_files_val = glob.glob('FAKE_val/*.mp4')
video_files_val += glob.glob('REAL_val/*.mp4')
X_val, Y_val = preprocess_video_data(video_files_val, labels, sequence_length)

video_files_test = glob.glob('FAKE_test/*.mp4')
video_files_test += glob.glob('REAL_test/*.mp4')
X_test, Y_test = preprocess_video_data(video_files_test, labels, sequence_length)


########## Define the model
# Conv3D Model
input_s = Input(shape=(sequence_length, 112, 112, 3), dtype='float32', name='input')
x = Conv3D(64, (3, 3, 3), padding='same')(input_s)
x_1 = Conv3D(64, (1, 1, 1), padding='same')(input_s)
sum1 = Activation(activation='relu')(x + x_1)
block1 = MaxPooling3D((1, 2, 2))(sum1)

x = Conv3D(128, (3, 3, 3), padding='same')(block1)
x_1 = Conv3D(128, (1, 1, 1), padding='same')(block1)
sum2 = Activation(activation='relu')(x + x_1)
block2 = MaxPooling3D((1, 2, 2))(sum2)

x = Conv3D(128, (3, 3, 3), padding='same')(block2)
x_1 = Conv3D(128, (1, 1, 1), padding='same')(block2)
sum3 = Activation(activation='relu')(x + x_1)
block3 = MaxPooling3D((1, 2, 2))(sum3)

x = Conv3D(256, (3, 3, 3), padding='same')(block3)
x_1 = Conv3D(256, (1, 1, 1), padding='same')(block3)
sum4 = Activation(activation='relu')(x + x_1)
block4 = MaxPooling3D((1, 2, 2))(sum4)

output = layers.GlobalAveragePooling3D()(block4)
dropout = layers.Dropout(0.25)(output)
output = Dense(512, activation='relu')(dropout)
output = Dense(num_classes, activation='softmax')(output)

resnet = Model(input_s, output)
resnet.summary()


########## Train the model
resnet.compile(optimizer=optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history = resnet.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=num_epochs, batch_size=batch_size)


########### Test the model
loss, accuracy = resnet.evaluate(X_train, Y_train)
print('Train Loss: {:.4f}'.format(loss))
print('Train Accuracy: {:.2f}%'.format(accuracy * 100))

test_loss, test_accuracy = resnet.evaluate(X_test, Y_test)
print('Test Loss: {:.4f}'.format(test_loss))
print('Test Accuracy: {:.2f}%'.format(test_accuracy * 100))


# Visualization (validation accuracy & validation loss)
def plot_acc(h, title='accuracy'):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def plot_loss(h, title='loss'):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

plt.clf()
plot_loss(history)
plt.savefig('Conv3D.loss.png')
plt.clf()
plot_acc(history)
plt.savefig('Conv3D.accuracy.png')


## Predict the test set
y_pred = resnet.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)


## Plot the confusion matrix
labels = ['FAKE', 'REAL']
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Conv3D Confusion_1.png')