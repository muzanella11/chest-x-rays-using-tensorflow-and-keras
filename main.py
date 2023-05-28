import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                if resized_arr.shape != (img_size, img_size):
                    continue  # Skip images with incorrect shape
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    data = np.array(data, dtype=object)  # Convert the list to a NumPy array

    return data

def show_list_data_frame(datasets, label):
    result = []
    for i in datasets:
        if(i[1] == 0):
            result.append("Pneumonia")
        else:
            result.append("Normal")

    print('result :: ', result)

    # Convert list to DataFrame
    df = pd.DataFrame(result, columns=[label])

    # Set SNS
    sns.set_style('darkgrid')
    sns.countplot(data=df, x=label)

    # Set PLT
    plt.figure(figsize = (5,5))
    plt.imshow(train[0][0], cmap='gray')
    plt.title(labels[train[0][1]])

    plt.figure(figsize = (5,5))
    plt.imshow(train[-1][0], cmap='gray')
    plt.title(labels[train[-1][1]])
    plt.show()

# Loading Datasets
train = get_training_data('./datasets/train')
test = get_training_data('./datasets/test')
val = get_training_data('./datasets/val')

# Reshape and convert training data
x_train = np.array([i[0] for i in train]).reshape(-1, img_size, img_size, 1)
y_train = np.array([i[1] for i in train])

# Reshape and convert test data
x_test = np.array([i[0] for i in test]).reshape(-1, img_size, img_size, 1)
y_test = np.array([i[1] for i in test])

# Reshape and convert validation data
x_val = np.array([i[0] for i in val]).reshape(-1, img_size, img_size, 1)
y_val = np.array([i[1] for i in val])

# Print the shape of the resized data
print("Resized Train Data Shape:", x_train.shape)
print("Resized Test Data Shape:", x_test.shape)
print("Resized Validation Data Shape:", x_val.shape)

# Show list for train, test and val
show_list_data_frame(train, 'train')
show_list_data_frame(test, 'test')
show_list_data_frame(val, 'val')

# Datagen
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1
)


datagen.fit(x_train)

# Training model
model = Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1 , activation = 'sigmoid'))
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = 12 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])

print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

print('hello')