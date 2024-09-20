import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Load data
train_data = pd.read_csv('processed_data/train_landmarks.csv')
test_data = pd.read_csv('processed_data/test_landmarks.csv')

# split data into features and labels
X_train = train_data.drop('Label', axis=1)
y_train = train_data['Label']

X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# encode labels into numeric values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# define neural network
model = keras.models.Sequential()

# input layer
model.add(keras.layers.Dense(512, input_shape=(
    X_train.shape[1],), activation='relu'))

# hidden layers
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.3))

# output layer 
model.add(keras.layers.Dense(len(label_encoder.classes_), activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#print model summary
model.summary()

# train model
history = model.fit(X_train, y_train_encoded, epochs=25, batch_size=32, validation_data=(X_test, y_test_encoded))

# evaluate model on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# save model
model.save('models/sign_language_model.h5')

# Plot accuracy
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()