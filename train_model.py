import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Load data
train_data = pd.read_csv('processed_data/train_landmarks.csv')
test_data = pd.read_csv('processed_data/test_landmarks.csv')

# split data into features and labels
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']

X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# encode labels into numeric values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# define neural network
"""
I decided to use a Fully-Connected Neural Network over something more complicated like a Convolutional Neural Network due to its simplicity.
Data flows in one direction, layer by layer.
I'm working with numberical values that represent hand landmarks, so using a CNN is not necessary.
If I was analyzing pixel data then a CNN could be more helpful to identify patterns and shapes. 
"""
model = keras.models.Sequential()

# input layer
'''
Dense layers help connect everything together, allowing the model to learn from features in data.
Relu is used to ignore negative values, which helps the model pickup more complex patterns and improves its learning capabiltiies.
Dropout layers help get rid of some data to prevent overfitting. 
'''
model.add(keras.layers.Dense(512, input_shape=(
    X_train.shape[1],), activation='relu'))
# hidden layers
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.3))


# output layer
'''
Softmax is used to convert the outputs into probabilities, which is how the model determines its prediction. 
'''
model.add(keras.layers.Dense(len(label_encoder.classes_), activation='softmax'))

# compile model
'''
The adam optimizer is used for updating the model to improve accuracy.
Sparse_categorical_crossentropy is used to calculate how far the model's predictions are from the correct category.
'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# print model summary
model.summary()

# Add early stopping to prevent overfitting
'''
The loss is monitored and if it doesn't improve for 5 epochs, then training stops.
Model's weights will be set to the point where the model's loss was lowest.
'''
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# train model
history = model.fit(X_train, y_train_encoded, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test_encoded), callbacks=[early_stopping])

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
