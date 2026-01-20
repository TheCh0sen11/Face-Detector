import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

def enhance_image(img):

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
  
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
    

    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced

def adjust_brightness_contrast(img, brightness=0, contrast=0):

    img = img.astype(np.float32)
    if brightness != 0:
        img = img + brightness
    if contrast != 0:
        img = img * (1.0 + contrast/100.0)
    
    return np.clip(img, 0, 255).astype(np.uint8)

def preprocess_image(img, enhance=True, augment_brightness=False):

    if enhance:
        img = enhance_image(img)
    
    if augment_brightness:

        brightness = np.random.randint(-30, 30)
        contrast = np.random.randint(-10, 10)
        img = adjust_brightness_contrast(img, brightness, contrast)
    
    return img

def create_augmentations():
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def load_data(apply_preprocessing=True):

    images = []
    labels = []
    
    print("Загрузка и предобработка изображений...")
    

    for filename in os.listdir('with_face'):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join('with_face', filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                

                if apply_preprocessing:
                    img = preprocess_image(img, enhance=True, augment_brightness=False)
                
                images.append(img)
                labels.append(1)

    for filename in os.listdir('without_face'):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join('without_face', filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                

                if apply_preprocessing:
                    img = preprocess_image(img, enhance=True, augment_brightness=False)
                
                images.append(img)
                labels.append(0)
    
    print(f"Загружено {len(images)} изображений")
    return np.array(images), np.array(labels)

def create_model():

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.25),
        
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():

    X, y = load_data(apply_preprocessing=True)
    X = X.astype('float32') / 255.0
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    datagen = create_augmentations()
    model = create_model()
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint('face_detection_improved5.h5', save_best_only=True)
    ]
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

if __name__ == "__main__":
    train_model()