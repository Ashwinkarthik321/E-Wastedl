import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json, os

train_dir = 'E_waste/modified-dataset'
img_size = (224,224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train = datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    subset="training", class_mode='categorical')

val = datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    subset="validation", class_mode='categorical')

base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train.num_classes, activation='softmax')(x)

model = Model(base.input, output)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train, validation_data=val, epochs=8)

# fine-tune
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=5)

os.makedirs("models", exist_ok=True)
model.save("models/ewaste_classifier.keras")

with open("models/class_indices.json", "w") as f:
    json.dump(train.class_indices, f)
