import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# paths
train_dir = 'E_waste/modified-dataset'
val_dir = 'E_waste/modified-dataset'
img_size = (224,224)
batch_size = 32
num_epochs = 8

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# base model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False  # freeze

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
preds = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=preds)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# train
history = model.fit(train_gen, validation_data=val_gen, epochs=num_epochs)

# optional: fine-tune unfreeze some layers
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_ft = model.fit(train_gen, validation_data=val_gen, epochs=5)

# save
model.save('models/ewaste_classifier.keras')

# save class indices mapping
import json
with open('models/class_indices.json','w') as f:
    json.dump(train_gen.class_indices, f)
