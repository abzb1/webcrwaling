import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
import time
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Data pre-processing
PATH = "./Ryan_Brown_dataset"
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")

BATCH_SIZE = 20
IMG_SIZE = (160, 160)
# Make train_dataset
train_dataset = image_dataset_from_directory(train_dir,
											shuffle = True,
											batch_size = BATCH_SIZE,
											image_size = IMG_SIZE)
# Make validation_dataset
validation_dataset = image_dataset_from_directory(validation_dir,
												shuffle = True,
												batch_size = BATCH_SIZE,
												image_size = IMG_SIZE)

#show first two image with label
class_names = train_dataset.class_names
plt.figure(figsize = (10,10))
for images, labels in train_dataset.take(1):
	for i in range(9):
		ax = plt.subplot(3, 3, i+1)
		plt.imshow(images[i].numpy().astype("uint8"))
		plt.title(class_names[labels[i]])
		plt.axis("off")
plt.savefig("train.jpg")
plt.close()


# Make test_dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches//5)
validation_dataset = validation_dataset.skip(val_batches//5)
"""
print("Number of validation batches: %d" %tf.data.experimental.cardinality(validation_dataset))
print("Number of test batches: %d" %tf.data.experimental.cardinality(test_dataset))
"""

# organizing dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size = AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size = AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size = AUTOTUNE)

# Use data augmentation
data_augmentation = tf.keras.Sequential([
					tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
					tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
])


# check the augmentation way
for image, _ in train_dataset.take(1):
	plt.figure(figsize = (10, 10))
	first_image = image[0]
	for i in range(9):
		ax = plt.subplot(3, 3, i + 1)
		augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
		plt.imshow(augmented_image[0]/255)
		plt.axis('off')
plt.savefig("augmentation.jpg")
plt.close()


# resize the pixel value for base model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset = -1)

# instatiate a base model(MobileNetV2) with pre-trained weights
IMG_SHAPE = IMG_SIZE + (3,)
base_model = keras.applications.MobileNetV2(
	weights = 'imagenet', # Load weights pre-trained on ImageNet
	input_shape = IMG_SHAPE,
	include_top = False) # Do not include the ImageNet classifier at the top

# freeze the base model
base_model.trainable = False


# check the model operating way(feature extraction)
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# check the model summary
base_model.summary()


# create prediction on feature block
# Convert features of shape 'base_model.output_shape[1:]' to vectors
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# A Dense classifier with a single unit (binary classification)
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


# create a new model
inputs = tf.keras.Input(shape = IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training = False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# compile model before training
base_learning_rate = 0.0001
model.compile(optimizer = keras.optimizers.RMSprop(lr = base_learning_rate),
			loss = keras.losses.BinaryCrossentropy(from_logits = True),
			metrics = ['accuracy'])

# model summary
model.summary()
print(len(model.trainable_variables))

# check the initial condition
loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}" .format(loss0))
print("initial accuracy: {:.2f}" .format(accuracy0))

# Make a check point for compare to GPU
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
												save_weights_only = True,
												verbose = 1)

# Train the model on new data by CPU
initial_epochs = 200
start = time.time()
history = model.fit(train_dataset,
					epochs = initial_epochs,
					validation_data = validation_dataset)
print("CPU learning time :", time.time()-start)
# show learning curve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize = (8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label = "Traning Accuracy")
plt.plot(val_acc, label = 'Validation Accuracy')
plt.legend(loc = "lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label = "Training Loss")
plt.plot(val_loss, label = "Validation Loss")
plt.legend(loc = "upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 1.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.savefig("result.jpg")
plt.close()

# Go back to checkpoint
model.load_weights(checkpoint_path)

# check the initial condition
loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}" .format(loss0))
print("initial accuracy: {:.2f}" .format(accuracy0))

# Train the model on new data by GPU
start = time.time()
history = model.fit(train_dataset,
					epochs = initial_epochs,
					validation_data = validation_dataset)
print("GPU learning time :", time.time()-start)
# show learning curve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize = (8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label = "Traning Accuracy")
plt.plot(val_acc, label = 'Validation Accuracy')
plt.legend(loc = "lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label = "Training Loss")
plt.plot(val_loss, label = "Validation Loss")
plt.legend(loc = "upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 1.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.savefig("result.jpg")
plt.close()

model.summary()

#save model
os.mkdir("saved_model")
model.save("saved_model/my_model")
