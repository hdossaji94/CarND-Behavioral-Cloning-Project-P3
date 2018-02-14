import csv
import cv2
import numpy as np
import os
import sklearn

# function to open up extracted data of driving log. returns the lines 
def getLines(dataPath):
	lines = []
	with open('data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	return lines

# Collects all images from center, left and right camera and also measurement angles and 
# puts then in respective lists.
def accumulateImages(dataPath):
	directories = [x[0] for x in os.walk(dataPath)]
	dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))

	centerImages, leftImages, rightImages, measurementTotal = [], [], [], []
	for directory in dataDirectories:
		lines = getLines(directory)
		center = []
		left = []
		right = []
		measurements = []
		for line in lines:
			measurements.append(float(line[3]))
			center.append(directory + '/' + line[0].strip())
			left.append(directory + '/' + line[1].strip())
			right.append(directory + '/' + line[2].strip())
		centerImages.extend(center)
		leftImages.extend(left)
		rightImages.extend(right)
		measurementTotal.extend(measurements)
	return (centerImages, leftImages, rightImages, measurementTotal)

# Adds the correction factor to the image paths based on what camera took the shot so the
# car can correctly learn how to stay in the center
def correctImages(center, left, right, measurement, correction):
	Path = []
	Path.extend(center)
	Path.extend(left)
	Path.extend(right)
	measurements = []
	measurements.extend(measurement)
	measurements.extend([x + correction for x in measurement])
	measurements.extend([x - correction for x in measurement])
	return (Path, measurements)

# Generator that loads the corresponding images and measurements in batch sizes to increase
# memory effeciency
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Forever Loop
		samples = sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for Path, measurement in batch_samples:		
				firstImage = cv2.imread(Path)
				image = cv2.cvtColor(firstImage, cv2.COLOR_BGR2RGB)
				images.append(image)
				angles.append(measurement)
				# Augmenting: Flipping image and adding to list
				images.append(cv2.flip(image,1))
				angles.append(measurement*-1.0)

			
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# PreProcessing Layer and NVIDIA Autonomous Car Group Model
def runModel():
	
	# PreProcessing Layer consisting of normalizing and cropping image
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	
	# NVIDIA Autonomous Car Group Model
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

# Calling Functions to read image locations
centers, lefts, rights, measurements = accumulateImages('data')
Paths, measurements = correctImages(centers, lefts, rights, measurements, 0.2)
print('Total Images: {}'.format(len(Paths)))

# Creating Generators and splitting samples into training samples and validation samples
from sklearn.model_selection import train_test_split
samples = list(zip(Paths, measurements))
train_samples, validation_samples = train_test_split(samples,test_size=0.2)
print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Creating the model
model = runModel()

# Training and compiling model using mse (mean squared error) and adam optimizier
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)



from keras.models import Model
import matplotlib.pyplot as plt

# Saving model
model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.histroy['val_loss'])

# plt.plot(history_object.history['loss'])
# plt.plot(histroy_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()


exit()
