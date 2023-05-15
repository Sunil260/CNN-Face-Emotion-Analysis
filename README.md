# Face-Emotional-Analysis
The goal of this project is to use a CNN model to analyze live video input of multiple facial expressions and create a generealization of mood among people.
Once a consensus is made a playlist with metadata matching the expression/mood will play. 

Newer implementation with a deeper network, data augmentation and learning rate schedule. 
Current itteration:

#First layer to normalize the RGB chanels 
model.add(layers.Rescaling(1./255))

model.add(data_augmentation)

#4 layer stack for convlutional block

model.add(layers.Conv2D(64, 3, padding='same',activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(128, 3, padding='same',activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, 3, padding='same',activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.20))

model.add(layers.Conv2D(256, 3, padding='same',activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.20))

model.add(layers.Conv2D(512, 3, padding='same',activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.20))

#Flatten for dense layer
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dropout(.20, input_shape=(3,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(.20))
model.add(layers.Dense(num_classes,activation = 'softmax'))

<img width="724" alt="Screen Shot 2023-05-15 at 10 38 08 AM" src="https://github.com/Sunil260/CNN-Face-Emotion-Analysis/assets/44715832/d7898139-95a6-4146-8b6f-cd2cc5be1981">
