# Face-Emotional-Analysis
The goal of this project is to use a CNN model to analyze live video input of multiple facial expressions and create a generealization of mood among people.
Once a consensus is made a playlist with metadata matching the expression/mood will play. 

CNN Architecture:
model = models.Sequential()

#First layer to normalize the RGB chanels 
model.add(layers.Rescaling(1./255))

#3 layer stack for convlutional block
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPooling2D())

#Flatten for dense layer
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes))

