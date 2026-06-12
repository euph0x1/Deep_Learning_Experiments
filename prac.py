import tensorflow as tf
from tensorflow.keras import models, layers

(x_train, y_train),(x_test, y_test) =tf.keras.datasets.mnist.load_data()

#mean= x_train.mean(axis=0)
#std= x_train.std(axis=0)

# = (x_train - mean)/ std
#x_test = (x_test - mean)/ std

x_train=x_train/255.0
x_teest = x_test/255.0

x_train=x_train.reshape(-1,28,28,1)

model =models.Sequential([
        layers.Conv2D(32,(3,3) ,input_shape=(28,28,1),activation= 'relu' ),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(32, activation= 'relu'),
        layers.Dense(10, activation='softmax')

])

model.compile( optimizer='adam',
              loss= 'sparse_categorical_crossentropy',
              metrics= ['accuracy']
)

model.fit(x_train, y_train, epochs=2, verbose=0)

loss, acc =model.evaluate(x_test, y_test, verbose=0)

print(loss)
print(acc)