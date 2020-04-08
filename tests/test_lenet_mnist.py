from networks.convolutional.lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

print("[INFO] accessing MNIST...")
dataset = datasets.fetch_openml("mnist_784")
data = dataset.data

if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 28, 28)
else:
    data = data.reshape(data.shape[0], 28, 28, 1)

(X_train, X_test, y_train, y_test) = train_test_split(data / 255.0,
                                   dataset.target.astype("int"),
                                   test_size=0.25, random_state=42)

le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print("[INFO] compiling model...")
opt = SGD(lr=0.01)
lenet = LeNet(width=28, height=28, depth=1, classes=10)
lenet.build()
lenet.compile(loss_function="categorical_crossentropy", optimizer=opt,
              metrics_list=["accuracy"])

print("[INFO] training network...")
history = lenet.trainNetwork(data=X_train, labels=y_train,
                             validation_data=(X_test,y_test),
                             batch_size=128, epochs=20)

print("[INFO] evaluating network...")
predictions = lenet.predict(data=X_test, batch_size=128)
print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), history.history["accuracy"], label="acc")
plt.plot(np.arange(0, 20), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()