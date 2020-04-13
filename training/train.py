import argparse
import numpy             as np
import matplotlib.pyplot as plt

from keras.utils                  import np_utils
from dataset.datasetloader        import DatasetLoader
from sklearn.metrics              import classification_report
from sklearn.preprocessing        import LabelEncoder
from sklearn.model_selection      import train_test_split
from networks.convolutional.lenet import LeNet

def main():
    print("[INFO] loading dataset...")
    data, labels = loadDataset()

    print("[INFO] applying one-hot encoding to labels...")
    label_encoder, labels = oneHotEncoding(labels)

    print("[INFO] computing class weights to compensate data imbalance")
    class_weight = computeClassWeight(labels)

    print("[INFO] partitioning data into training and testing...")
    (X_train, X_test, y_train, y_test) = train_test_split(data, labels,
                                                          test_size=0.3,
                                                          stratify=labels,
                                                          random_state=13)

    print("[INFO] compiling model...")
    lenet_model = createModel()

    print("[INFO] training network...")
    model_history = trainModel(lenet_model, X_train, X_test,
                               y_train, y_test, class_weight)

    print("[INFO] evaluating network...")
    evaluateNetwork(lenet_model, model_history, label_encoder, X_test, y_test)

    print("[INFO] serializing network...")

def getArguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset of faces")
    ap.add_argument("-m", "--model", default="models/lenet_smiles.hdf5",
                    help="path to save model")
    return vars(ap.parse_args())

def loadDataset():
    args = getArguments()
    dataset_loader = DatasetLoader(args["dataset"])
    (data, labels) = dataset_loader.load(verbose=100)
    return data, labels

def oneHotEncoding(labels):
    label_encoder = LabelEncoder().fit(labels)
    labels = np_utils.to_categorical(label_encoder.transform(labels), 2)
    return label_encoder, labels

def computeClassWeight(labels):
    class_totals = labels.sum(axis=0)
    class_weight = class_totals.max() / class_totals
    return class_weight

def createModel():
    lenet_model = LeNet(width=28, height=28, depth=1, classes=2)
    lenet_model.build()
    lenet_model.compile(loss_function="binary_crossentropy",
                        optimizer="adam", metrics_list=["accuracy"])
    return lenet_model

def trainModel(model, data_train, data_test, label_train,
               label_test, class_weight):
    return model.trainNetwork(data=data_train, labels=label_train,
                             validation_data=(data_test, label_test),
                             class_weight=class_weight, batch_size=64,
                             epochs=15)

def evaluateNetwork(model, model_history, label_encoder, data, labels):
    predictions = model.predict(data, batch_size=64)
    print(classification_report(labels.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=label_encoder.classes_))
    plot(model_history)

def plot(history):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 15), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 15), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 15), history.history["accuracy"], label="acc")
    plt.plot(np.arange(0, 15), history.history["val_accuracy"],
             label="val_acc")
    plt.title("Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()