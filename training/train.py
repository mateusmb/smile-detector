import argparse

from keras.utils             import np_utils
from dataset.datasetloader   import DatasetLoader
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    print("[INFO] loading dataset...")
    data, labels = loadDataset()
    # print("[TEST] First 20 data images loaded: {}".format(data[:20]))
    # print("[TEST] First 20 labels loaded: {}".format(labels[:20]))

    print("[INFO] applying one-hot encoding to labels...")
    label_encoder, labels = oneHotEncoding(labels)

    print("[INFO] computing class weights to compensate data imbalance")
    class_weight = computeClassWeight(labels)

    print("[INFO] partitioning data into training and testing...")
    (X_train, X_test, y_train, y_test) = train_test_split(data, labels,
                                                          test_size=0.3,
                                                          stratify=labels,
                                                          random_state=13)

    print("[TEST] checking 5 first of each train and test partition...")
    print("[TEST] X train: {}".format(X_train[:5]))
    print("[TEST] y train: {}".format(y_train[:5]))
    print("[TEST] X test: {}".format(X_test[:5]))
    print("[TEST] y test: {}".format(y_test[:5]))

    print("[INFO] compiling model...")
    # lenet.build(), lenet.compile()

    print("[INFO] training network...")
    # lenet.trainNetwork()

    print("[INFO] evaluating network...")
    print("[INFO] serializing network...")
    print("[INFO] plotting loss and accuracy")

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

if __name__ == '__main__':
    main()