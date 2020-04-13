import argparse
from   dataset.datasetloader import DatasetLoader


def getArguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset of faces")
    ap.add_argument("-m", "--model", default="models/lenet_smiles.hdf5",
                    help="path to save model")
    return vars(ap.parse_args())

def main():
    print("[INFO] loading dataset...")
    args = getArguments()
    dataset_loader = DatasetLoader(args["dataset"])
    (data, labels) = dataset_loader.load(verbose=100)
    print("[TEST] First 20 data images loaded: {}".format(data[:20]))
    print("[TEST] First 20 data images loaded: {}".format(labels[:20]))
    # dataset_loader.load(data_path=image_paths, verbose=1)
    # returns tuple of numpy array data and labels

    print("[INFO] scaling raw pixel intensities...")
    # preprocessor.scalePixel(images=data)
    # returns numpy array

    print("[INFO] applying one-hot encoding to labels...")
    # preprocessor.oneHotEncoding(labels=labels)
    # return labels numpy array

    print("[INFO] partitioning data into training and testing...")
    # train_test_split(data, labels, test_size=0.3, random_state=13)
    # returns tuple X_train, X_test, y_train, y_test

    print("[INFO] compiling model...")
    # lenet.build(), lenet.compile()

    print("[INFO] training network...")
    # lenet.trainNetwork()

    print("[INFO] evaluating network...")
    print("[INFO] serializing network...")
    print("[INFO] plotting loss and accuracy")



if __name__ == '__main__':
    main()