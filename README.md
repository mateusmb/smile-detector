# smile-detector
A real-time smile detector program, based on Adrian Rosebrock's Deep Learning for Computer Vision with Python, Chapter 22.

<img src="https://imgur.com/moPlcdw.jpg" width="360" height="320" alt="No Smile"> <img src="https://imgur.com/76OjrGR.jpg" width="360" height="320" alt="Smile">

## How to Run

Install git
```bash
sudo apt install git
```

Clone this repository
```bash
git clone https://github.com/mateusmb/smile-detector.git
```

Install python3
```bash
sudo apt install python3
```

Install pip3
```bash
sudo apt install python3-pip
```

Install the requirements
```bash
sudo pip3 install -r smile-detector/requirements.txt
```

Run the program
```bash
cd smile-detector
python3 smile-detector.py
```


(Optionally) Run with a custom pre-trained model
```bash
python3 smile-detector/smile-detector.py -m path/to/model.hdf5
```

(Optionally) Run with a video file
```bash
python3 smile-detector/smile-detector.py -v path/to/video.mp4|.avi|.mkv|.wmv
```


## Synopsis
Main program captures real-time video from webcam. Then, detects faces and draw a rectangle around the face area. When the person smiles, it should print a text "Smiling" above the rectangle area. When not smiling, it should print a text "Not Smiling".

## General Guidelines
* Get smiling and not smiling dataset images;
* Train a network in the dataset;
* Evaluate the network;
* Detect face with Haar Cascade;
* Extract the Region of Interest (ROI);
* Pass the ROI through trained network;
* Output the result from trained network.

## Project structure
cascade/  - Folder for cascade classifiers. Provide any classifiers here.

dataset/  - Scripts utils for dataset manipulation. Don't put your datasets here, as will increase project size.

models/   - Folder for storing custom trained models

networks/ - Classes for deep learning models creation and building
    
---- convolutional/ - Folder for convolutional neural networks classes
    
plots/    - Accuracy, loss and other metrics plots can be stored here

training/ - Train scripts for model creation

