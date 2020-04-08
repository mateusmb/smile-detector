# smile-detector
A real-time smile detector program

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
