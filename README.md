# airplane_cargo_door_yolo_keras

REQUIREMENTS:
Python 2.7
keras >=2.0.8
imgaug


FOR CUSTOM DATA:
Use labeling program : LabelImg
Refer here: LabelImg is a graphical image annotation tool and label object bounding boxes in images (https://youtu.be/p0nR2YsCY_U)
VOC  format is ok

Step 1: Data Preparation
Download the required data set(here airplane cargo door dataset). See airplane_cargo_door_dataset in files.

Organize the dataset into 4 folders:

train_images <= the folder that contains the images to be trained on

train_annotate <= the folder that contains the train annotations in VOC format

validate_images <=the folder that contains the validation images

validate_annotate <=the folder that contains the validation annotations in VOC format

There is a one to one correspondence between the images and anntations. If the validation set is empty, the training set will be automatically splitted into the training set and the validation set using the ratio of 0.8

Step 2: Edit the configuration file
The configuration file is a json file, which looks like this:

