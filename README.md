AIR-ASVspoof
===============
# oc-asv-sppof
To detect voice spoofing using OC

## Requirements
python==3.6

pytorch==1.1.0

## Data Preparation
The LFCC features are extracted using the ASVspoof 2019 organizers. Please first run the `process.py`, and then run `python3 reload_data.py` with python.
Make sure you change the directory path to the path on your machine.
## Run the training code
Before running the `train.py`, please change the `path_to_database`, `path_to_features`, `path_to_protocol` according to the files' location on your machine.
```
python3 train.py --add_loss ocsoftmax -o ./models/ocsoftmax --gpu 0
```
## Run the test code with trained model
You can change the `model_dir` to the location of the model you would like to test with.
```
python3 test.py -m ./models/ocsoftmax -l ocsoftmax --gpu 0
