# EgoCentricGestureNet
A repo with model and links to dataset from Egocentric Gesture Recognition for Head-Mounted AR devices (ISMAR 2018 Adjunct)


# Steps to test the model
- Make sure you all the dependencies below installed
- Run download_dataset.sh script, this will download the dataset(**EgoCentricGestures.tar.gz**) to current directory
- Move this file to an appropriate location and untar the file. This will generates two directories and README.txt files explaining the structure of the dataset and directories
- run the model with 'python ego_gesture_net_test.py --test_dir=<full_path_to_test_directory>'. This should output the actual gesture id and the recognised id by the network.

# Dependencies
- wget
- python - 3.5
- torch - 0.3.1
- torchvision - 0.2.1
- Pillow - 5.2.0

#Citation
[Here is a link to the publication](https://arxiv.org/abs/1808.05380)
- Bibtex Entry

@inproceedings{tejo2018ismar,
  author    = {Tejo Chalasani and
               Jan Ondrej and
               Aljosa Smolic},
  title     = {Egocentric Gesture Recognition for Head-Mounted AR devices},
  booktitle = {2018 {IEEE} International Symposium on Mixed and Augmented Reality,
               {ISMAR} 2018 Adjunct},
  year      = {2018}
}
