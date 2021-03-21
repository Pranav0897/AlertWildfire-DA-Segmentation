Code for training and evaluation of AdaptSegNet.

To train the model, first download the dataset into the dataset folder (see the readme in `project_root`/dataset/README.md), and then run:
`train_model.sh`

For evaluation, either use the freshly trained model, or download the model checkpoint from: https://drive.google.com/drive/folders/1-osMbQ2HcCHtLT3hc2KWPQ5nzZPIob1C?usp=sharing

Set the path to the checkpoint in `--restore_from` in `eval_model.sh`, and run it.