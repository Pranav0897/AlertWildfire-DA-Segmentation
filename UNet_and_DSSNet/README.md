To train the UNet model, run:

`scripts/train_unet.sh`

To finetune it on target data, run:

`scripts/transfer_learn.sh`

To evaluate on labelled or unlabelled target data, run:

`scripts/test_unet.sh`

Modify the path to the checkpoint file to test on source only model or finetuned model.
To run on labelled data, change the `test_loader` argument to `annotated`, otherwise use `no_annotated`.