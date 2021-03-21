python3 train_multigpu.py \
    --batch-size 12 \
    --num-workers 4 \
    --num-classes 2 \
    --ignore-label 253 \
    --exp_dir 'deeplab_crossentropy' \
    --data-dir "/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/synth_dataset/" \
    --data-dir-target "/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/smoke" \
    --num_gpus=3 \
    --iter-size 4
    # --restore-from "/home/chei/pranav/alertwildfire/AdaptSegNet/snapshots/smoke_cross_entropy_25000.pth" \
