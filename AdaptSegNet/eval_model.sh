python3 evaluate_cityscapes.py \
    --num-classes 2 \
    --restore-from "/home/chei/pranav/alertwildfire/AdaptSegNet/snapshots/lmda_adv_0.1_25000.pth" \
    --data-dir "/data/field/nextcloud_nautilius_sync/AlertWildfire/Labelled Frames/synth_dataset/" \
    --gpu 0 \
    --save "/home/chei/pranav/alertwildfire/AdaptSegNet/results/new/"
