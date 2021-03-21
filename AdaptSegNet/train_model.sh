python3 train_gta2cityscapes_multi.py \
    --batch-size 8 \
    --num-workers 4 \
    --num-classes 2 \
    --ignore-label 253 \
    --exp_dir 'deeplab_crossentropy_single_gpu' \
    --data-dir "/data/field/AlertWildfire/Labelled Frames/synth_dataset/" \
    --data-dir-target "/data/field/AlertWildfire/Labelled Frames/smoke" \
    --gpu 0 \
    --iter-size 4
