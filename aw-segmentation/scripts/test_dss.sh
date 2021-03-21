python3 test.py --data_root="/data/field/AlertWildfire/Labelled Frames/smoke/" \
                --exp_dir="dss_new_smoke" \
                --exp_name="new_data_trained_ep5" \
                --num_test_images=400\
                --model_name="dss" \
                --log_dir="/data/field/AlertWildfire/deep_smoke_segmentation/checkpoints" \
                --checkpoint_path="/data/field/AlertWildfire/deep_smoke_segmentation/checkpoints/dss_new_smoke/113221-20201022/5.ckpt" \
                --test_loader "eh" \
                --num_test_images 50
                # --checkpoint_path="/data/field/AlertWildfire/deep_smoke_segmentation/checkpoints/unet/033529-20201030/8.ckpt" \
