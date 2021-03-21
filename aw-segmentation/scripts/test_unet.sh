python3 test.py --data_root="/data/field/nextcloud_nautilus_sync/AlertWildfire/Labelled Frames/smoke/" \
                --exp_dir="unet" \
                --exp_name="frac_0.2_images_unlabelled_masks" \
                --num_test_images=400\
                --model_name="unet" \
                --log_dir="/home/chei/pranav/alertwildfire/aw-segmentation/checkpoints" \
                --test_loader="no_annotated" \
                --checkpoint_path="/home/chei/pranav/alertwildfire/aw-segmentation/checkpoints/unet/unet_finetuning_frac_real_0.2/finetune_30.ckpt"
                
                # --checkpoint_path="/data/field/nextcloud_nautilus_sync/AlertWildfire/checkpoints/unet/resnet_new_data/30.ckpt"
                
                
                # --checkpoint_path="/data/field/nextcloud_nautilus_sync/AlertWildfire/deep_smoke_segmentation/checkpoints/unet/033529-20201030/8.ckpt" \
