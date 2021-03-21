Download pretrained model in the root directory
'''
wget https://github.com/foamliu/Deep-Image-Matting-PyTorch/releases/download/v1.0/BEST_checkpoint.tar
'''

Run the demo with required arguments
'''
python demo.py -i /path/to/image/folder -t path/to/trimap/folder -o path/to/output/folder 
-b /path/to/new/background/image
'''