# DGMDC-Net
Difference-guided Multi-directional Complementary  Network for Remote Sensing Imagery Semantic  Segmentation
<img width="799" height="528" alt="image" src="https://github.com/user-attachments/assets/e51cce99-c3e0-4a87-9434-d6790696774b" />


# Pretrained Weights
https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k

# Data Preprocessing
Download the datasets from the official website and split them yourself.


# Training
python DGMDC/train_supervision.py -c DGMDC/config/uavid/***.py
Use different config to train different models.

# Testing
python DGMDC/loveda_test.py -c DGMDC/config/loveda/***.py -o fig_results/loveda/*** -t 'd4'

python DGMDC/inference_uavid.py \
-i 'data/uavid/uavid_test' \
-c DGMDCconfig/uavid/***.py \
-o fig_results/uavid/*** \
-t 'lr' -ph 1152 -pw 1024 -b 2 -d "uavid"


