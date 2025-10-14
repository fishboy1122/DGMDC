# DGMDC-Net
Difference-guided Multi-directional Complementary  Network for Remote Sensing Imagery Semantic  Segmentation
<img width="799" height="528" alt="image" src="https://github.com/user-attachments/assets/e51cce99-c3e0-4a87-9434-d6790696774b" />


# Pretrained Weights
```
https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k
```

# Data Preprocessing
Download the datasets from the official website and split them yourself.  

**Vaihingen**  

Generate the training set.
```
python DGMDC/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/train_images" \
--mask-dir "data/vaihingen/train_masks" \
--output-img-dir "data/vaihingen/train/images_1024" \
--output-mask-dir "data/vaihingen/train/masks_1024" \
--mode "train" --split-size 1024 --stride 512 
```
Generate the testing set.  
```
python DGMDC/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks_eroded" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded
```
Generate the masks_1024_rgb (RGB format ground truth labels) for visualization.
```
python DGMDC/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt
```
As for the validation set, you can select some images from the training set to build it.  
**Potsdam**  
```
python DGMDC/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/train_images" \
--mask-dir "data/potsdam/train_masks" \
--output-img-dir "data/potsdam/train/images_1024" \
--output-mask-dir "data/potsdam/train/masks_1024" \
--mode "train" --split-size 1024 --stride 1024 --rgb-image
```
```
python DGMDC/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks_eroded" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded --rgb-image
```
```
python DGMDC/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt --rgb-image

**Uavid**
```
python DGMDC/tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train_val" \
--output-img-dir "data/uavid/train_val/images" \
--output-mask-dir "data/uavid/train_val/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```
```
python DGMDC/tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train" \
--output-img-dir "data/uavid/train/images" \
--output-mask-dir "data/uavid/train/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```
```
python DGMDC/tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_val" \
--output-img-dir "data/uavid/val/images" \
--output-mask-dir "data/uavid/val/masks" \
--mode 'val' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```
 
```
**LoveDA**  
```
python DGMDC/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
python DGMDC/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
python DGMDC/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
python DGMDC/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```

# Training
```
python DGMDC/train_supervision.py -c DGMDC/config/uavid/***.py
```
Use different config to train different models.

# Testing
"-c" denotes the path of the config, Use different config to test different models.  

"-o" denotes the output path  

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA  

"--rgb" denotes whether to output masks in RGB format  

**Vaihingen**  
```
python DGMDC/vaihingen_test.py -c DGMDC/config/vaihingen/DGMDC_config.py -o fig_results/vaihingen/DGMDC --rgb -t 'd4'
```
**Potsdam**  
```
python DGMDC/potsdam_test.py -c DGMDC/config/potsdam/DGMDC_config.py -o fig_results/potsdam/DGMDC --rgb -t 'lr'
```
**Uavid**  
```
python DGMDC/inference_uavid.py \
-i 'data/uavid/uavid_test' \
-c DGMDC/config/uavid/DGMDC_config.py \
-o fig_results/uavid/DGMDC \
-t 'lr' -ph 1152 -pw 1024 -b 2 -d "uavid"
```
**LoveDA**  
```
python DGMDC/loveda_test.py -c DGMDC/config/loveda/DGMDC_config.py -o fig_results/loveda/DGMDC -t 'd4'
```

# Inference on huge remote sensing image
```
python DGMDC/inference_huge_image.py \
-i data/vaihingen/test_images \
-c DGMDC/config/vaihingen/DGMDC_config.py \
-o fig_results/vaihingen/DGMDC \
-t 'lr' -ph 512 -pw 512 -b 2 -d "pv"
```

