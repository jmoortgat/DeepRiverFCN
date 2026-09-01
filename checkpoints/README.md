# Trained model checkpoints

The trained TensorFlow checkpoints for all five model architectures
(panchromatic and 4-band multispectral variants) are hosted externally:

- Google Drive: https://drive.google.com/drive/folders/1v5SMzqkjqHaC7YlimeY0exOFTsQ_onZU?usp=sharing

Download the folder(s) you need into this directory, then point
`inference_fcnn.py --checkpoint_path` at the desired checkpoint, e.g.:

```bash
python inference_fcnn.py \
    --checkpoint_path checkpoints/Panchromatic/U-Net/ResNet18/cp.080.ckpt \
    --input_path your_image.tif --output_folder . \
    --data_dim 1 --model_index 2 --downscale_factor 6 --mask_name mask.tif
```
