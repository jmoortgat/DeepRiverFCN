# Trained model checkpoints

Trained TensorFlow checkpoints (epoch 80) for all five architectures are
attached to the GitHub release:

- **Primary**: https://github.com/jmoortgat/DeepRiverFCN/releases/tag/v1.1.0
  - `DeepRiverFCN_checkpoints_Panchromatic.zip` (1-band models)
  - `DeepRiverFCN_checkpoints_Multispectral.zip` (4-band models)
- Mirror (original hosting): [Google Drive](https://drive.google.com/drive/folders/1v5SMzqkjqHaC7YlimeY0exOFTsQ_onZU?usp=sharing)

Unzip into this directory so the layout looks like
`checkpoints/Panchromatic/U-Net/ResNet18/cp.080.ckpt.*`, then point
`inference_fcnn.py --checkpoint_path` at the desired checkpoint, e.g.:

```bash
python inference_fcnn.py \
    --checkpoint_path checkpoints/Panchromatic/U-Net/ResNet18/cp.080.ckpt \
    --input_path your_image.tif --output_folder . \
    --data_dim 1 --model_index 2 --downscale_factor 6 --mask_name mask.tif
```

Directory names map to `--model_index` as: `DeepWaterMap` = 1 (the
ArcticRiverMap network), `U-Net/ResNet18` = 2, `U-Net/ResNet34` = 3,
`LinkNet/ResNet18` = 4, `LinkNet/ResNet34` = 5. Use `--data_dim 1` with
`Panchromatic` and `--data_dim 4` with `Multispectral` checkpoints.
