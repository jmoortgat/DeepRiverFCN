# Rivers Segmentation using FCNs

### Requirements
- TensorFlow (tested on tensorflow-gpu==2.4.1 python==3.8.5)
- OpenCV
- rasterio
- xarray 
- rioxarray
- geopandas

To create the TensorFlow environment for training and inference please use the file [environment.yml](https://github.com/Ziwei-0129/DeepRiverFCNNs/blob/main/environment.yml)

### Running Our Codes

**Model Training:**
```python
python train_fcnn.py --checkpoint_dir ckpts --data_path TensorFlowRecords --figure_path figs --data_dim 1 --model_index 1 --num_epoch 2 --batch_size 24 --learning_rate 0.1
```

**Inference:**

Five fully convolutional neural network models are supported in our codes. You can choose different type of FCNNs through "--model_index" option from 1 to 5:
```
1: DeepWaterMap
2: UNet with ResNet18 backbone
3: UNet with ResNet34 backbone
4: LinkNet with ResNet18 backbone
5: LinkNet with ResNet34 backbone
```

You can specify the channel number of your input data using the "--data_dim" option (e.g., 1 for panchromatic or 4 for 4-band images).

Our model checkpoints can be downloaded from [checkpoints](https://drive.google.com/drive/folders/1v5SMzqkjqHaC7YlimeY0exOFTsQ_onZU?usp=sharing)

```python
python inference_fcnn.py --checkpoint_path checkpoints/Panchromatic/U-Net/ResNet18/cp.080.ckpt --input_path test_tiffimg_pan.tif --output_folder . --data_dim 1 --model_index 2 --downscale_factor 6 --mask_name mask_pan.tif
```

### Acknowledgement

Five fully connected neural network architectures are implemented based on the [Segmentation_models](https://github.com/qubvel/segmentation_models) and [DeepWaterMap](https://github.com/isikdogan/deepwatermap).
