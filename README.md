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

### Publication

Full details of this work can be found in our paper "Deep Learning Models for River Classification at Sub-Meter Resolutions from Multispectral and Panchromatic Commercial Satellite Imagery", published in Remote Sensing of Environment, doi:10.1016/j.rse.2022.113279. Full citation:

Moortgat, J., Li, Z., Durand, M., Howat, I., Yadav, B. and Dai, C., 2022. Deep learning models for river classification at sub-meter resolutions from multispectral and panchromatic commercial satellite imagery. Remote Sensing of Environment, 282, p.113279.

@article{moortgat2022deep,
  title={Deep learning models for river classification at sub-meter resolutions from multispectral and panchromatic commercial satellite imagery},
  author={Moortgat, Joachim and Li, Ziwei and Durand, Michael and Howat, Ian and Yadav, Bidhyananda and Dai, Chunli},
  journal={Remote Sensing of Environment},
  volume={282},
  pages={113279},
  year={2022},
  publisher={Elsevier}
}

### Acknowledgement

Five fully convolutional neural network architectures are implemented based on the [Segmentation_models](https://github.com/qubvel/segmentation_models) and [DeepWaterMap](https://github.com/isikdogan/deepwatermap).
