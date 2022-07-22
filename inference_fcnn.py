from __future__ import division
import argparse
import numpy as np
import cv2, time, os, pickle, sys

import rasterio
from rasterio.enums import Resampling
from copy import deepcopy
import segmentation_models as sm
import arcticrivermap4
import arcticrivermappan
 
import xarray as xr
import tifffile as tiff
import rioxarray 
import pandas as pd
import geopandas as gpd 
import tensorflow as tf

# # Use CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def find_padding(v, divisor=64):
    v_divisible = max(divisor, int(divisor * np.ceil(v / divisor)))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2


def inference(model, image, data_dim):

    # load and preprocess the input image 
    if data_dim != 1:
        [ny, nx, bands] = np.shape(image)
        if(bands==8):
            image = np.delete(image,[0,3,5,7], 2) 
        elif(bands==7) :
            image = np.delete(image, 6, 2) # LandSat
   
    print("size of image:", np.shape(image), "min/max/mean", np.min(image), np.max(image), np.mean(image))

    pad_r = find_padding(image.shape[0])
    pad_c = find_padding(image.shape[1])

    if data_dim == 1: 
        image = np.pad(image, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1])), 'reflect')
    else:
        image = np.pad(image, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0,0)), 'reflect')

    image = image.astype(np.float32)
    image = image - np.min(image)
    image = image / np.maximum(np.max(image), 1)

    print("Image properties", type(image), np.shape(image), np.min(image), np.max(image))

    # run inference
    image = np.expand_dims(image, axis=0)
    inference = model.predict(image)
    inference = np.squeeze(inference)
    inference = inference[pad_r[0]:-pad_r[1], pad_c[0]:-pad_c[1]]

    # soft threshold
    inference = 1./(1+np.exp(-(16*(inference-0.5))))
    inference = np.clip(inference, 0, 1)

    return inference

 

def main(args):
     
    with rasterio.open(args.input_path) as dataset:

        if args.data_dim == 1:
            print('Orign Dim: ', dataset.shape)
            image = dataset.read(out_shape=(dataset.count,
                                            int(dataset.shape[-2]/args.downscale_factor),
                                            int(dataset.shape[-1]/args.downscale_factor)),
                                resampling=Resampling.bilinear)

            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / image.shape[-1]),
                (dataset.height / image.shape[-2]))

            # image crop:
            if args.central_fraction != None:
                image = tf.image.central_crop(image, args.central_fraction)
            image = np.squeeze(image)
            image = np.array(image) 
            print('New Dim: ', image.shape)
        else:
            image = dataset.read() 
            if (np.ndim(image) == 3):
                image = np.moveaxis(image, 0, -1)
            # image crop:
            if args.central_fraction != None:
                image = tf.image.central_crop(image, args.central_fraction)

    # Load inference model 
    model = None
    try:
        if args.model_index == 1 and args.data_dim == 4:
            model = arcticrivermap4.model()
        elif args.model_index == 1 and args.data_dim == 1:
            model = arcticrivermappan.model()
        elif args.model_index == 2:
            model = sm.Unet(backbone_name='resnet18', input_shape=(None, None, args.data_dim), 
                                encoder_weights=None, classes=1, activation='sigmoid')
        elif args.model_index == 3:
            model = sm.Unet(backbone_name='resnet34', input_shape=(None, None, args.data_dim), 
                                encoder_weights=None, classes=1, activation='sigmoid')
        elif args.model_index == 4:
            model = sm.Linknet(backbone_name='resnet18', input_shape=(None, None, args.data_dim), 
                                encoder_weights=None, classes=1, activation='sigmoid')
        elif args.model_index == 5:
            model = sm.Linknet(backbone_name='resnet34', input_shape=(None, None, args.data_dim), 
                                encoder_weights=None, classes=1, activation='sigmoid')
    except:
        print('please recheck the supporting neural networks and backbones') 

    model.load_weights(args.checkpoint_path)

    mask = inference(model, image, args.data_dim)
    mask = np.array(np.round((mask) * 255, 0), dtype=np.uint8)
 
    kwargs = dataset.meta
    kwargs.update(
        dtype=rasterio.uint8, 
        count=1,
        compress='lzw'
    )

    output_tif = os.path.join(args.output_folder, args.mask_name) 

    with rasterio.open(output_tif, 'w', **kwargs) as dst:
        dst.write_band(1, mask.astype(rasterio.uint8))    
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help="Path to the dir where the checkpoints are saved")
    parser.add_argument('--input_path', type=str, help="Path to the input image") 
    parser.add_argument('--output_folder', type=str, help="Path to the output folder") 
    parser.add_argument('--mask_name', type=str, default="mask.tif", help="Name of output mask") 
    parser.add_argument('--data_dim', type=int, help="Dimension of the training data, e.g., 1 or 4")
    parser.add_argument('--model_index', type=int, help="Index of the FCNN model")
    parser.add_argument('--downscale_factor', default=None, type=int, help="Image downscaling ratio for panchromatic images")
    parser.add_argument('--central_fraction', default=None, type=float, help="Index of the FCNN model")
    
    args = parser.parse_args() 

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    main(args) 
 

