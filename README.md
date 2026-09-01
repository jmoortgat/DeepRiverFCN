# DeepRiverFCN — Arctic river segmentation with fully convolutional networks

[![License: MIT](https://img.shields.io/badge/License-MIT-BA0C2F.svg?style=flat-square)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-RSE%202022-2563EB.svg?style=flat-square)](https://doi.org/10.1016/j.rse.2022.113279)
[![Python](https://img.shields.io/badge/Python-3.8%E2%80%933.11-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

TensorFlow code to segment (Arctic) rivers in sub-meter-resolution
panchromatic and 4-band multispectral commercial satellite imagery, using
five fully convolutional neural network architectures. This is the code
behind:

> Moortgat, J., Li, Z., Durand, M., Howat, I., Yadav, B., & Dai, C. (2022).
> Deep learning models for river classification at sub-meter resolutions
> from multispectral and panchromatic commercial satellite imagery.
> *Remote Sensing of Environment*, **282**, 113279.
> [doi:10.1016/j.rse.2022.113279](https://doi.org/10.1016/j.rse.2022.113279)

## Models

Select the architecture with `--model_index`:

| Index | Model |
|:---:|---|
| 1 | ArcticRiverMap (DeepWaterMap-V2-style encoder–decoder) |
| 2 | U-Net, ResNet-18 backbone |
| 3 | U-Net, ResNet-34 backbone |
| 4 | LinkNet, ResNet-18 backbone |
| 5 | LinkNet, ResNet-34 backbone |

Set the number of input bands with `--data_dim` (1 = panchromatic,
4 = multispectral). Models 2–5 come from the
[segmentation_models](https://github.com/qubvel/segmentation_models)
library; model 1 is defined in [`arcticrivermap.py`](arcticrivermap.py).

## Installation

```bash
git clone https://github.com/jmoortgat/DeepRiverFCN.git
cd DeepRiverFCN
pip install -r requirements.txt        # or: conda env create -f environment.yml
```

Works on modern TensorFlow (≥ 2.16) via the `tf_keras` compatibility shim
(the scripts set `TF_USE_LEGACY_KERAS=1` automatically) as well as on the
originally tested stack (TF 2.4, Python 3.8).

## Usage

**Training** (expects TFRecord data; see the paper for the dataset
construction):

```bash
python train_fcnn.py --checkpoint_dir ckpts --data_path TensorFlowRecords \
    --figure_path figs --data_dim 1 --model_index 1 \
    --num_epoch 2 --batch_size 24 --learning_rate 0.1
```

**Inference** on a GeoTIFF, with a trained checkpoint:

```bash
python inference_fcnn.py \
    --checkpoint_path checkpoints/Panchromatic/U-Net/ResNet18/cp.080.ckpt \
    --input_path test_tiffimg_pan.tif --output_folder . \
    --data_dim 1 --model_index 2 --downscale_factor 6 --mask_name mask_pan.tif
```

## Trained checkpoints

Checkpoints for all five architectures (panchromatic and multispectral,
epoch 80) are attached to the
[v1.1.0 release](https://github.com/jmoortgat/DeepRiverFCN/releases/tag/v1.1.0)
(with the original
[Google Drive](https://drive.google.com/drive/folders/1v5SMzqkjqHaC7YlimeY0exOFTsQ_onZU?usp=sharing)
as a mirror); see [`checkpoints/README.md`](checkpoints/README.md) for the
expected layout.

## Training data

The training and validation imagery are commercial sub-meter satellite
scenes (WorldView) whose license does not permit redistribution, so the
labeled training dataset cannot be shared publicly. U.S. federally funded
researchers can obtain the underlying imagery through the
[Polar Geospatial Center](https://www.pgc.umn.edu/); the trained
checkpoints above let anyone apply the models without the training data.

## Citation

If you use this code or the trained models, please cite the paper (see
[`CITATION.cff`](CITATION.cff)):

```bibtex
@article{moortgat2022deep,
  title={Deep learning models for river classification at sub-meter
         resolutions from multispectral and panchromatic commercial
         satellite imagery},
  author={Moortgat, Joachim and Li, Ziwei and Durand, Michael and
          Howat, Ian and Yadav, Bidhyananda and Dai, Chunli},
  journal={Remote Sensing of Environment},
  volume={282},
  pages={113279},
  year={2022},
  doi={10.1016/j.rse.2022.113279}
}
```

## Acknowledgements

Models 2–5 are built with Pavel Iakubovskii's
[segmentation_models](https://github.com/qubvel/segmentation_models)
library. The ArcticRiverMap architecture follows DeepWaterMap-V2 by
[Isikdogan, Bovik & Passalacqua (2019)](https://doi.org/10.1109/LGRS.2019.2953261).
