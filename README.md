## CDAC: Cross-domain Attention Consistency in Transformer for Domain Adaptive Semantic Segmentation

Official release of the source code for [CDAC: Cross-domain Attention Consistency in Transformer for Domain Adaptive Semantic Segmentation](https://arxiv.org/abs/2211.14703) at ICCV 2023.

## Overview
We propose Cross-Domain Attention Consistency (CDAC), to perform adaptation on attention maps using cross-domain attention layers that share features between source and target domains. Specifically, we impose consistency between predictions from cross-domain attention and self-attention modules to encourage similar distributions across domains in both the attention and output of the model, i.e., attention-level and output-level alignment. We also enforce consistency in attention maps between different augmented views to further strengthen the attention-based alignment. Combining these two components, CDAC mitigates the discrepancy in attention maps across domains and further boosts the performance of the transformer under unsupervised domain adaptation settings. 
Our method is evaluated on various widely used benchmarks and outperforms the state-of-the-art baselines, including GTAV-to-Cityscapes by 1.3 and 1.5 percent point (pp) and Synthia-to-Cityscapes by 0.6 pp and 2.9 pp when combining with two competitive Transformer-based backbones, respectively.

## Installation and Data Preparation

Since our model is primarily built on the basis of DAFormer, please refer to the `Setup Environment` and the `Setup Datasets` section in the [original repo](https://github.com/lhoyer/DAFormer/) for instructions to set up the environment and prepare for the datasets.

## Training

For training our model on GTAV->Cityscapes:
```shell
python run_experiments.py --config configs/cdac/gta2cs_uda_dacs_cda_mitb5_b2_s0.py
```

For training our model on Synthia->Cityscapes:
```shell
python run_experiments.py --config configs/cdac/synthia2cs_uda_dacs_cda_mitb5_b2_s0.py
```

For training our model on Cityscapes->ACDC:
```shell
python run_experiments.py --config configs/cdac/cs2acdc_uda_dacs_cda_mitb5_b2_s0.py
```

## Testing

Our models pretrained on the three benchmarks are also saved and available online. Please kindly find them [here]([https://drive.google.com/file/d/1Zcb2E6or31_JgLFhaQgeT9UD-7TtkUyl/view?usp=sharing](https://www.dropbox.com/scl/fo/zshfbb85djhxuuu2qx32q/AN_oH5stBEqEE_CRcobFmMs?rlkey=pe0zqg3vf067ig8w9jbwpoiun&st=ecfe0mmh&dl=0)). After downloading the files, please run the following command:

```shell
sh test.sh path/to/checkpoint_directory
```

## Acknowledgements

The code of this project is heavily borrowed from DAFormer and its dependent repo. 
We thank their authors for making the source code publically available.

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
* [DAFormer](https://github.com/lhoyer/DAFormer)

