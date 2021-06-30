# [Why do These Match? Explaining the Behavior of Image Similarity Models (ECCV 2020)](https://arxiv.org/pdf/1905.10797.pdf)

 This repository implements:

Bryan A. Plummer*, Mariya I. Vasileva*, Vitali Petsiuk, Kate Saenko, David Forsyth.

[Why do These Match? Explaining the Behavior of Image Similarity Models. ECCV, 2020.](https://arxiv.org/pdf/1905.10797.pdf)

## Environment
This code was tested with Python 3.6 and Pytorch 1.4.

## Preparation

Download data and unzip it in ./data
- [Polyvore Outfits](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view?usp=sharing)
- [Animals with Attributes 2](https://cvml.ist.ac.at/AwA2/)

## Saliency Map Test
  To reproduce results from Table 1 in our paper you can use the saliency.py script in tools, e.g.,

  ```Shell
  python ./tools/saliency.py --fixed_ref --dataset polyvore_outfits --method rise
  ```

## Training Attribute Classifier

1. Cache saliency maps (used in Eq.2 of our paper)
  ```Shell
  python ./tools/saliency.py --fixed_ref --dataset polyvore_oufits --method rise --split train
  ```
2. Train the model

  ```Shell
  python ./tools/train_attribute_classifier.py --fixed_ref --dataset polyvore_outfits --method rise
  ```
### Citation
If you find our code useful please consider citing:

    @InProceedings{plummerSimilarityExplanations2020,
         author={Bryan A. Plummer and Mariya I. Vasileva and Vitali Petsiuk and Kate Saenko and David Forsyth},
         title={Why do These Match? Explaining the Behavior of Image Similarity Models},
         booktitle = {The European Conference on Computer Vision (ECCV)},
         year = {2020}
    }
