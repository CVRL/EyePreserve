# EyePreserve: Identity-Preserving Iris Synthesis

Synthesis of same-identity biometric iris images, both for existing and non-existing identities while preserving the identity across a wide range of pupil sizes, is complex due to intricate iris muscle constriction mechanism, requiring a precise model of iris non-linear texture deformations to be embedded into the synthesis pipeline. This paper presents the first method of identity-preserving, pupil size-varying synthesis of iris images. This approach is capable of both synthesizing images of irises with different pupil sizes representing non-existing identities as well as non-linearly deforming iris images of the existing subjects given the segmentation mask of the target image. Iris recognition experiments suggest that the proposed deformation model not only preserves the identity when changing the pupil size, but offers better similarity between same-identity iris samples with significant difference in pupil size, compared to a state-of-the-art linear iris deformation model. Two immediate applications of the proposed approach are (a) synthesis of or enhancing existing biometric datasets for iris recognition, mimicking those acquired with iris sensors, and (b) helping forensic human experts in examining iris image pairs with significant differences in pupil dilation or constriction which can be light-induced and drug-induced.

![image](/assets/teaser.png)

### Here's a demo gif comparing the different iris deformation methods:

![comparison](/assets/comparison.gif)

### Here are samples of diseased iris images "rectified" by EyePreserve to look like healthy irises and outputs for the opposite, where a healthy iris was deformed to mimic irregular pupil shapes seen in diseased irises:

![diseasetonormal](/assets/disease_comp.png)

![normaltodisease](/assets/normal_to_disease.png)

### Here are sample gifs where the iris is deformed into arbitrary pupil shapes:
| Original | Cat Eye | Star | Heart | Double Pupil |
| --- | --- | --- | --- | --- |
| ![original](/assets/animation_source.png) | ![cateye](/assets/cateye_loop.gif) | ![star](/assets/star.gif) |  ![heart](/assets/heart.gif) | ![dualpupil](/assets/dualpupil.gif) |

> [!NOTE]
> Video files for all of the above GIFs are provided in the "sample_videos" folder.

# Demo GUI:
## Summary:

"LinearDeformer.py" contains code that linearly deforms the iris image to have a different pupil size based on Daugman's normalization [1]. "BiomechDeformer.py" contains code that deforms based on the biomechanical model proposed by Tomeo-Reyes et al. [2]. "EyePreserve.py" contains the deep autoencoder-based model, EyePreserve, that tries to mimic the complex movements of iris texture features directly from the data. The autoencoder model takes two inputs, (a) near-infrared iris image with initial pupil size, and (b) the binary mask defining the target shape of the iris. The model makes all the necessary nonlinear deformations to the iris texture to match the shape of the iris in an image with the shape provided by the target mask.

We have two GUI codes: "Synthesis_GUI.py" and "Comparison_GUI.py"

### Demo GIFs showing how the GUI works:

"Synthesis_GUI.py" uses StyleGAN3 to generate a random iris image which can then be modified using the Linear deformation model and EyePreserve as illustrated below:
![synthesis_gui](/assets/synthesis_gui.gif)

## How to run:

Download the models zip and extract it from here: [models.zip](https://notredame.box.com/s/us71ubwjzebxi2r015whrmdkb3rrtjn2)

Put the models directory in the same directory as LinearDeformer.py, EyePreserve.py and GUI.py

You can create a conda environment to run the code as follows:

1. Create the conda environment with the name of your choice 

> conda create -n \<name\>

> conda activate \<name\>

2. Run the appropriate pytorch installation instruction from www.pytorch.org based on your system. 

For MAC:

> conda install pytorch::pytorch torchvision torchaudio -c pytorch

For Linux or Windows (CPU only):

> conda install pytorch torchvision torchaudio cpuonly -c pytorch

3. Install the required packages in conda environment.

> conda install -c conda-forge opencv

> conda install -c anaconda scikit-learn

> pip install -U Pillow kornia[x] ninja click

And the GUI using:

> python Synthesis_GUI.py

If you have CUDA support (you need to have a CUDA-supported GPU and have the appropriate version of pytorch installed), you can run the GUIs with CUDA enabled:

> python Synthesis_GUI.py cuda

The video illustrating what the GUI does is provided in the "sample_videos" folder as well. 

**** The files in "dnnlib" and "torch_utils" are from the StyleGAN3 pytorch github repo to load and use the StyleGAN3 generator.

# Training codes

To the run the training codes you require the WBPD and CSOISPAD dataset divided into bins based on pupil-to-iris ratios.

To run the training code, run:

python train_quadruplets_mask_ldplus_adv_vec.py --cuda ---cudnn --parent_dir_wsd <path_to_wbpd> --train_bins_path_wsd <path_to_train_bin_info_for_wbpd> --val_bins_path_wsd <path_to_val_bin_info_for_wbpd> --parent_dir_csoispad <path_to_csoispad> --train_bins_path_csoispad <path_to_train_bin_info_for_csoispad> --val_bins_path_csoispad <path_to_val_bin_info_for_csoispad> --use_lpips_loss --use_msssim_loss --use_iso_loss --use_patch_adv_loss --no_ld_in --ema'

References:

[1] J. Daugman, "How iris recognition works." In The Essential Guide to Image Processing, pp. 715-739. Academic Press, 2009.

[2] I. Tomeo-Reyes, A. Ross, A. D. Clark and V. Chandran, "A biomechanical approach to iris normalization," 2015 International Conference on Biometrics (ICB), Phuket, Thailand, 2015, pp. 9-16, doi: 10.1109/ICB.2015.7139041

