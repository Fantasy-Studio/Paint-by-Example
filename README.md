# Paint by Example: Exemplar-based Image Editing with Diffusion Models
![Teaser](figure/teaser.png)
### [Project page (Coming Soon)]() |   [Paper](https://arxiv.org/abs/2211.13227) 
<!-- <br> -->
[Binxin Yang](https://orcid.org/0000-0003-4110-1986), [Shuyang Gu](http://home.ustc.edu.cn/~gsy777/), [Bo Zhang](https://bo-zhang.me/), [Ting Zhang](https://www.microsoft.com/en-us/research/people/tinzhan/), [Xuejin Chen](http://staff.ustc.edu.cn/~xjchen99/), [Xiaoyan Sun](http://staff.ustc.edu.cn/~xysun720/), [Dong Chen](https://www.microsoft.com/en-us/research/people/doch/) and [Fang Wen](https://www.microsoft.com/en-us/research/people/fangwen/).
<!-- <br> -->

## Abstract
>Language-guided image editing has achieved great success recently. In this paper, for the first time, we investigate exemplar-guided image editing for more precise control. We achieve this goal by leveraging self-supervised training to disentangle and re-organize the source image and the exemplar. However, the naive approach will cause obvious fusing artifacts. We carefully analyze it and propose an information bottleneck and strong augmentations to avoid the trivial solution of directly copying and pasting the exemplar image. Meanwhile, to ensure the controllability of the editing process, we design an arbitrary shape mask for the exemplar image and leverage the classifier-free guidance to increase the similarity to the exemplar image. The whole framework involves a single forward of the diffusion model without any iterative optimization. We demonstrate that our method achieves an impressive performance and enables controllable editing on in-the-wild images with high fidelity.
>
## News

- *2022-11-29* Upload code.

## Requirements
A suitable [conda](https://conda.io/) environment named `Paint-by-Example` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate Paint-by-Example
```

## Pretrained Model
We provide the checkpoint ([Google Drive](https://drive.google.com/file/d/11HSX11bIVzsPuv7-cIsANN2BkzuX5pLm/view?usp=share_link)) that is trained on Open-Images for 40 epochs. By default, we assume that the pretrained model is downloaded and saved to the directory `checkpoints`.

## Testing

To sample from our model, run the following, you can use `scripts/inference.py`. For example, 
```
python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_1.png \
--mask_path examples/mask/example_1.png \
--reference_path examples/reference/example_1.jpg \
--seed 321 \
--scale 5
```
or simply run:
```
sh test.sh
```
Additional notes to consider: 
- During inference, the options used during training are loaded from the saved checkpoint and are then updated using the 
test options passed to the inference script. For example, there is no need to pass `--dataset_type` or `--label_nc` to the 
 inference script, as they are taken from the loaded `opts`.
- When running inference for segmentation-to-image or sketch-to-image, it is highly recommend to do so with a style-mixing,
as is done in the paper. This can simply be done by adding `--latent_mask=8,9,10,11,12,13,14,15,16,17` when calling the 
script.
- When running inference for super-resolution, please provide a single down-sampling value using `--resize_factors`.
- Adding the flag `--couple_outputs` will save an additional image containing the input and output images side-by-side in the sub-directory
`inference_coupled`. Otherwise, only the output image is saved to the sub-directory `inference_results`.
- By default, the images will be saved at resolutiosn of 1024x1024, the original output size of StyleGAN. If you wish to save 
outputs resized to resolutions of 256x256, you can do so by adding the flag `--resize_outputs`.


## Training
Coming Soon.

## Acknowledgements

- This code borrows heavily from [Stable Diffusion](https://github.com/CompVis/stable-diffusion). We also thank the contributors of [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion) and [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).

## License
The codes and the pretrained model in this repository are under the CreativeML OpenRAIL M license as specified by the LICENSE file.
