# RGINP: Reference Guided Image Inpainting <br> using Facial Attributes

This repository is a official Pytorch implementation of RGINP.

![Teaser Image](imgs/main_img.jpg)

## [Paper](https://arxiv.org/abs/2301.08044)
>**"Reference Guided Image Inpainting using Facial Attributes"** <br>
>Dongsik Yoon, Jeong-gi Kwak, Yuanming Li, David K Han, Youngsaeng Jin and Hanseok Ko<br>
>**British Machine Vision Conference (BMVC), 2021** <br>

## Architecture
![Architecture](imgs/architecture.jpg)

$I_{masked}$ and $M$ are the input of Enc, we omit $M$ in this figure to express clearly our framework. For the test stage (red line), the user extract desired attributes using our attributes extractor to a reference image.

## Dependencies
- pytorch
- numpy
- Python3
- munch
- Pillow

## **Preparing datasets**
We utilize all the experiments in this paper using [CelebA-HQ dataset](https://github.com/tkarras/progressive_growing_of_gans/tree/original-theano-version) and [Quick Draw Irregular Mask dataset](https://github.com/karfly/qd-imd).
Please download the datasets and then construct them as shown below.<br>
If you want to learn and evaluate different images you have to change `dataloader.py`.

```
    --data
      --train
        --CelebAMask-HQ-attribute-anno.txt
        --CelebA_HQ
          --0.jpg
            ⋮
          --27999.jpg
      --test
        --CelebA_HQ
          --28000.jpg
            ⋮
          --29999.jpg
      --masks
        --train
            --00000_train.png
              ⋮
        --test
            --00000_test.png
              ⋮
        
    --RGINP
        ⋮
```

## Training
Please select the desired attributes from the `CelebAMask-HQ-attribute-anno.txt`. <br>
The default attributes of this experiment are as follows.
```bash
attrs_default = ['Bushy_Eyebrows', 
                'Mouth_Slightly_Open', 
                'Big_Lips', 
                'Male', 
                'Mustache', 
                'Young', 
                'Smiling', 
                'Wearing_Lipstick',
                'No_Beard']
```

To train the model:
```bash
python main.py --mode train
```

To resume the model:
```bash
python main.py --mode train --resume_iter 52500
```

## Validation
```bash
python main.py --mode val --resume_iter 200000
```

## Test
To test the model, you need to provide an input image, a reference image, and a mask file. <br>
Please make sure that the mask file covers the entire mask region in the input image. <br>
All file names must be the same and constructed as shown below.

```
    --RGINP
        ⋮
        --user_test
            --test_result
            --user_input
                --image
                    --ref
                        --sample.jpg   # reference image
                    --src
                        --sample.jpg   # input image
                --mask
                    --sample.jpg       # mask image
```

To test the model:
```bash
python main.py --mode test --resume_iter 200000
```

## Citation
```
@article{bmvc2021_RGINP,
  title={Reference Guided Image Inpainting using Facial Attributes},
  author={Yoon, Dongsik and Kwak, Jeonggi and Li, Yuanming and Han, David and Jin, Youngsaeng and Ko, Hanseok},
  journal={arXiv preprint arXiv:2301.08044},
  year={2023}
}
```

## Acknowledgments
RGINP is bulided upon the [LBAM](https://github.com/Vious/LBAM_Pytorch) implementation and inspired by [MLGN](https://github.com/JieLiu95/MLGN). <br> We appreciate the authors' excellent work!
