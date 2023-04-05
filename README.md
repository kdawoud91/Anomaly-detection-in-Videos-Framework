# "Anomaly Detection in Surveillance Videos Framework"

## Dependencies
* Python 3.6
* PyTorch = 1.7.0 
* Detecron2
* Numpy
* Sklearn

## Datasets
* USCD Ped2 [[dataset](https://drive.google.com/file/d/1w1yNBVonKDAp8uxw3idQkUr-a9Gj8yu1/view?usp=sharing)]
* CUHK Avenue [[dataset](https://drive.google.com/file/d/1q3NBWICMfBPHWQexceKfNZBgUoKzHL-i/view?usp=sharing)]
* ShanghaiTech [[dataset](https://drive.google.com/file/d/1rE1AM11GARgGKf4tXb2fSqhn_sX46WKn/view?usp=sharing)]

Download the datasets into ``dataset`` folder, like ``./dataset/ped2/``, ``./dataset/avenue/``, ``./dataset/shanghai/``, ``./dataset/cifar100/``, ``./dataset/imagenet/``

## Detectron 2 preperation.
Install detectron 2 from [detectron2](https://github.com/facebookresearch/detectron2.git)
paste the files located in the "Detection_Files" folder to the main directory of Detectron2.
* Training baseline NLM
```bash
python train.py --dataset_type ped2
```

* Training skip frame based baseline NLM
```bash
python train.py --dataset_type ped2 --pseudo_anomaly_jump_inpainting 0.2 --jump 2 3 4 5
```
Select --dataset_type from ped2, avenue, or shanghai.

For more details, check train.py


## Evaluation
* Test the model
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth
```
* Test the model and save result image
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth --img_dir folder_path_to_save_image_results
```
* Test the model and generate demonstration video frames
```bash
python evaluate.py --dataset_type ped2 --model_dir path_to_weight_file.pth --vid_dir folder_path_to_save_video_results
```
Then compile the frames into video. For example, to compile the first video in ubuntu:
```bash
ffmpeg -framerate 10 -i frame_00_%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p video_00.mp4
```

```

## Acknowledgement
The code is built on top of code provided by Park et al. [ [github](https://github.com/cvlab-yonsei/MNAD) ] and Gong et al. [ [github](https://github.com/donggong1/memae-anomaly-detection) ] and Astrid et al. [https://github.com/aseuteurideu/LearningNotToReconstructAnomalis]
