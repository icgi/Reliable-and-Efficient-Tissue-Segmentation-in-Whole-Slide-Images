# This is the code repo for our paper "Reliable and Efficient Tissue Segmentation in Whole-Slide Images" 

This repo is a fork of the official nnU-Net (https://github.com/MIC-DKFZ/nnUNetwith) an additional inference file (predict_tissue.py) created for simple, straightforward inference without prior knowledge of how the pipeline works. We have included minor modifications to the pipeline to simplify the inference run. This includes: 
- Removing the need to add a modality ID at the end of input scans (some_scan_0000.png, some_scan_0001.png, etc.)
- Outputting masks in a more user-friendly way ([0,255] instead of [0,1])
- Removing the need to specify a model path, as this is done automatically.

The model assumes that the input images are downsampled to 10 μm per pixel. The network is image dimension invariance by nature, but we can not guarantee quality segmentation if used at other resolutions.

### In our paper, we show the strong performance of our model and compare it to other baselines: 
![box_plot_of_dice_scores_logit_imagesTs (1)](https://github.com/user-attachments/assets/24297616-61a2-4319-b95d-a04aecd16082)

| Resolution (um/px) |     Model    | Dice score (%) | Inference time (s) |
| -----------------: | :----------- | -------------: | -----------------: |
|          5         | nnU-Net      |    98.51       |          5.88      |
|          5         | ResEnc       |    98.96       |         11.70      |
|         10         | nnU-Net      |    98.48       |          1.42      |
|         10         | ResEnc       |    98.87       |          3.09      |
|         20         | nnU-Net      |    98.46       |          0.44      |
|         20         | ResEnc       |    98.26       |          0.89      |
|          8         | Pathprofiler |    94.40       |          2.25      |



## Getting started
Firstly, download the models folder: 

```bash
pip install gdown
gdown 1WL_eB88yu6gr89AMdF4ktmaXpCCHknxS
unzip model.zip
```

Alternatively, you can download it directly from the gdrive link:

https://drive.google.com/file/d/1WL_eB88yu6gr89AMdF4ktmaXpCCHknxS/view?usp=sharing
Place and unzip the folder inside the project folder like this:

```plaintext
project_root/                                                              
├── models
│   ├── trained_on_10um
│   │   ├── checkpoint_10um.pth
│   │   ├── dataset_fingerprint.json
│   │   ├── dataset.json
│   │   └── plans.json
│   └── trained_on_10um_ResEnc
│       ├── checkpoint_10um_ResEnc.pth
│       ├── dataset_fingerprint.json
│       ├── dataset.json
│       └── plans.json
├── ... 
├── nnunetv2/                                                                                                
└──dockerfiles/    
```

## Setup
To set up the environment, simply build and run the Docker project within the dockerfiles folder:

```bash
cd /path/to/dockerfiles/
docker-compose build nnunet && docker-compose run nnunet
```

If you have already built a container, to avoid rebuilding every time, you can use:
```bash
docker-compose run nnunet
```

## Running inference for tissue segmentation

We explain the use of inference with the following parameters.
```bash
nnUNetv2_predict_tissue -i /path/to/images/ -o /path/to/output \
 -suffix suffix_name \
 -exclude exclusion_folder \
 -resenc \
 --b01 \
 --continue_prediction
```

By default, the nnU-Net model takes a folder of PNG images and runs inference on them. However, we also have included support for sending a txt list file containing paths to WSIs for inference. This approach will automatically downsample the images to 10um in the inference loop.

```bash
nnUNetv2_predict_tissue -i /path/to/images/ -o /path/to/output
```
or
```bash
nnUNetv2_predict_tissue -i /path/to/scan_list.txt -o /path/to/output
```

The layout for the text file should look like this:
```text
/path/to/scan1.suffix
/path/to/scan2.suffix
/path/to/scan3.suffix
/path/to/scan4.suffix
...
```

Alternatively, you can specify a path to WSIs with the inclusion of the file ending suffix (.svs, .ndpi, etc.).
```bash
nnUNetv2_predict_tissue -i /path/to/WSIs -o /path/to/output -suffix suffix_name
```

By setting the exclude flag, you can exclude unwanted folders from projects (excluded scans, etc.). 

```bash
nnUNetv2_predict_tissue -i /path/to/WSIs -o /path/to/output -suffix suffix_name -exclude exclusion_folder
```

By default, the standard nnUNetv2 model will be used. If you want to use the **residual encoder (ResEnc)** model, please use the **-resenc** flag. Please be aware that inference time will be slightly slower due to the complexity of the ResEnc network. 

```bash
nnUNetv2_predict_tissue -i /path/to/images -o /path/to/output -resenc
```

We have modified the pipeline to output [0,255] instead of the original [0,1] output. If you still want to have your segmentation as **binary [0,1]**. please use the **--b01** flag during inference:

```bash
nnUNetv2_predict_tissue -i /path/to/images -o /path/to/output --b01
```

If you have an incomplete run of segmentation masks, you can continue where the model stopped by running the continue_prediction flag.

```bash
nnUNetv2_predict_tissue -i /path/to/images -o /path/to/output --continue_prediction
```

If you plan on running inference on a folder of png images, you can use the help flag to check extra commands. However, some of these features are not yet supported for the txt input. 

```bash
nnUNetv2_predict_tissue -h
```

## Hardware requirements and inference times
We present average inference times for both models tested on an RTX3090 with 24GB of GPU memory. For the ResEnc architecture, you will need a minimum of 24GB of GPU memory:

|  Model | Avg. infernce time | Scan count |
| :------: | ------------------: | ----------: |
| nnUNet        |      1.42 seconds  |     100    |
| nnUNet ResEnc |      3.09 seconds  |     100    |
