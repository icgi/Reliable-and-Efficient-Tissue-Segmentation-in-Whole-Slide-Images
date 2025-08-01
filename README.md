# Reliable and Efficient Tissue Segmentation in Whole-Slide Images 

This is the code repo for our paper "Reliable and Efficient Tissue Segmentation in Whole-Slide Images". The repository introduces a simple-to-use pipeline for segmenting Tissue in Whole-Slide images (WSIs). We include the pretrained weights for our best model as well as a Docker project for easy setup and use. Our model is based on the nnU-Net pipeline (https://github.com/MIC-DKFZ/nnUNetwith). We have included several modifications to the pipeline adapted for WSIs to simplify inference. This includes: 
- Removing the need to add a modality ID at the end of input scans (some_scan_0000.png, some_scan_0001.png, etc.)
- Outputting masks in a more user-friendly way ([0,255] instead of [0,1])
- Removing the need to specify a model path, as this is done automatically.

The model assumes that the input images are downsampled to 10 μm per pixel. The network is scale-invariant by nature, but we can not guarantee quality segmentation if used at other resolutions.

## Table of Contents
1. [Results](#results)
2. [Installation and Setup](#installation-and-setup)
3. [Running inference for tissue segmentation](#running-inference-for-tissue-segmentation)
4. [Hardware requirements and inference times](#hardware-requirements-and-inference-times)
5. [Acknowledgments and Disclosure of Funding](#acknowledgments-and-disclosure-of-funding)
6. [License](#license)

## Results
### In our paper, we show the strong performance of our model and compare it to other baselines: 

<img src="https://github.com/user-attachments/assets/24297616-61a2-4319-b95d-a04aecd16082" alt="drawing" width="700"/>
<img src="https://github.com/user-attachments/assets/ac0be4bd-2ee6-49c9-ae48-ecaf21ef51a5" alt="drawing" width="700"/>

We also test the trade-offs in performance and inference speed at different resolutions.

| Resolution (um/px) |     Model    | Dice score (%) | Inference time (s) |
| -----------------: | :----------- | -------------: | -----------------: |
|          5         | nnU-Net      |    98.51       |          5.88      |
|          5         | ResEnc       |    98.96       |         11.70      |
|         10         | nnU-Net      |    98.48       |          1.42      |
|         10         | ResEnc       |    98.87       |          3.09      |
|         20         | nnU-Net      |    98.46       |          0.44      |
|         20         | ResEnc       |    98.26       |          0.89      |

## Installation and Setup
Firstly, download the models folder containing the various model weights: 

```bash
pip install gdown
gdown 12jdV2BTT0sZX5SciBNNuwT_ZBQhITKdQ
unzip models.zip
```

Alternatively, you can download it directly from the Google drive link:

https://drive.google.com/file/d/12jdV2BTT0sZX5SciBNNuwT_ZBQhITKdQ/view?usp=sharing
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

To set up the environment, build and run the Docker project within the dockerfiles folder:

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
nnUNetv2_predict_tissue
 -i /path/to/images/          # Input path (expects path to PNGs or WSIs; alternatively, you can send a text file list with paths to WSIs.)
 -o /path/to/output \         # Output path (will be output in a flat layout unless the keep_parent flag is included).
 -suffix suffix_name \        # Define WSI file ending to search for (ndpi, svs, etc.).
 -exclude exclusion_folder \  # Name of folder to be ignored during search for slides in a path.
 -extension file_extension \  # Applies an extension to the file name, separated with a _.
 -pp rule \                   # Postprocessing. Takes preconfigured rules (lite/strict) or you can adjust yourself by setting each parameter separately. See full list with the help -h flag                 
 --keep_parent \              # Saves output in same directory layout as input scans.
 --resenc \                   # Use the Residual Encoder network (resource-heavy and significant increase in inference time).
 --lowres \                   # Use 20 um/px instead of 10 um/px model (Much faster with similar performance to original model).
 --b01 \                      # Get output in binary [0,1] instead of [0,255].
 --continue_prediction        # Continue prediction if output already exists in path. 
```

**By default, the nnU-Net model takes a folder of PNG images. However, we have also included support for sending a text file containing a list of paths to WSIs for inference. This approach will automatically downsample the images to 10um in the inference loop.**

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

### Suffix
Alternatively, you can specify a path to WSIs by including the file ending suffix (.svs, .ndpi, etc.).
```bash
nnUNetv2_predict_tissue -i /path/to/WSIs -o /path/to/output -suffix suffix_name
```

### Exclude

By setting the exclude flag, you can exclude unwanted folders from projects (excluded scans, etc.). 

```bash
nnUNetv2_predict_tissue -i /path/to/WSIs -o /path/to/output -suffix suffix_name -exclude exclusion_folder
```

### Extension
If your pipeline expects a certain extension to a filename, you can include extensions with -extension. It will automatically include "_" to separate the name and extension.

```bash
nnUNetv2_predict_tissue -i /path/to/WSIs -o /path/to/output -extension name_extension
```

### Post processing
In most cases, the nnUNet pipeline should already create clean outputs that don't require postprocessing. However, we include a few postprocessing options if needed. The -pp parameter can take in several rules. For simple use, we recommend using the two presets included (lite/strict). 

Lite rule:
 - Fill holes
 - Min area relative 0.002

Strict rule:
 - Keep the largest only
 - Fill holes
 - Min area 1000 pixels

```bash
nnUNetv2_predict_tissue -i /path/to/WSIs -o /path/to/output -pp (lite/strict)
```

For more advanced users, you can adjust each parameter manually by listing several pp arguments. Example:

```bash
nnUNetv2_predict_tissue -i /path/to/WSIs -o /path/to/output -pp min_area_rel=0.002 -pp fill_holes=True -pp close_r=8
```

### Keep parent
If you want output predictions to be saved in their respective parent folders, use the 'keep_parent' flag. The search for WSIs will occur recursively and can be performed on multiple levels; therefore, the output will vary depending on the starting path.

```bash
nnUNetv2_predict_tissue -i /path/to/WSIs -o /path/to/output -suffix suffix_name --keep_parent
```

For example, scans stored in unique ID folders, the structure would be saved like this:

```plaintext
output_folder/                                                              
├── scan_id_1
│   └── scan_id_1.suffix
├── scan_id_2
│   └── scan_id_2.suffix                                                                                               
└── ...    
```

### Lowres (20um/px)
While our paper illustrates our 10 um/px model, giving a good balance in accuracy and efficiency, the 20 um/px model can often be a good enough alternative. With more than 3 times faster segmentation speed, it only gives a slight reduction in segmentation accuracy. If you don't require highly accurate masks, we recommend using the 20 um/px model as the performance is very similar, with a great boost in inference speed. To use the 20 um/px model, use the lowres flag.

```bash
nnUNetv2_predict_tissue -i /path/to/images -o /path/to/output --lowres
```

### Resenc (residual encoder UNet)
By default, the standard nnUNetv2 model will be used. If you want to use the **residual encoder (ResEnc)** model, please use the **-resenc** flag. Please be aware that inference time will be slightly slower due to the complexity of the ResEnc network. 

```bash
nnUNetv2_predict_tissue -i /path/to/images -o /path/to/output --resenc
```

### Binary [0,1] output
We have modified the pipeline to output [0,255] instead of the original [0,1] output. To get **binary [0,1]** output, please use the **--b01** flag during inference:

```bash
nnUNetv2_predict_tissue -i /path/to/images -o /path/to/output --b01
```

### Continue prediction
If you have an incomplete run of segmentation masks, you can continue where the model stopped by running the continue_prediction flag.

```bash
nnUNetv2_predict_tissue -i /path/to/images -o /path/to/output --continue_prediction
```

### Help
The nnU-Net architecture contains additional arguments not listed here. Please use the help flag 'h' to see more. 

```bash
nnUNetv2_predict_tissue -h
```

## Hardware requirements and inference times
We present average inference times for both models tested on an RTX3090 with 24GB of GPU memory. For the ResEnc architecture, you will need a minimum of 24GB of GPU memory:

|  Model | Avg. infernce time | Scan count |
| :------: | ------------------: | ----------: |
| nnUNet        |      1.42 seconds  |     100    |
| nnUNet ResEnc |      3.09 seconds  |     100    |

## Acknowledgments and Disclosure of Funding
The code for this project is heavily based on the nnU-Net pipeline (https://github.com/MIC-DKFZ/nnUNetwith). We take no credit for the architecture or pipeline, and only introduce small adjustments for easier use in Pathology. 

We thank Krishanthi Harikaran, Ingrid Elise Weydahl, and Maria Isaksen for laboratory
assistance. We are also grateful to Zhen Qian for facilitating the acquisition of the dataset
from the Erasmus University Medical Center Cancer Institute. This work was supported
by the South-Eastern Norway Regional Health and Authority research fund (grant number
2024039) and The Norwegian Cancer Society (grant number 273051).

## License
This work is licensed under the **Creative Commons
Attribution-NonCommercial 4.0 International** license  
[https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc/4.0/).

