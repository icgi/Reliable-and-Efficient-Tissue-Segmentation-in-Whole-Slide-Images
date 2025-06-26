from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from pathlib import Path
import openslide


import inspect
import multiprocessing
import os
from copy import deepcopy
from time import sleep, time
from typing import Tuple, Union, List, Optional


from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json


from concurrent.futures import ThreadPoolExecutor, as_completed

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

import nnunetv2
from torch._dynamo import OptimizedModule
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class TissueNNUnetPredictor(nnUNetPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_from_trained_tissue_model_folder(self, model_training_output_dir: str,
                                             use_folds='all',
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)


        parameters = []
        
        checkpoint = torch.load(join(model_training_output_dir, checkpoint_name),
                                map_location=torch.device('cpu'))
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']
        inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
            'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def predict_tissue_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0,
                           binary_01: bool = False):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_tissue_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export, binary_01)

    def predict_tissue_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes,
                                   binary_01: bool = False):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                    #                               self.dataset_json, ofile, save_probabilities)
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities, binary_01),)
                        )
                    )
                else:
                    # convert_predicted_logits_to_segmentation_with_correct_shape(
                    #             prediction, self.plans_manager,
                    #              self.configuration_manager, self.label_manager,
                    #              properties,
                    #              save_probabilities)

                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

def convert_output_name(slide):
    dataset = slide.split('/')[4]
    slide = Path(slide)
    name = ''

    if dataset in ['Q64', 'S77', 'R75', 'R27']:
        scanner = slide.parent.parent.name

        if scanner == 'Aperio' or scanner == 'AP':
            start_part = f'LEICA_APERIO_{dataset}___LEICA_APERIO'
            name = f'{start_part}___{slide.stem}.png'
        elif scanner == 'XR':
            start_part = f'HAMAMATSU_NANOZOOMER_{dataset}___HAMAMATSU_NANOZOOMER'
            name = f'{start_part}___{slide.stem}.png'

    elif dataset in ['S20_SM', 'S98']:
        start_part = f'3DHISTECH_PANNORAMIC_{dataset}___3DHISTECH_PANNORAMIC'
        name = f'{start_part}___{slide.stem}.png'
    elif dataset == 'T18':
            start_part = f'HAMAMATSU_NANOZOOMER_{dataset}___HAMAMATSU_NANOZOOMER'
            name = f'{start_part}___{slide.stem}.png'

    return name

def convert_wsi_to_mpp(
    filepaths,
    desired_mpp: float = 16.3745,
    output_dir: Path = None,
    save_format: str = "tiff",
    auto_crop: bool = True
):
    """
    For each WSI in `filepaths`, open it, downsample/resample so that
    the output has exactly `desired_mpp` microns per pixel, optionally auto-crop
    to tissue, and save.

    Args:
        filepaths (List[str or Path]): list of WSI filenames.
        desired_mpp (float): target microns-per-pixel.
        output_dir (Path): directory where outputs go (defaults to cwd).
        save_format (str): one of "tiff", "png", "jpeg", etc.
        auto_crop (bool): if True, crops output to non-black tissue region.
    """
    output_dir = Path(output_dir or Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)

    for fp in map(Path, filepaths):
        if not fp.exists():
            print(f"[WARN] File not found: {fp}")
            continue

        file_name = convert_output_name(str(fp)).replace('.png', '')
        out_name = f"{file_name}.{save_format}"
        out_path = output_dir / out_name

        if out_path.exists():
            print(f"File already exists! Skipping...")
            continue

        try:
            slide = openslide.OpenSlide(str(fp))

            # 1) Read base MPP
            exit()
            mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X) \
                    or slide.properties.get("openslide.mpp-x")
            if mpp_x is None:
                print(f"[WARN] No MPP metadata for {fp.name}, skipping.")
                continue
            mpp_x = float(mpp_x)

            # 2) Choose pyramid level closest to desired_mpp
            downs = slide.level_downsamples  # floats
            mpp_levels = [mpp_x * ds for ds in downs]
            lvl = min(range(len(mpp_levels)), key=lambda i: abs(mpp_levels[i] - desired_mpp))
            native_mpp = mpp_levels[lvl]
            dims = slide.level_dimensions[lvl]

            # 3) Read full region at that level
            img_full = slide.read_region((0, 0), lvl, dims).convert("RGB")

            # 4) Optional auto-crop to non-black
            if auto_crop:
                arr = np.array(img_full)
                mask = np.any(arr != 0, axis=2)
                ys, xs = np.where(mask)
                if ys.size == 0:
                    print(f"[WARN] No tissue detected in {fp.name}, saving full slide.")
                    img_crop = img_full
                else:
                    y0, y1 = ys.min(), ys.max()
                    x0, x1 = xs.min(), xs.max()
                    # crop box = (left, upper, right, lower)
                    img_crop = img_full.crop((x0, y0, x1+1, y1+1))
            else:
                img_crop = img_full

            # 5) Rescale if needed to hit exactly desired_mpp
            scale = native_mpp / desired_mpp
            if abs(scale - 1.0) > 1e-3:
                new_size = (int(img_crop.width * scale), int(img_crop.height * scale))
                img_out = img_crop.resize(new_size, Image.LANCZOS)
            else:
                img_out = img_crop

            # 6) Save result
            img_out.save(str(out_path))
            print(f"→ Saved {out_path}")
        except Exception as e:
            print(f'Skipping scan {file_name}: {e}')

def predict_tissue_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('--b01', action='store_true', required=False, default=False,
                        help='Converts output masks to binary 0,1 instead of standard 0,255')
    parser.add_argument('-resenc', action='store_true', required=False, default=False,
                        help='Use the Residual encoder nnUNet (new recommended base model from author)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()


    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    t0 = time()

    predictor = TissueNNUnetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar)


    if not args.resenc:
        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_tissue_model_folder(
            '/models/trained_on_10um',
            use_folds='all',
            checkpoint_name='checkpoint_10um.pth',
        )
    else:
        predictor.initialize_from_trained_tissue_model_folder(
            '/models/trained_on_10um_ResEnc',
            use_folds='all',
            checkpoint_name='checkpoint_10um_ResEnc.pth',
        )
    print(f"Time for loading model {time() - t0} seconds")
        # # initialize the ResEnc version of the model
        # predictor.initialize_from_trained_tissue_model_folder(
        #     join(nnUNet_results, 'Dataset021_TissueSegmentation_10um/nnUNetTrainer__nnUNetResEncUNetLPlans__2d'),
        #     use_folds='all',
        #     checkpoint_name='checkpoint_final.pth',
        # )

    predicted_segmentations = predictor.predict_tissue_from_files(args.i,
                                                           args.o,
                                                           save_probabilities=False,
                                                           overwrite=True,
                                                           num_processes_preprocessing=args.npp,
                                                           num_processes_segmentation_export=args.nps,
                                                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                                           num_parts=args.num_parts,
                                                           part_id=args.part_id,
                                                           binary_01=args.b01)
    t1 = time()

    print(f"Total time including model loading and inference {t1-t0} seconds")

if __name__ == '__main__':
    predict_tissue_entry_point()