# Copyright (c) 2021 nEsensee, F. Et al.
# Modifications 2025 Institute for Cancer Genetics and Informatics
# Licensed under the Apache License, Version 2.0

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
import gc


from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json


from concurrent.futures import ThreadPoolExecutor, as_completed, FIRST_COMPLETED, wait

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


from PIL import PngImagePlugin

import re


LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from batchgenerators.utilities.file_and_folder_operations import load_pickle


def _preprocess_to_memory(case_id: str,
                          wsi_path: str,
                          pp: DefaultPreprocessor,
                          plans, cfg, ds_json,
                          lowres:bool):
    try:
        data, _, props = pp.run_case([wsi_path], None, plans, cfg, ds_json, lowres)
    except Exception as e:
        print(f'Failed {wsi_path}: {e}')
        return case_id, None, None, wsi_path
    return case_id, data, props, wsi_path




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
                                  list_of_lists_or_source_folder,
                                  output_folder_or_list_of_truncated_output_files,
                                  suffix: str, 
                                  extension: str,
                                  exclude: str,
                                  save_probabilities: bool = False,
                                  overwrite: bool = True,
                                  num_processes_preprocessing: int = default_num_processes,
                                  num_processes_segmentation_export: int = default_num_processes,
                                  folder_with_segs_from_prev_stage: str = None,
                                  num_parts: int = 1,
                                  part_id: int = 0,
                                  binary_01: bool = False,
                                  keep_parent: bool = False,
                                  lowres: bool = False,
                                  pp_cfg: dict = None):
        
        output_folder = output_folder_or_list_of_truncated_output_files
        maybe_mkdir_p(output_folder)



        if list_of_lists_or_source_folder.lower().endswith('.txt') or suffix is not None:
            print(f'[TissueNNUnetPredictor] Streaming WSI fully in memory...')
            self.predict_wsi_streaming(list_of_lists_or_source_folder, output_folder, suffix, extension, exclude, lowres=lowres, postproc_cfg=pp_cfg, overwrite=overwrite, cpu_workers=num_processes_preprocessing, binary_01=binary_01, keep_parent=keep_parent)

            return


        print(f'[TissueNNUnetPredictor] Detected img folder; using stock nnU-Net pipeline (blocking).')
        super().predict_from_files(
            list_of_lists_or_source_folder,
            output_folder,
            save_probabilities,
            overwrite,
            num_processes_preprocessing,
            num_processes_segmentation_export,
            folder_with_segs_from_prev_stage,
            num_parts,
            part_id,
            pp_cfg,
            extension
        )

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
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities, binary_01),)
                        )
                    )
                else:
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

    def _preprocess_wrapper(self, p, pp, plans, cfg, ds_json, lowres):
        return _preprocess_to_memory(
            Path(p).stem, p, pp, plans, cfg, ds_json, lowres
        )

    def _handle_result(self, fut, keep_parent, wsi_txt, cfg, plans, ds_json, postproc_cfg, extension, binary_01, output_folder):
        try:
            cid, np_data, props, wsi_path = fut.result()
        except Exception as e:
            print(f"Preprocessing failed for {wsi_path}: {e}")
            return

        # corrupted slides can make _preprocess_to_memory return None
        if np_data is None:
            print(f"Skipping failed slide: {wsi_path}")
            return

        try:
            print(f"Now predicting {cid}")
            logits = self.predict_logits_from_preprocessed_data(
                torch.from_numpy(np_data).to(self.device)
            ).cpu()

            if keep_parent:
                rel_parent = Path(wsi_path).relative_to(Path(wsi_txt)).parent
                png_file = os.path.join(output_folder, rel_parent, cid)
                os.makedirs(os.path.dirname(png_file), exist_ok=True)
            else:
                png_file = os.path.join(output_folder, cid)

            export_prediction_from_logits(
                logits, props, cfg, plans, ds_json,
                png_file, save_probabilities=False, binary_01=binary_01, postproc_cfg=postproc_cfg, extension=extension
            )

        except Exception as e:
            print(f"Prediction failed for {wsi_path}: {e}")

        finally:
            # ----- critical for not being OOM-killed -----
            del np_data, logits
            torch.cuda.empty_cache()
            gc.collect()  
            
    def predict_wsi_streaming(
            self,
            wsi_txt: str,
            output_folder: str,
            suffix: str,
            extension: str,
            exclude: str,
            overwrite: bool,
            cpu_workers: int = 8,
            binary_01: bool = False,
            keep_parent: bool = False,
            lowres: bool = False,
            postproc_cfg: dict = None
    ):
        if os.path.splitext(wsi_txt)[1] == '.txt':
            with open(wsi_txt) as f:
                wsi_paths = [ln.strip() for ln in f if ln.strip()] 
        else:
            wsi_txt = Path(wsi_txt)
            wsi_paths = list(wsi_txt.rglob(f'*{suffix}'))
            if exclude is not None:
                wsi_paths = [str(wsi) for wsi in wsi_paths if not exclude in str(wsi)]
            else:
                wsi_paths = [str(wsi) for wsi in wsi_paths]


        print(f"Found {len(wsi_paths)} scans")

        if not overwrite:
            files_for_inference = []
            predictions = list(Path(output_folder).rglob('*.png'))
            predictions = [p.stem for p in predictions]

            for wp in wsi_paths:
                #filename = os.path.basename(wp).split('.')[0]
                filename = Path(wp).stem
                if not filename in predictions:
                    files_for_inference.append(wp)

            wsi_paths = files_for_inference
            print(f"Overwrite is set to False. Running inference on remaining {len(wsi_paths)} scans.")
        else:
            print(f"Running inference on {len(wsi_paths)} scans.")

        pp = DefaultPreprocessor(verbose=False)
        plans, cfg, ds_json = self.plans_manager, self.configuration_manager, self.dataset_json

        max_in_flight = cpu_workers * 2
        pending = set()

        completed_counter = 0
        total_count = len(wsi_paths)

        with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
            for path in wsi_paths:
                pending.add(pool.submit(self._preprocess_wrapper, path, pp, plans, cfg, ds_json, lowres))

                if len(pending) >= max_in_flight:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        self._handle_result(fut, keep_parent=keep_parent, wsi_txt=wsi_txt, cfg=cfg, plans=plans, ds_json=ds_json, postproc_cfg=postproc_cfg, extension=extension, binary_01=binary_01, output_folder=output_folder)

                        completed_counter += 1
                        print(f"{completed_counter}/{total_count} scans completed...")

            for fut in pending:
                self._handle_result(fut, keep_parent=keep_parent, wsi_txt=wsi_txt, cfg=cfg, plans=plans, ds_json=ds_json, postproc_cfg=postproc_cfg, extension=extension, binary_01=binary_01, output_folder=output_folder)

                completed_counter += 1
                print(f"{completed_counter}/{total_count} scans completed...")

def predict_tissue_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    ########################## Set these parameters for inference ##########################
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-suffix', required=False, default=None,
                        help='Add suffix for what scanner type to look for when running inference on WSIs.')
    parser.add_argument('-exclude', required=False, default=None, 
                        help='Name of folder to be excluded when rglobing for WSIs in a directory.')
    parser.add_argument('--lowres', action='store_true', required=False, default=False,
                        help='Use this flag for a significant speed up in inference using the 20um model (segmentation quality might be slightly reduced).')
    parser.add_argument('-extension', required=False, default=None,
                        help='Add file extension to file. File extension will be added with a _.')
    parser.add_argument('--resenc', action='store_true', required=False, default=False,
                        help='Use the Residual encoder nnUNet (new recommended base model from author)')
    parser.add_argument('--b01', action='store_true', required=False, default=False,
                        help='Converts output masks to binary 0,1 instead of standard 0,255')
    parser.add_argument('--keep_parent', action='store_true', required=False, default=False,
                        help='Include this flag if you want each prediction to be saved in a parent folder in the same order as the source folder.')
    ########################################################################################


    ############################ PostProcessing Parameters ################################
    parser.add_argument('-pp',
                        action='append', default=[],
                        metavar="RULE",
                        help=(
                            'Post-processing rule(s). May be given multiple times. \n'
                            'Accepted tokens: \n'
                            ' keepLargest, fillHoles, strict, lite,\n'
                            ' minArea=<int>, minAreaRel=<float>, close_r=<int>'))
    #######################################################################################

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

    args = parser.parse_args()

    if args.pp:
        pp_cfg = dict(keep_largest=False, fill_holes=False,
                min_area=None, min_area_rel=None, close_r=0)
    
        presets = {
            'strict': dict(keep_largest=True, fill_holes=True, min_area=1000),
            'lite': dict(fill_holes=True, min_area_rel=0.002)
        }

        for t in args.pp:
            if t in presets:
                pp_cfg.update(presets[t]); continue
            if t == 'keepLargest':
                pp_cfg['keep_largest'] = True; continue
            if t == 'fillHoles':
                pp_cfg['fill_holes'] = True; continue

            m = re.fullmatch(r'minArea=(\d+)', t)
            if m: pp_cfg['min_area'] = int(m[1]); continue
            m = re.fullmatch(r'minAreaRel=([\d.]+)', t)
            if m: pp_cfg['min_area_rel'] = float(m[1]); continue
            m = re.fullmatch(r'close_r=(\d+)', t)
            if m: pp_cfg['close_r'] = int(m[1]); continue
    else:
        pp_cfg = args.pp


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

    resolution = '10' if not args.lowres else '20'

    if not args.resenc:
        print(f'Using {resolution}um model')
        predictor.initialize_from_trained_tissue_model_folder(
            f'/models/trained_on_{resolution}um',
            use_folds='all',
            checkpoint_name=f'checkpoint_{resolution}um.pth',
        )
    else:
        print(f'Using {resolution}um model with ResEnc architecture')
        predictor.initialize_from_trained_tissue_model_folder(
            f'/models/trained_on_{resolution}um_ResEnc',
            use_folds='all',
            checkpoint_name=f'checkpoint_{resolution}um_ResEnc.pth',
        )



    print(f"Time for loading model {time() - t0} seconds")

    predicted_segmentations = predictor.predict_tissue_from_files(args.i,
                                                           args.o,
                                                           save_probabilities=False,
                                                           overwrite=not args.continue_prediction,
                                                           suffix=args.suffix,
                                                           extension=args.extension,
                                                           exclude=args.exclude,
                                                           num_processes_preprocessing=args.npp,
                                                           num_processes_segmentation_export=args.nps,
                                                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                                           num_parts=args.num_parts,
                                                           part_id=args.part_id,
                                                           binary_01=args.b01,
                                                           keep_parent=args.keep_parent,
                                                           lowres=args.lowres,
                                                           pp_cfg=pp_cfg)
    t1 = time()

    print(f"Total time including model loading and inference {t1-t0} seconds")

if __name__ == '__main__':
    predict_tissue_entry_point()
