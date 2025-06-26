from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from glob import glob


import inspect
import multiprocessing
import os
from copy import deepcopy
from time import sleep
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

class DoubleFilterNNUnetPredictor(nnUNetPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def downscale_to_10mpp(self, image, output_path):
        image = image[0]
        image_name = image.split('/')[-1]
        if image != "downscaled_images":
            img_output = f"{output_path}/{image_name.replace('_0000.png', '_10mpp_0000.png')}" 
            image = Image.open(image)
            new_size = (image.width //2, image.height // 2)
            downscaled_image = image.resize(new_size, Image.LANCZOS)
            downscaled_image.save(img_output)

        return [img_output]

    def apply_filter_to_image(self, image_path, mask_path, output):
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        mask = mask.convert("L")
        mask = mask.point(lambda p: 255 if p == 1 else 0)

        new_mask_size = (image.width, image.height)
        upscaled_mask = mask.resize(new_mask_size, Image.LANCZOS)

        white_background = Image.new("RGB", image.size, "white")
        filtered_image = Image.composite(image, white_background, upscaled_mask)


        image_name = image_path.split('/')[-1]
        output_file_path = f"{output}/{image_name}"
        filtered_image.save(output_file_path)

        return [output_file_path]

    def predict_double_filtering_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           downscaled_predictor,
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
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

        #### Save original input path for middle stage outputs ####
        input_path = list_of_lists_or_source_folder
        output_path = output_folder_or_list_of_truncated_output_files

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        #### Create output folders for middle stage outputs ####
        output_path_downscaled_predictions = output_folder + "/downscaled_images_predictions"
        maybe_mkdir_p(output_path_downscaled_predictions)
        downscaled_images_path = f"{output_folder}/downscaled_images"
        maybe_mkdir_p(downscaled_images_path)


        total_downscaled_images = os.listdir(downscaled_images_path)
        if len(list_of_lists_or_source_folder) == len(total_downscaled_images): 
            print("Downscaled images already exists. Skipping to next step")
            downscaled_images = [[f"{downscaled_images_path}/{image}"] for image in total_downscaled_images]
        else:
            #### Downscale images for first inference ####
            downscaled_images = []
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.downscale_to_10mpp, image, downscaled_images_path) for image in list_of_lists_or_source_folder}

                for future in tqdm(as_completed(futures), total=len(futures), desc="Downscaling images"):
                    downscaled_images.append(future.result())

        output_filename_truncated_downscaled = [join(output_path_downscaled_predictions, i[0].split("/")[-1].replace("_10mpp_0000.png", "")) for i in downscaled_images]
        total_downscaled_predictions = os.listdir(output_path_downscaled_predictions)


        if len(list_of_lists_or_source_folder) == len(total_downscaled_predictions):
            print("Downscaled image predictions already exists. Skipping to next step")
        else:
            #### Prepare and run inference on downscaled images ####
            data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(downscaled_images,
                                                                                    seg_from_prev_stage_files,
                                                                                    output_filename_truncated_downscaled,
                                                                                    num_processes_preprocessing)
            data_downscaled_iterator = downscaled_predictor._internal_get_data_iterator_from_lists_of_filenames(downscaled_images,
                                                                                    seg_from_prev_stage_files,
                                                                                    output_filename_truncated_downscaled,
                                                                                    num_processes_preprocessing)

            # self.predict_double_filtering_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)
            downscaled_predictor.predict_double_filtering_from_data_iterator(data_downscaled_iterator, save_probabilities, num_processes_segmentation_export)


        #### Prepare and run filtering on original images ####
        files_for_filtering = glob(f"{input_path}/*.png")
        # predictions_for_filtering = glob(f"{output_path_downscaled_predictions}/*.png")

        get_file_name = lambda file: file.split("/")[-1]
        predictions_for_filtering = [f"{output_path_downscaled_predictions}/{get_file_name(image).replace('_0000.png', '.png')}" for image in files_for_filtering]

        
        filtered_images_output = f"{output_folder}/filtered_images"
        maybe_mkdir_p(filtered_images_output)

        total_filtered_images = os.listdir(filtered_images_output)
        filtered_images = []

        if len(list_of_lists_or_source_folder) == len(total_filtered_images):
            print("Filtered images already exists. Skipping to next step.")
            filtered_images = [[f"{filtered_images_output}/{image}"] for image in total_filtered_images]
        else:
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.apply_filter_to_image, img, mask, filtered_images_output) for img, mask in zip(files_for_filtering, predictions_for_filtering)}

                for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering images"):
                    filtered_images.append(future.result())

        output_filename_truncated = [f"{output_path}/{get_file_name(image[0]).replace('_0000.png', '')}" for image in filtered_images]

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(filtered_images,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_double_filtering_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def predict_double_filtering_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
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
                              self.dataset_json, ofile, save_probabilities),)
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

def predict_double_masking_entry_point():

    # input_path = "/media/domore_rackstation4_vol1/sander/dev/masking_experiments/imagesTs_datasets/imagesTs"
    # output_path = "/media/domore_rackstation4_vol1/sander/dev/masking_experiments/predictions_output/prediction_imagesTs_new_preprocessing_double_filtering_pipeline_output"
    input_path = "/media/domore_rackstation4_vol1/sander/dev/masking_experiments/imagesTs_datasets/imagesTs"
    output_path = "/media/domore_rackstation4_vol1/sander/dev/masking_experiments/prediction_imagesTs_trained_on_10um_double_filtering_pipeline_output"

    # instantiate the nnUNetPredictor
    predictor = DoubleFilterNNUnetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
        
    )
    
    predictor_downscaled = DoubleFilterNNUnetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
        
    )

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset020_TissueSegmentation_final_repreprocessed/nnUNetTrainer__nnUNetPlans__2d'),
        use_folds='all',
        checkpoint_name='checkpoint_final.pth',
    )

    predictor_downscaled.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset021_TissueSegmentation_10um/nnUNetTrainer__nnUNetPlans__2d'),
        use_folds='all',
        checkpoint_name='checkpoint_final.pth'
    )


    # run double masking predictions
    predicted_segmentations = predictor.predict_double_filtering_from_files(input_path,
                                                           output_path,
                                                           predictor_downscaled,
                                                           save_probabilities=False, overwrite=True,
                                                           num_processes_preprocessing=2,
                                                           num_processes_segmentation_export=2,
                                                           folder_with_segs_from_prev_stage=None, num_parts=1,
                                                           part_id=0)

if __name__ == '__main__':
    predict_double_masking_entry_point()