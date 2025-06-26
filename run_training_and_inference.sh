nnUNetv2_train 9 2d all -p nnUNetResEncUNetLPlans && \
echo "Training completed successfully" || { echo "Training failed"; exit 1; }
nnUNetv2_predict -i /mnt/raid0/nnUNet_raw/Dataset009_TissueSegmentation_final/imagesTs/ -o /mnt/raid0/tissueSegmentation_ResEnc_final_test_data_output -d 9 -c 2d -f all -p nnUNetResEncUNetLPlans
