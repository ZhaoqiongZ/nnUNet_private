on CPU

export env
```
export nnUNet_raw_data_base="/home/zzq/winning/nnUNet/nnUNet_raw_data_base"
export nnUNet_preprocessed="/home/zzq/winning/nnUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/zzq/winning/nnUNet/nnUNet_trained_models"

```

convert and preprocess data to standard format, tooth already convert, so skip this step
```
# nnUNet_convert_decathlon_task -i /home/zhaoqion/winning/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task01_Tooth -p 56
mv /home/zhaoqion/winning/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task01_Tooth  /home/zhaoqion/winning/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task501_Tooth
nnUNet_plan_and_preprocess -t 501 --verify_dataset_integrity

```

train with tooth data

```
# change do_mirror = False in /home/zhaoqion/winning/nnUNet/nnunet/training/data_augmentation/default_data_augmentation.py
nnUNet_train 3d_fullres nnUNetTrainerV2   Task501_Tooth  0 --npz 
numactl -C 0-55 -m 0 python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 Task501_Tooth 0 --npz  --disable_tta
```

add ipex optimization
add to file /home/zhaoqion/winning/nnUNet/nnunet/training/network_training/nnUNetTrainerV2.py  line 123

```
            import intel_extension_for_pytorch as ipex
            self.network.train()
            self.network, self.optimizer = ipex.optimize(self.network, optimizer=self.optimizer, dtype=torch.bfloat16)
            self.print_to_log_file('==========================IPEX OPTIMIZATION BF16 on SPR =====================remember to switch to FP32 on ICX')
            self.bf16 = True


```