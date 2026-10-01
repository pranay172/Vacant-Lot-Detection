# src/train_detection.py
import argparse
import logging  # For print_log
import os
# Note: MMDetection's training script is typically run directly.
# from mmengine.runner import Runner # MMEngine's way
# from mmdet.utils import setup_cache_size_limit_ # MMDetection's way
# However, it's more common to use the tools/train.py script provided by MMDetection.

# This script essentially prepares the command to run MMDetection's training.
# For programmatic training (less common for full MMDetection runs):
from mmengine.config import Config
from mmengine.logging import print_log # For print_log
from mmengine.runner import Runner

def main(args):
    cfg_path = args.config
    work_dir = args.work_dir
    
    # load config
    cfg = Config.fromfile(cfg_path)

    if work_dir is not None:
        cfg.work_dir = work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(cfg_path))[0])

    # AMP (Automatic Mixed Precision) training
    if args.amp:
        # Check if AMP is already configured in the optim_wrapper
        optim_wrapper_cfg = cfg.get('optim_wrapper', {})
        if optim_wrapper_cfg.get('type') == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current', # Use MMEngine's default logger
                level=logging.WARNING)
        else:
            # Ensure it's a standard OptimWrapper before overriding
            assert optim_wrapper_cfg.get('type', 'OptimWrapper') == 'OptimWrapper', ( # Default to OptimWrapper if not set
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper_cfg.get("type")}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            # loss_scale can be 'dynamic' or a float
            if 'loss_scale' not in cfg.optim_wrapper: # only set if not already defined
                 cfg.optim_wrapper.loss_scale = 'dynamic'
            print_log(
                'AMP training enabled via command line flag.',
                logger='current',
                level=logging.INFO)

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DINO detector with MMDetection.")
    parser.add_argument(
        "config", 
        help="Path to the MMDetection config file",
        default="configs/dino_swinl_finetune_vacantlot.py"
    )
    parser.add_argument(
        "--work-dir", 
        help="The directory to save logs and models."
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False, # Default to False, can be enabled by flag or in config
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--resume',
        type=str,
        nargs='?',
        default=None, # Do not resume by default
        const='auto', # If --resume is given without a path, resume from latest in work_dir
        help='Resume training from a specific checkpoint file or automatically from the latest checkpoint in the work_dir.'
    )
    # Add other arguments MMDetection's train.py might take, like --launcher, --seed, etc.
    # For simplicity, we'll keep it minimal here.
    
    args = parser.parse_args()
    
    # --- PyTorch load patch ---
    import torch
    import pickle

    _original_torch_load_seg = torch.load
    def _patched_torch_load_seg(*args_patch, **kwargs_patch):
        # PyTorch 2.6+ makes weights_only=True the default.
        # For older versions that warn or default to False, this ensures False.
        if 'weights_only' not in kwargs_patch or kwargs_patch.get('weights_only') is not False:
            # print("Patching torch.load to use weights_only=False for this call.")
            kwargs_patch['weights_only'] = False
        try:
            return _original_torch_load_seg(*args_patch, **kwargs_patch)
        except pickle.UnpicklingError as e_patch:
            # Specifically catch errors related to unpickling non-weight objects
            # For a more general approach if other MMEngine objects cause issues:
            print(f"Pickle error encountered ('{str(e_patch)}'). Forcing weights_only=False.")
            kwargs_patch['weights_only'] = False
            return _original_torch_load_seg(*args_patch, **kwargs_patch)
        # It's better to let other UnpicklingErrors raise if they are not related to weights_only
        except Exception as e_patch_other: # Catch any other exception during load
           print(f"Another error during torch.load: {e_patch_other}")
           raise e_patch_other

    torch.load = _patched_torch_load_seg
    # --- End of PyTorch load patch ---
    
    main(args)

    # --- Restore original torch.load ---
    torch.load = _original_torch_load_seg
    # --- End of restore ---
