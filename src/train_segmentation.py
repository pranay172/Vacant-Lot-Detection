# src/train_segmentation.py
import argparse
import logging # For print_log
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log # For print_log
from mmengine.runner import Runner

# Import necessary components from MMSegmentation
from mmseg.registry import RUNNERS # If you use custom runner types
from mmseg.utils import register_all_modules # To register custom MMSeg components

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor using custom script')
    parser.add_argument('config', help='train config file path',
                        default='configs/mask2former_swinl_finetune_vacantlot_seg.py') # Default config path
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    
    # Modified resume logic:
    # --resume will attempt to auto-resume from the latest checkpoint in work_dir
    # --resume-from [PATH] will resume from a specific checkpoint path
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically.'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to a checkpoint file to resume from.'
    )
    parser.add_argument(
        '--load-from', # Kept for clarity if someone wants to load weights but not resume optimizer/epoch
        type=str,
        default=None,
        help='Path to a checkpoint file to load model weights from (does not resume training state).'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False, # Default to False, can be enabled by flag or in config
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher (handled by MMEngine, mostly for distributed training setup)')
    
    # For distributed training, MMEngine/PyTorch handles this environment variable
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0) # For PyTorch DDP
    
    args = parser.parse_args()

    # Ensure LOCAL_RANK is set for MMEngine's distributed setup if using 'pytorch' launcher
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args

def main():
    args = parse_args()

    # Register all modules in mmseg into the registries
    register_all_modules(init_default_scope=True) 

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher # Inform MMEngine about the launcher type
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Work_dir priority: CLI > config > default
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # Use config filename as default work_dir
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    
    # Create work_dir if not exists
    if not osp.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir, exist_ok=True) # Added this for robustness

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


    # Resume logic:
    # args.resume_from takes precedence for specific checkpoint resumption.
    # args.resume (action='store_true') enables auto-resuming from `latest.pth` in work_dir.
    # cfg.load_from is for loading pre-trained weights for fine-tuning (not resuming state).
    
    if args.resume_from:
        cfg.resume = True # Enable MMEngine's resume functionality
        cfg.load_from = args.resume_from # Specify the checkpoint to resume from
        print_log(f"Resuming from specific checkpoint: {args.resume_from}", logger='current', level=logging.INFO)
    elif args.resume:
        cfg.resume = True # Enable MMEngine's auto-resume (expects latest.pth in work_dir)
        # load_from should be None or handled by MMEngine's auto-resume logic
        # It will search for `latest.pth` in cfg.work_dir
        print_log(f"Auto-resuming from latest checkpoint in work_dir: {cfg.work_dir}", logger='current', level=logging.INFO)
    elif args.load_from: # Not resuming, but loading weights for fine-tuning
        cfg.load_from = args.load_from
        cfg.resume = False # Ensure resume is False if only loading weights
        print_log(f"Loading weights for fine-tuning from: {args.load_from}", logger='current', level=logging.INFO)
    else: # Neither resume nor load_from via CLI
        # Behavior depends on cfg.load_from and cfg.resume in the config file itself
        if cfg.get('resume'):
             print_log(f"Resuming based on 'resume=True' in config from work_dir: {cfg.work_dir}", logger='current', level=logging.INFO)
        elif cfg.get('load_from'):
             print_log(f"Loading weights based on 'load_from' in config: {cfg.load_from}", logger='current', level=logging.INFO)
        else:
             print_log("No checkpoint specified for resume or load_from. Training from scratch or backbone's init_cfg.", logger='current', level=logging.INFO)


    # Build the runner from config
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        # Build customized runner (e.g. IterBasedRunner)
        runner = RUNNERS.build(cfg)

    # Start training
    runner.train()


if __name__ == '__main__':
    # Setup basic logging to see MMEngine's print_log outputs
    logging.basicConfig(level=logging.INFO)
    
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
    
    main()

    # --- Restore original torch.load ---
    torch.load = _original_torch_load_seg
    # --- End of restore ---
