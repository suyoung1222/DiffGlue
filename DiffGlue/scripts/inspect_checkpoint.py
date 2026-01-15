#!/usr/bin/env python3
"""
Simple script to inspect checkpoint files and view model architecture information.

Usage:
    python inspect_checkpoint.py --checkpoint <path_to_checkpoint.tar>
    python inspect_checkpoint.py -c <path_to_checkpoint.tar>
"""

import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
from collections import OrderedDict


def count_parameters(state_dict):
    """Count total parameters and their sizes."""
    total_params = 0
    trainable_params = 0
    param_info = []
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            num_params = param.numel()
            total_params += num_params
            trainable_params += num_params  # All saved params are trainable
            param_info.append({
                'name': name,
                'shape': tuple(param.shape),
                'numel': num_params,
                'dtype': str(param.dtype)
            })
    
    return total_params, trainable_params, param_info


def format_size(size_bytes):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def inspect_optimizer(optimizer_state):
    """Inspect optimizer state."""
    if optimizer_state is None:
        return "No optimizer state found"
    
    info = []
    if 'param_groups' in optimizer_state:
        info.append(f"Parameter groups: {len(optimizer_state['param_groups'])}")
        for i, group in enumerate(optimizer_state['param_groups']):
            info.append(f"  Group {i}: {len(group.get('params', []))} parameters")
            if 'lr' in group:
                info.append(f"    Learning rate: {group['lr']}")
    else:
        info.append("Parameter groups: N/A")
    
    if 'state' in optimizer_state:
        info.append(f"Optimizer states: {len(optimizer_state['state'])} parameter states")
    
    return "\n".join(info)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect checkpoint files and view model architecture information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_checkpoint.py -c checkpoint_best.tar
  python inspect_checkpoint.py --checkpoint outputs/training/exp_name/checkpoint_10_50000.tar
        """
    )
    parser.add_argument(
        '-c', '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (.tar)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed parameter information'
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return 1
    
    print("=" * 80)
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 1
    
    # Basic checkpoint information
    print("\n[CHECKPOINT INFORMATION]")
    print("-" * 80)
    print(f"Checkpoint file: {checkpoint_path.name}")
    print(f"Checkpoint directory: {checkpoint_path.parent}")
    
    # Training metadata
    if 'epoch' in checkpoint:
        print(f"\nEpoch: {checkpoint['epoch']}")
    if 'losses' in checkpoint and checkpoint['losses']:
        print(f"\nLosses (last batch):")
        for key, value in checkpoint['losses'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.6f}")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}" if value.numel() == 1 else f"  {key}: tensor{value.shape}")
    
    if 'eval' in checkpoint and checkpoint['eval']:
        print(f"\nEvaluation metrics:")
        for key, value in checkpoint['eval'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.6f}")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}" if value.numel() == 1 else f"  {key}: tensor{value.shape}")
    
    # Configuration
    if 'conf' in checkpoint:
        print("\n[CONFIGURATION]")
        print("-" * 80)
        conf = OmegaConf.create(checkpoint['conf'])
        if hasattr(conf, 'model'):
            print("\nModel configuration:")
            print(OmegaConf.to_yaml(conf.model))
        if hasattr(conf, 'train'):
            print("\nTraining configuration:")
            train_conf_str = OmegaConf.to_yaml(conf.train)
            # Limit output length for training config
            lines = train_conf_str.split('\n')
            if len(lines) > 30:
                print('\n'.join(lines[:30]))
                print(f"... ({len(lines) - 30} more lines)")
            else:
                print(train_conf_str)
    
    # Model state dict
    if 'model' in checkpoint:
        print("\n[MODEL ARCHITECTURE]")
        print("-" * 80)
        model_state = checkpoint['model']
        total_params, trainable_params, param_info = count_parameters(model_state)
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Estimate model size (assuming float32 = 4 bytes per parameter)
        model_size_bytes = total_params * 4
        print(f"Estimated model size: {format_size(model_size_bytes)}")
        
        # Count parameters by module prefix
        module_counts = OrderedDict()
        for name in model_state.keys():
            prefix = name.split('.')[0] if '.' in name else name
            if prefix not in module_counts:
                module_counts[prefix] = 0
            if isinstance(model_state[name], torch.Tensor):
                module_counts[prefix] += model_state[name].numel()
        
        print(f"\nParameters by module:")
        for module, count in module_counts.items():
            percentage = (count / total_params) * 100
            print(f"  {module}: {count:,} ({percentage:.2f}%)")
        
        print(f"\nTotal parameter keys: {len(model_state)}")
        
        if args.detailed:
            print(f"\n[DETAILED PARAMETER INFORMATION]")
            print("-" * 80)
            # Group by prefix for better readability
            grouped_params = OrderedDict()
            for info in param_info:
                prefix = info['name'].split('.')[0] if '.' in info['name'] else info['name']
                if prefix not in grouped_params:
                    grouped_params[prefix] = []
                grouped_params[prefix].append(info)
            
            for prefix, params in grouped_params.items():
                print(f"\n{prefix}:")
                for param in params[:20]:  # Limit to first 20 per module
                    print(f"  {param['name']}")
                    print(f"    Shape: {param['shape']}, Params: {param['numel']:,}, Dtype: {param['dtype']}")
                if len(params) > 20:
                    print(f"  ... ({len(params) - 20} more parameters)")
        else:
            print("\n[PARAMETER KEYS (first 50)]")
            print("-" * 80)
            for i, name in enumerate(list(model_state.keys())[:50]):
                if isinstance(model_state[name], torch.Tensor):
                    shape = tuple(model_state[name].shape)
                    numel = model_state[name].numel()
                    print(f"  {name}: shape={shape}, params={numel:,}")
            if len(model_state) > 50:
                print(f"  ... ({len(model_state) - 50} more keys)")
                print("\n  Use --detailed flag for full parameter information")
    
    # Optimizer state
    if 'optimizer' in checkpoint:
        print("\n[OPTIMIZER STATE]")
        print("-" * 80)
        optimizer_info = inspect_optimizer(checkpoint['optimizer'])
        print(optimizer_info)
    
    # Learning rate scheduler
    if 'lr_scheduler' in checkpoint:
        print("\n[LEARNING RATE SCHEDULER]")
        print("-" * 80)
        lr_scheduler = checkpoint['lr_scheduler']
        if isinstance(lr_scheduler, dict):
            print(f"Scheduler state keys: {list(lr_scheduler.keys())}")
            if 'last_epoch' in lr_scheduler:
                print(f"Last epoch: {lr_scheduler['last_epoch']}")
    
    print("\n" + "=" * 80)
    print("Inspection complete!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())
