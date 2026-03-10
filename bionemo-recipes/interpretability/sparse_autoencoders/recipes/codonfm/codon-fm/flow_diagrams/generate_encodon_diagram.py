#!/usr/bin/env python3
"""
Script to generate various diagrams for the Encodon model architecture.

This script provides multiple visualization approaches:
1. Model architecture diagram using torchview
2. Computational graph using torchviz
3. Custom architectural diagram using matplotlib
# 4. Export to ONNX for Netron visualization
"""

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
from typing import Dict, Any, Optional

try:
    from torchview import draw_graph
    TORCHVIEW_AVAILABLE = True
except ImportError:
    TORCHVIEW_AVAILABLE = False
    print("Warning: torchview not available. Install with: pip install torchview")

try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    print("Warning: torchviz not available. Install with: pip install torchviz")

# Import the model
from src.models.encodon import EnCodon, EnCodonConfig


def create_model_config(model_size: str = "80m") -> EnCodonConfig:
    """Create model configuration based on size."""
    import yaml
    from pathlib import Path
    
    # Load model configs from config files
    config_dir = Path("configs/model")
    configs = {}
    
    for config_file in config_dir.glob("encodon_*.yaml"):
        model_size = config_file.stem.split("_")[1] # Get size from filename
        with open(config_file) as f:
            config = yaml.safe_load(f)
            # Skip optimizer, scheduler and __target__ keys
            configs[model_size] = {
                k: v for k, v in config.items() 
                if not k.startswith('_') and k not in ['optimizer', 'scheduler']
            }
    
    config_params = configs.get(model_size, configs["80m"])
    return EnCodonConfig(**config_params)


def generate_torchview_diagram(model: nn.Module, input_shape: tuple, vocab_size: int, output_path: str):
    """Generate model architecture diagram using torchview."""
    if not TORCHVIEW_AVAILABLE:
        print("Skipping torchview diagram - torchview not available")
        return
    
    try:
        # Create a copy of the model on CPU to avoid xformers issues
        model_cpu = type(model)(model.config)
        model_cpu.load_state_dict(model.state_dict())
        model_cpu.eval()
        
        # Disable flash attention to avoid xformers compatibility issues
        for module in model_cpu.modules():
            if hasattr(module, 'use_flash_attn'):
                module.use_flash_attn = False
        
        # Create dummy input on CPU
        batch_size, seq_len = input_shape
        dummy_input = {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long)
        }
        
        # Generate the diagram with CPU model
        model_graph = draw_graph(
            model_cpu, 
            input_data=dummy_input,
            expand_nested=True,
            depth=6,  # Reduced depth to avoid too much detail
            device='cpu'  # Use CPU instead of meta
        )
        
        # Save the diagram
        model_graph.visual_graph.render(output_path.replace('.png', '_torchview'), format='png', cleanup=True)
        print(f"Torchview diagram saved to {output_path.replace('.png', '_torchview.png')}")
        
    except Exception as e:
        print(f"Error generating torchview diagram: {e}")
        print("This is likely due to xformers/flash attention compatibility issues.")
        print("Try using --diagram_types custom instead.")


def generate_torchviz_diagram(model: nn.Module, input_shape: tuple, vocab_size: int, output_path: str):
    """Generate computational graph using torchviz."""
    if not TORCHVIZ_AVAILABLE:
        print("Skipping torchviz diagram - torchviz not available")
        return
    
    try:
        # Create a copy of the model on CPU to avoid xformers issues
        model_cpu = type(model)(model.config)
        model_cpu.load_state_dict(model.state_dict())
        model_cpu.eval()
        
        # Disable flash attention to avoid xformers compatibility issues
        for module in model_cpu.modules():
            if hasattr(module, 'use_flash_attn'):
                module.use_flash_attn = False
        
        batch_size, seq_len = input_shape
        
        # Create dummy input on CPU
        dummy_input = {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long)
        }
        
        # Forward pass
        output = model_cpu(**dummy_input)
        
        # Generate computational graph
        dot = make_dot(output.logits, params=dict(model_cpu.named_parameters()), show_attrs=True, show_saved=True)
        dot.render(output_path.replace('.png', '_torchviz'), format='png', cleanup=True)
        print(f"Torchviz diagram saved to {output_path.replace('.png', '_torchviz.png')}")
        
    except Exception as e:
        print(f"Error generating torchviz diagram: {e}")
        print("This is likely due to xformers/flash attention compatibility issues.")
        print("Try using --diagram_types custom instead.")


def create_custom_architecture_diagram(config: EnCodonConfig, vocab_size: int, output_path: str):
    """Create a custom architectural diagram using matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Colors
    colors = {
        'embedding': '#FFE6CC',
        'attention': '#E6F3FF', 
        'ffn': '#E6FFE6',
        'norm': '#FFE6F3',
        'output': '#F0F0F0'
    }
    
    # Title
    ax.text(5, 19.5, f'Encodon Model Architecture ({config.hidden_size}d)', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Input
    input_box = FancyBboxPatch((1, 17.5), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 18, f'Input Tokens\n(Batch Size × Sequence Length)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Codon Embedding
    emb_box = FancyBboxPatch((1, 16), 8, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['embedding'], 
                             edgecolor='black', linewidth=1)
    ax.add_patch(emb_box)
    ax.text(5, 16.5, f'Codon Embedding\n(Vocab: {config.vocab_size}, Hidden: {config.hidden_size})', 
            ha='center', va='center', fontsize=10)
    
    # Transformer Layers
    y_start = 14.5
    layer_height = 0.8
    spacing = 0.2
    
    for layer_idx in range(config.num_hidden_layers):
        y_pos = y_start - layer_idx * (layer_height + spacing)
        
        # Layer box
        layer_box = FancyBboxPatch((0.5, y_pos - layer_height/2), 9, layer_height, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='white', 
                                   edgecolor='black', linewidth=1)
        ax.add_patch(layer_box)
        
        # Multi-Head Attention
        attn_box = FancyBboxPatch((1, y_pos - 0.3), 3.5, 0.6, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['attention'], 
                                  edgecolor='blue', linewidth=1)
        ax.add_patch(attn_box)
        ax.text(2.75, y_pos, f'Multi-Head\nAttention\n({config.num_attention_heads} heads)', 
                ha='center', va='center', fontsize=8)
        
        # Feed Forward
        ffn_box = FancyBboxPatch((5.5, y_pos - 0.3), 3.5, 0.6, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=colors['ffn'], 
                                 edgecolor='green', linewidth=1)
        ax.add_patch(ffn_box)
        ax.text(7.25, y_pos, f'Feed Forward\n({config.intermediate_size}d)', 
                ha='center', va='center', fontsize=8)
        
        # Layer number
        ax.text(0.2, y_pos, f'L{layer_idx+1}', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Output Head
    output_y = y_start - config.num_hidden_layers * (layer_height + spacing) - 1
    output_box = FancyBboxPatch((1, output_y), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, output_y + 0.5, f'Language Model Head\n(Hidden: {config.hidden_size} → Vocab: {config.vocab_size})', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Model Statistics
    stats_y = output_y - 1.5
    stats_text = f"""Model Statistics:
• Vocabulary Size: {config.vocab_size:,}
• Hidden Size: {config.hidden_size:,}
• Number of Layers: {config.num_hidden_layers}
• Attention Heads: {config.num_attention_heads}
• Intermediate Size: {config.intermediate_size:,}
• Max Position Embeddings: {config.max_position_embeddings:,}
• Parameters: ~{estimate_parameters(config):,}"""
    
    ax.text(5, stats_y, stats_text, ha='center', va='top', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    # Arrows
    for i in range(config.num_hidden_layers + 2):
        if i == 0:
            # Input to embedding
            ax.arrow(5, 17.4, 0, -0.3, head_width=0.1, head_length=0.05, fc='black', ec='black')
        elif i <= config.num_hidden_layers:
            # Between layers
            y_from = 15.9 - (i-1) * (layer_height + spacing)
            y_to = y_from - spacing - 0.1
            ax.arrow(5, y_from, 0, y_to - y_from, head_width=0.1, head_length=0.05, fc='black', ec='black')
        else:
            # Last layer to output
            y_from = 15.9 - (config.num_hidden_layers-1) * (layer_height + spacing) - layer_height/2
            ax.arrow(5, y_from, 0, -0.4, head_width=0.1, head_length=0.05, fc='black', ec='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Custom architecture diagram saved to {output_path}")


def estimate_parameters(config: EnCodonConfig) -> int:
    """Estimate the number of parameters in the model."""
    # Embedding parameters
    embedding_params = config.vocab_size * config.hidden_size
    
    # Transformer layer parameters
    # Attention: Q, K, V projections + output projection
    attention_params = 4 * (config.hidden_size * config.hidden_size)
    
    # Feed-forward: two linear layers
    ffn_params = 2 * (config.hidden_size * config.intermediate_size)
    
    # Layer norms (4 per layer: pre-attn, post-attn, pre-ffn, post-ffn)
    layernorm_params = 4 * config.hidden_size
    
    # Total per layer
    per_layer_params = attention_params + ffn_params + layernorm_params
    
    # Total transformer parameters
    transformer_params = config.num_hidden_layers * per_layer_params
    
    # Output head parameters
    output_params = config.hidden_size * config.vocab_size
    
    # Total
    total_params = embedding_params + transformer_params + output_params
    
    return total_params


def export_to_onnx(model: nn.Module, input_shape: tuple, vocab_size: int, output_path: str):
    """Export model to ONNX format for Netron visualization."""
    try:
        model.eval()
        batch_size, seq_len = input_shape
        
        # Get the device of the model
        device = next(model.parameters()).device
        
        # Create dummy input on the same device as the model
        dummy_input = (
            torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
            torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        )
        
        # Export to ONNX
        onnx_path = output_path.replace('.png', '.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'}
            }
        )
        print(f"ONNX model exported to {onnx_path}")
        print(f"You can visualize this with Netron: https://netron.app")
        
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate Encodon model diagrams")
    parser.add_argument("--model_size", choices=["80m", "600m", "1b"], default="80m",
                        help="Model size to visualize")
    parser.add_argument("--output_dir", default="./diagrams", 
                        help="Output directory for diagrams")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for dummy input")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Sequence length for dummy input")
    parser.add_argument("--diagram_types", nargs="+", 
                        choices=["custom", "torchview", "torchviz", "all"],
                        default=["custom"],
                        help="Types of diagrams to generate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model configuration
    config = create_model_config(args.model_size)
    print(f"Creating diagrams for Encodon {args.model_size} model...")
    print(f"Model config: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
    print('Model vocab size: ', config.vocab_size)
    
    # Create model
    model = EnCodon(config)
    
    # Only move to CUDA if available and if we're not just doing custom diagrams
    use_cuda = torch.cuda.is_available() and not (len(args.diagram_types) == 1 and args.diagram_types[0] == "custom")
    
    if use_cuda:
        model = model.cuda()
        print("Using CUDA for model operations")
    else:
        print("Using CPU for model operations")
        # Disable flash attention for CPU operations
        for module in model.modules():
            if hasattr(module, 'use_flash_attn'):
                module.use_flash_attn = False
    
    model.eval()
    
    input_shape = (args.batch_size, args.seq_len)
    
    # Generate requested diagrams
    diagram_types = args.diagram_types
    if "all" in diagram_types:
        diagram_types = ["custom", "torchview", "torchviz", ]
    
    base_output_path = os.path.join(args.output_dir, f"encodon_{args.model_size}")
    
    for diagram_type in diagram_types:
        print(f"\nGenerating {diagram_type} diagram...")
        try:
            if diagram_type == "custom":
                create_custom_architecture_diagram(config, vocab_size=config.vocab_size, output_path=f"{base_output_path}_architecture.png")
            elif diagram_type == "torchview":
                generate_torchview_diagram(model, input_shape, vocab_size=config.vocab_size, output_path=f"{base_output_path}_torchview.png")
            elif diagram_type == "torchviz":
                generate_torchviz_diagram(model, input_shape, vocab_size=config.vocab_size, output_path=f"{base_output_path}_torchviz.png")
            # elif diagram_type == "onnx":
            #     export_to_onnx(model, input_shape, vocab_size=config.vocab_size, output_path=f"{base_output_path}_onnx.png")
        except Exception as e:
            print(f"Error generating {diagram_type} diagram: {e}")
            print("Continuing with other diagram types...")
    
    print("\nDiagram generation complete!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main() 