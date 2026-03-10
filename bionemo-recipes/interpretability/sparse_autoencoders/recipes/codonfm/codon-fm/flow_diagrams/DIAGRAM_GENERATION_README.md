# Encodon Model Diagram Generation

This guide explains how to generate various types of architectural diagrams for the Encodon model.

## Overview

The Encodon model is a transformer-based architecture designed for codon sequence modeling. This repository provides several tools to visualize the model architecture:

1. **Custom Architecture Diagrams** - High-level, publication-ready diagrams
2. **TorchView Diagrams** - Detailed computational flow visualization  
3. **TorchViz Diagrams** - Computational graph with gradient flow
4. **ONNX Export** - For use with Netron and other visualization tools

## Quick Start

### 1. Install Dependencies

```bash
# Run the installation script
chmod +x install_visualization_deps.sh
./install_visualization_deps.sh

# Or install manually
pip install torchview torchviz graphviz matplotlib onnx
```

### 2. Generate Diagrams

```bash
# Generate a custom architecture diagram for the 80M model
python generate_encodon_diagram.py --model_size 80m --diagram_types custom

# Generate all diagram types
python generate_encodon_diagram.py --model_size 80m --diagram_types all

# Generate diagrams for different model sizes
python generate_encodon_diagram.py --model_size 600m --diagram_types custom
python generate_encodon_diagram.py --model_size 1b --diagram_types custom
```

### 3. View Results

Diagrams will be saved in the `./diagrams/` directory by default. You can also use the Jupyter notebook for interactive exploration:

```bash
jupyter notebook encodon_diagram_example.ipynb
```

## Command Line Options

```bash
python generate_encodon_diagram.py [OPTIONS]

Options:
  --model_size {80m,600m,1b}     Model size to visualize (default: 80m)
  --output_dir DIR               Output directory (default: ./diagrams)
  --batch_size INT               Batch size for dummy input (default: 4)
  --seq_len INT                  Sequence length for dummy input (default: 128)
  --diagram_types {custom,torchview,torchviz,onnx,all}
                                Types of diagrams to generate (default: custom)
```

## Diagram Types

### 1. Custom Architecture Diagram (`--diagram_types custom`)

- **Purpose**: Publication-ready, high-level architecture overview
- **Output**: Clean matplotlib-based diagram showing:
  - Input/output layers
  - Transformer layers with attention and FFN components
  - Model statistics and parameter counts
- **Best for**: Papers, presentations, documentation

![Custom Architecture Example](./diagrams/encodon_80m_architecture.png)

### 2. TorchView Diagram (`--diagram_types torchview`)

- **Purpose**: Detailed computational flow visualization
- **Output**: Hierarchical view of all operations
- **Features**: Shows tensor shapes, operations, and data flow
- **Best for**: Debugging, understanding implementation details

### 3. TorchViz Diagram (`--diagram_types torchviz`)

- **Purpose**: Computational graph with gradient information
- **Output**: Graph showing forward and backward pass
- **Features**: Parameter connections, gradient flow
- **Best for**: Understanding training dynamics

### 4. ONNX Export (`--diagram_types onnx`)

- **Purpose**: Export for external visualization tools
- **Output**: `.onnx` file for use with Netron, ONNX Runtime, etc.
- **Features**: Standard format, interactive exploration
- **Best for**: Cross-platform visualization, detailed inspection

## Model Configurations

The script supports three pre-configured model sizes:

| Model | Hidden Size | Layers | Attention Heads | Intermediate Size | Parameters |
|-------|-------------|---------|-----------------|-------------------|------------|
| 80M   | 768         | 12      | 12              | 3,072             | ~80M       |
| 600M  | 1,536       | 24      | 24              | 6,144             | ~600M      |
| 1B    | 2,048       | 24      | 32              | 8,192             | ~1B        |

## Examples

### Generate diagrams for all model sizes:

```bash
for size in 80m 600m 1b; do
    python generate_encodon_diagram.py --model_size $size --diagram_types custom --output_dir ./diagrams
done
```

### Generate all diagram types for 80M model:

```bash
python generate_encodon_diagram.py \
    --model_size 80m \
    --diagram_types all \
    --output_dir ./diagrams \
    --batch_size 8 \
    --seq_len 256
```

### Use in Python script:

```python
from generate_encodon_diagram import create_model_config, create_custom_architecture_diagram
from src.models.encodon import EnCodon

# Create model
config = create_model_config("80m")
model = EnCodon(config)

# Generate custom diagram
create_custom_architecture_diagram(config, "my_diagram.png")
```

## Troubleshooting

### Common Issues

1. **Import Error: Cannot import encodon**
   ```bash
   # Make sure you're in the project root directory
   cd /path/to/codon_fm
   python generate_encodon_diagram.py
   ```

2. **GraphViz Not Found**
   ```bash
   # Install system graphviz package
   sudo apt-get install graphviz  # Ubuntu/Debian
   brew install graphviz          # macOS
   ```

3. **Memory Issues with Large Models**
   ```bash
   # Use smaller batch size and sequence length
   python generate_encodon_diagram.py --model_size 1b --batch_size 1 --seq_len 64
   ```

4. **TorchView/TorchViz Errors**
   ```bash
   # Update to latest versions
   pip install --upgrade torchview torchviz
   
   # Or use only custom diagrams
   python generate_encodon_diagram.py --diagram_types custom
   ```

### Alternative Visualization Tools

If the provided tools don't work, consider these alternatives:

1. **Netron** (web-based): https://netron.app
   - Upload the generated .onnx file
   - Interactive exploration

2. **TensorBoard**:
   ```python
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter()
   writer.add_graph(model, dummy_input)
   ```

3. **Weights & Biases**:
   ```python
   import wandb
   wandb.watch(model, log_graph=True)
   ```

## Customization

To customize the diagrams, you can modify the `generate_encodon_diagram.py` script:

- **Colors**: Change the `colors` dictionary in `create_custom_architecture_diagram()`
- **Layout**: Modify positioning and sizing parameters
- **Content**: Add or remove information from the diagram
- **Styling**: Adjust fonts, line styles, and other visual elements

## Output Files

The script generates the following files in the output directory:

```
diagrams/
├── encodon_80m_architecture.png      # Custom matplotlib diagram
├── encodon_80m_torchview.png         # TorchView diagram
├── encodon_80m_torchviz.png          # TorchViz diagram
└── encodon_80m_onnx.onnx             # ONNX export file
```

## Contributing

To add new diagram types or improve existing ones:

1. Create a new function following the pattern of existing generators
2. Add the new type to the `diagram_types` choices in the argument parser
3. Add the function call in the main loop
4. Update this README with usage instructions

## License

This visualization code is part of the Codon-FM project and follows the same license terms. 