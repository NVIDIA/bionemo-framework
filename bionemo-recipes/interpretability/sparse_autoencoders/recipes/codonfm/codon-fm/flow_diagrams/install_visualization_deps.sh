#!/bin/bash

echo "Installing visualization dependencies for Encodon model diagrams..."

# Install Python packages for model visualization
pip install torchview
pip install torchviz
pip install graphviz
pip install matplotlib
pip install onnx

# Install system graphviz (for torchviz)
if command -v apt-get >/dev/null 2>&1; then
    echo "Installing graphviz system package (Ubuntu/Debian)..."
    sudo apt-get update
    sudo apt-get install -y graphviz
elif command -v yum >/dev/null 2>&1; then
    echo "Installing graphviz system package (CentOS/RHEL)..."
    sudo yum install -y graphviz
elif command -v brew >/dev/null 2>&1; then
    echo "Installing graphviz system package (macOS)..."
    brew install graphviz
else
    echo "Warning: Could not automatically install graphviz system package."
    echo "Please install graphviz manually for your system."
fi

echo "Installation complete!"
echo ""
echo "You can now generate diagrams using:"
echo "python generate_encodon_diagram.py --model_size 80m --diagram_types custom"
echo "python generate_encodon_diagram.py --model_size 80m --diagram_types all" 