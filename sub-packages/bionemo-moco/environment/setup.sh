#!/bin/bash

# Set the path to your Conda environment YAML file
ENV_YAML="environment/moco_env.yaml"

# Extract the environment name from the YAML file
ENV_NAME=$(head -n 1 "$ENV_YAML" | cut -d':' -f2- | tr -d ' ')

# Load Conda to enable command
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create the Conda environment from the YAML file
echo "Creating Conda environment $ENV_NAME from $ENV_YAML..."
conda env create -f "$ENV_YAML"

# Activate the Conda environment
echo "Activating Conda environment $ENV_NAME..."
conda activate "$ENV_NAME"

# Check if the environment was successfully activated
if [ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]; then
    echo "Conda environment $ENV_NAME activated successfully."
    # Navigate to your project directory if needed
    # cd /path/to/your/project  # Uncomment and adjust this path as necessary
    # Install your project in editable mode using pip
    pip install pydoc-markdown>=4.8.2
    pip install pytest-cov==4.1.0 pytest-timeout==2.2.0 pytest-dependency==0.5.1
    pre-commit install
    echo "Installing bionemo-moco in editable mode using pip..."
    pip install -e .
    echo "Setup complete."
    # Run tests
    echo "Running tests..."
    pytest
    echo "Tests complete. You can now work within the $ENV_NAME environment."
else
    echo "Failed to activate Conda environment $ENV_NAME. Exiting..."
    exit 1
fi
