import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def clean_model_name(model_name):
    """Clean model name by extracting steps and samples values for display."""
    # Handle the new naming convention: evo2_7b_1m_50_ncbi_virus_train_set_caliciviridae_samples=400
    if "_ncbi_" in model_name and "samples=" in model_name:
        # Extract steps value (number after 1m_ and before _ncbi)
        parts = model_name.split("_")
        steps = None
        samples = None
        
        for i, part in enumerate(parts):
            if part == "1m" and i + 1 < len(parts):
                steps = parts[i + 1]
            elif part.startswith("samples="):
                samples = part.split("=")[1]
        
        if steps and samples:
            return f"steps={steps}_samples={samples}"
    
    # Handle epoch suffix (legacy)
    if "_epoch" in model_name:
        return model_name.split("_epoch")[0]
    
    return model_name

def extract_dataset_identifier(model_name):
    """Extract dataset identifier from model name for filename generation."""
    # Handle the new naming convention: evo2_7b_1m_50_ncbi_virus_train_set_caliciviridae_samples=400
    if "_ncbi_" in model_name and "samples=" in model_name:
        # Find the part between the steps and samples
        # Pattern: evo2_7b_1m_STEPS_DATASET_IDENTIFIER_samples=SAMPLES
        parts = model_name.split("_")
        
        # Find the start (after steps) and end (before samples=)
        start_idx = None
        end_idx = None
        
        for i, part in enumerate(parts):
            if i > 0 and parts[i-1] == "1m" and part.isdigit():
                start_idx = i + 1  # Start after the steps number
            elif part.startswith("samples="):
                end_idx = i  # End before samples=
                break
        
        if start_idx and end_idx:
            dataset_parts = parts[start_idx:end_idx]
            # Convert to kebab-case for filename
            return "-".join(dataset_parts)
    
    return None

def extract_experiment_name(base_path):
    """Extract experiment name from base_path."""
    # Example: "results/virus_reproduction/full" -> "virus_reproduction"
    # Example: "results/virus_ecoli/full" -> "virus_ecoli"
    path_parts = base_path.strip('/').split('/')
    
    # Find the part after "results"
    if len(path_parts) >= 2 and path_parts[0] == "results":
        return path_parts[1]
    elif len(path_parts) >= 1:
        # If no "results" prefix, take the first part
        return path_parts[0]
    
    return "unknown_experiment"

def extract_sample_type(base_path):
    """Extract sample type from base_path."""
    # Example: "results/virus_reproduction/full" -> "full"
    # Example: "results/virus_reproduction/sample=1000_seed=42" -> "sample=1000_seed=42"
    path_parts = base_path.strip('/').split('/')
    
    # Take the last part as sample type
    if len(path_parts) >= 1:
        sample_type = path_parts[-1]
        # Convert underscores to hyphens for filename compatibility
        return sample_type.replace('_', '-')
    
    return "unknown-sample"

def generate_plot_filename(plot_type, dataset_name, experiment_name, sample_type, is_all_models=False):
    """Generate standardized plot filename."""
    # Base format: fitness_spearman_{dataset_name}_{experiment_name}_{sample_type}
    
    if is_all_models or not dataset_name:
        # For all models or when no specific dataset is identified
        if plot_type == "taxon":
            return f"fitness_spearman_all-models_{experiment_name}_{sample_type}_by-taxon.png"
        else:  # model
            return f"fitness_spearman_all-models_{experiment_name}_{sample_type}_by-model.png"
    else:
        # For specific dataset
        if plot_type == "taxon":
            return f"fitness_spearman_{dataset_name}_{experiment_name}_{sample_type}_by-taxon.png"
        else:  # model
            return f"fitness_spearman_{dataset_name}_{experiment_name}_{sample_type}_by-model.png"

def extract_steps_from_model_name(model_name):
    """Extract steps value from model name, return None if no steps found."""
    if "_ncbi_" in model_name and "samples=" in model_name:
        parts = model_name.split("_")
        for i, part in enumerate(parts):
            if part == "1m" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
    return None

def sort_models_by_steps(models):
    """Sort models with original model first, then by ascending steps."""
    def sort_key(model_name):
        steps = extract_steps_from_model_name(model_name)
        if steps is None:
            # Original models (no steps) get priority (sort value 0)
            return (0, model_name)
        else:
            # Models with steps get sorted by steps value (sort value 1, then steps)
            return (1, steps)
    
    return sorted(models, key=sort_key)

def collect_fitness_data(model_names=None, base_path="results/virus_reproduction/full"):
    """Collect Spearman correlation values from all fitness CSV files, filtered by model names."""
    taxons = ["Human", "Virus", "Eukaryote","Prokaryote"]
    
    data = {}
    
    for taxon in taxons:
        data[taxon] = []
        taxon_path = os.path.join(base_path, taxon)
        
        if not os.path.exists(taxon_path):
            continue
            
        # Get all available model directories
        available_models = []
        for item in os.listdir(taxon_path):
            item_path = os.path.join(taxon_path, item)
            if os.path.isdir(item_path):
                available_models.append(item)
        
        # If model_names is specified, use substring matching
        if model_names:
            # Ensure model_names is a list
            if isinstance(model_names, str):
                model_names = [model_names]
            
            # Handle "all" models case
            if len(model_names) == 1 and model_names[0] == "all":
                selected_models = available_models[:]
            else:
                # Find models by substring matching
                selected_models = set()
                
                # Always include the base model if it exists
                base_model = "nemo2_evo2_7b_1m"
                if base_model in available_models:
                    selected_models.add(base_model)
                
                # Add models that match any of the provided substrings
                for substring in model_names:
                    matches = [model for model in available_models if substring in model]
                    selected_models.update(matches)
                
                selected_models = list(selected_models)
                
            for model_name in selected_models:
                model_path = os.path.join(taxon_path, model_name)
                if os.path.exists(model_path):
                    fitness_files = glob.glob(os.path.join(model_path, "*_fitness.csv"))
                    for file_path in fitness_files:
                        try:
                            # Read the CSV file
                            df = pd.read_csv(file_path)
                            
                            # Extract the Spearman correlation value (first column, first data row)
                            spearman_value = df.iloc[0, 0]  # First row of data, first column
                            
                            if pd.notna(spearman_value):
                                data[taxon].append(abs(float(spearman_value)))
                                
                            print(f"Loaded {os.path.basename(file_path)}: |Spearman| = {abs(spearman_value):.4f}")
                            
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
        else:
            # Look in all model subdirectories
            fitness_files = []
            for model_dir in os.listdir(taxon_path):
                model_path = os.path.join(taxon_path, model_dir)
                if os.path.isdir(model_path):
                    fitness_files.extend(glob.glob(os.path.join(model_path, "*_fitness.csv")))
            
            for file_path in fitness_files:
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Extract the Spearman correlation value (first column, first data row)
                    spearman_value = df.iloc[0, 0]  # First row of data, first column
                    
                    if pd.notna(spearman_value):
                        data[taxon].append(abs(float(spearman_value)))
                        
                    print(f"Loaded {os.path.basename(file_path)}: |Spearman| = {abs(spearman_value):.4f}")
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return data

def collect_fitness_data_by_model(model_names, base_path="results/virus_reproduction/full"):
    """Collect Spearman correlation values by model for Virus taxon only."""
    taxon = "Virus"
    
    data = {}
    
    taxon_path = os.path.join(base_path, taxon)
    
    if not os.path.exists(taxon_path):
        print(f"Warning: {taxon_path} does not exist")
        return data, []
    
    # Get all available model directories
    available_models = []
    for item in os.listdir(taxon_path):
        item_path = os.path.join(taxon_path, item)
        if os.path.isdir(item_path):
            available_models.append(item)
    
    print(f"Available models: {', '.join(available_models)}")
    
    # Handle "all" models case
    if isinstance(model_names, list) and len(model_names) == 1 and model_names[0] == "all":
        selected_models = available_models[:]
        print(f"Using all {len(selected_models)} models")
    else:
        # Ensure model_names is a list
        if isinstance(model_names, str):
            model_names = [model_names]
        
        # Find models by substring matching
        selected_models = set()
        
        # Always include the base model
        base_model = "nemo2_evo2_7b_1m"
        if base_model in available_models:
            selected_models.add(base_model)
            print(f"Added base model: {base_model}")
        else:
            print(f"Warning: Base model {base_model} not found in available models")
        
        # Add models that match any of the provided substrings
        for substring in model_names:
            matches = [model for model in available_models if substring in model]
            if matches:
                selected_models.update(matches)
                print(f"Substring '{substring}' matched: {', '.join(matches)}")
            else:
                print(f"Warning: No models found containing substring '{substring}'")
        
        selected_models = list(selected_models)
        print(f"Final selected models: {', '.join(selected_models)}")
    
    for model_name in selected_models:
        data[model_name] = []
        model_path = os.path.join(taxon_path, model_name)
        
        if os.path.exists(model_path):
            fitness_files = glob.glob(os.path.join(model_path, "*_fitness.csv"))
            
            for file_path in fitness_files:
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Extract the Spearman correlation value (first column, first data row)
                    spearman_value = df.iloc[0, 0]  # First row of data, first column
                    
                    if pd.notna(spearman_value):
                        data[model_name].append(abs(float(spearman_value)))
                        
                    print(f"Loaded {model_name}/{os.path.basename(file_path)}: |Spearman| = {abs(spearman_value):.4f}")
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        else:
            print(f"Warning: Model path {model_path} does not exist")
    
    return data, selected_models

def create_fitness_plot_by_taxon(data, model_names=None):
    """Create a bar plot showing Spearman correlations by taxon."""
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting
    taxons = list(data.keys())
    x_positions = np.arange(len(taxons))
    
    # Calculate statistics for each taxon
    means = []
    all_values = []
    
    for taxon in taxons:
        values = data[taxon]
        if values:
            means.append(np.mean(values))
            all_values.extend(values)
        else:
            means.append(0)
    
    # Create bar plot with mean values
    bars = ax.bar(x_positions, means, alpha=0.7, color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
    
    # Add individual data points as scatter plot
    for i, taxon in enumerate(taxons):
        values = data[taxon]
        if values:
            # Add some jitter to x-coordinates for better visibility
            x_jitter = np.random.normal(i, 0.05, len(values))
            ax.scatter(x_jitter, values, color='black', alpha=0.6, s=30)
    
    # Customize the plot
    ax.set_xlabel('Taxon', fontsize=12)
    ax.set_ylabel('|Spearman ρ|', fontsize=12)
    
    # Dynamic title based on model names
    if model_names:
        if isinstance(model_names, str):
            model_names = [model_names]
        # Clean model names for display
        clean_model_names = [clean_model_name(name) for name in model_names]
        models_str = ", ".join(clean_model_names)
        title = f'DMS Fitness Prediction Performance by Taxon ({models_str})'
    else:
        title = 'DMS Fitness Prediction Performance by Taxon (All Models)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(taxons)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits
    if all_values:
        y_min = min(all_values) - 0.05
        y_max = max(all_values) + 0.05
        ax.set_ylim(y_min, y_max)
    
    # Add statistics text
    for i, (taxon, values) in enumerate(data.items()):
        if values:
            mean_val = np.mean(values)
            n_samples = len(values)
            ax.text(i, mean_val + 0.01, f'n={n_samples}\nmean={mean_val:.3f}', 
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_fitness_plot_by_model(data, model_names):
    """Create a bar plot showing Spearman correlations by model for Virus taxon."""
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting
    models = sort_models_by_steps(list(data.keys()))
    x_positions = np.arange(len(models))
    
    # Calculate statistics for each model
    means = []
    all_values = []
    
    for model in models:
        values = data[model]
        if values:
            means.append(np.mean(values))
            all_values.extend(values)
        else:
            means.append(0)
    
    # Create bar plot with mean values
    bars = ax.bar(x_positions, means, alpha=0.7, color=['lightsteelblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightsalmon'])
    
    # Add individual data points as scatter plot
    for i, model in enumerate(models):
        values = data[model]
        if values:
            # Add some jitter to x-coordinates for better visibility
            x_jitter = np.random.normal(i, 0.05, len(values))
            ax.scatter(x_jitter, values, color='black', alpha=0.6, s=30)
    
    # Customize the plot
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('|Spearman ρ|', fontsize=12)
    
    # Generate title with dataset information
    title = 'DMS Fitness Prediction Performance by Model (Evaluated on Virus)'
    
    # Check if we have models from the same dataset or mixed datasets
    dataset_identifiers = set()
    for model in models:
        dataset_id = extract_dataset_identifier(model)
        if dataset_id:
            dataset_identifiers.add(dataset_id)
    
    # Add training dataset information
    if len(dataset_identifiers) == 1:
        # All models from same training dataset
        dataset_display = list(dataset_identifiers)[0].replace('_', ' ').title()
        title = f'{title}\nModels Trained on: {dataset_display}'
    elif len(dataset_identifiers) > 1:
        # Mixed training datasets
        title = f'{title}\nModels from Multiple Training Datasets'
    else:
        # No dataset identifiers found (original models)
        title = f'{title}\nOriginal Pre-trained Models'
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Set x-axis labels with cleaned model names
    ax.set_xticks(x_positions)
    clean_model_labels = [clean_model_name(model) for model in models]
    ax.set_xticklabels(clean_model_labels, rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits
    if all_values:
        y_min = min(all_values) - 0.05
        y_max = max(all_values) + 0.05
        ax.set_ylim(y_min, y_max)
    
    # Add statistics text
    for i, model in enumerate(models):
        values = data[model]
        if values:
            mean_val = np.mean(values)
            n_samples = len(values)
            ax.text(i, mean_val + 0.01, f'n={n_samples}\nmean={mean_val:.3f}', 
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to collect data and create plot."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot DMS fitness prediction performance')
    parser.add_argument('--type', type=str, choices=['taxon', 'model'], default='taxon',
                        help='Type of plot: by taxon or by model')
    parser.add_argument('--models', default="nemo2_evo2_7b_1m", type=str, nargs='+', 
                        help='List of model names or substrings to match. Always includes base model "nemo2_evo2_7b_1m". Use "all" for all models.')
    parser.add_argument('--base-path', default="results/virus_ecoli/full", type=str,
                        help='Base path to the results directory (default: results/virus_ecoli/full)')
    args = parser.parse_args()
    
    # Debug and ensure models is properly handled as a list of strings
    print(f"Debug: args.models = {args.models}")
    print(f"Debug: type(args.models) = {type(args.models)}")
    if args.models:
        print(f"Debug: args.models[0] = {args.models[0]}")
        print(f"Debug: type(args.models[0]) = {type(args.models[0])}")
    
    # Ensure models is properly handled as a list of strings
    if args.models and isinstance(args.models, list):
        # args.models should already be a list of strings from nargs='+'
        pass
    elif args.models and isinstance(args.models, str):
        # If somehow it's a single string, make it a list
        args.models = [args.models]
    
    if args.type == 'taxon':
        if args.models:
            if len(args.models) == 1 and args.models[0] == "all":
                print("Collecting fitness data by taxon from all models...")
            else:
                print(f"Collecting fitness data by taxon using substring matching: {', '.join(args.models)}")
                print("Note: Base model 'nemo2_evo2_7b_1m' will always be included")
        else:
            print("Collecting fitness data by taxon from all models...")
        
        data = collect_fitness_data(model_names=args.models, base_path=args.base_path)
        
        # Print summary
        print("\nSummary:")
        for taxon, values in data.items():
            if values:
                print(f"{taxon}: {len(values)} experiments, mean |Spearman| = {np.mean(values):.4f}")
            else:
                print(f"{taxon}: No data found")
        
        # Create and save plot
        print("\nCreating plot by taxon...")
        fig = create_fitness_plot_by_taxon(data, model_names=args.models)
        
        # Save the plot with dynamic filename using new naming convention
        experiment_name = extract_experiment_name(args.base_path)
        sample_type = extract_sample_type(args.base_path)
        
        if args.models:
            # Handle "all" case specially
            if len(args.models) == 1 and args.models[0] == "all":
                output_path = generate_plot_filename("taxon", None, experiment_name, sample_type, is_all_models=True)
            else:
                # Extract dataset identifier for filename
                dataset_identifier = None
                if args.models:
                    dataset_identifier = extract_dataset_identifier(args.models[0])
                
                output_path = generate_plot_filename("taxon", dataset_identifier, experiment_name, sample_type, is_all_models=False)
        else:
            output_path = generate_plot_filename("taxon", None, experiment_name, sample_type, is_all_models=True)
        
    elif args.type == 'model':
        if not args.models:
            print("Error: --models parameter is required when plotting by model")
            return
            
        if len(args.models) == 1 and args.models[0] == "all":
            print("Collecting fitness data by model for Virus taxon: all models")
        else:
            print(f"Collecting fitness data by model for Virus taxon using substring matching: {', '.join(args.models)}")
            print("Note: Base model 'nemo2_evo2_7b_1m' will always be included")
        
        data, actual_model_names = collect_fitness_data_by_model(model_names=args.models, base_path=args.base_path)
        
        # Print summary
        print("\nSummary:")
        for model, values in data.items():
            if values:
                print(f"{model}: {len(values)} experiments, mean |Spearman| = {np.mean(values):.4f}")
            else:
                print(f"{model}: No data found")
        
        # Create and save plot
        print("\nCreating plot by model...")
        fig = create_fitness_plot_by_model(data, model_names=actual_model_names)
        
        # Save the plot with dynamic filename using new naming convention
        experiment_name = extract_experiment_name(args.base_path)
        sample_type = extract_sample_type(args.base_path)
        
        # Handle "all" case specially
        if len(args.models) == 1 and args.models[0] == "all":
            output_path = generate_plot_filename("model", None, experiment_name, sample_type, is_all_models=True)
        else:
            # Extract dataset identifier from the first model that has one
            dataset_identifier = None
            for model_name in actual_model_names:
                dataset_identifier = extract_dataset_identifier(model_name)
                if dataset_identifier:
                    break
            
            output_path = generate_plot_filename("model", dataset_identifier, experiment_name, sample_type, is_all_models=False)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()