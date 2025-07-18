import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def collect_fitness_data(model_names=None):
    """Collect Spearman correlation values from all fitness CSV files, filtered by model names."""
    
    base_path = "results/test/nucleotides"
    taxons = ["Human", "Virus", "Eukaryote","Prokaryote"]
    
    data = {}
    
    for taxon in taxons:
        data[taxon] = []
        taxon_path = os.path.join(base_path, taxon)
        
        if not os.path.exists(taxon_path):
            continue
            
        # If model_names is specified, look in those model folders
        if model_names:
            # Ensure model_names is a list
            if isinstance(model_names, str):
                model_names = [model_names]
                
            for model_name in model_names:
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

def collect_fitness_data_by_model(model_names):
    """Collect Spearman correlation values by model for Virus taxon only."""
    
    base_path = "results/DMS_ProteinGym_substitutions"
    taxon = "Virus"
    
    data = {}
    
    # Ensure model_names is a list
    if isinstance(model_names, str):
        model_names = [model_names]
    
    taxon_path = os.path.join(base_path, taxon)
    
    if not os.path.exists(taxon_path):
        print(f"Warning: {taxon_path} does not exist")
        return data
    
    for model_name in model_names:
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
    
    return data

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
        models_str = ", ".join(model_names)
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
    models = list(data.keys())
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
    ax.set_title('DMS Fitness Prediction Performance by Model (Virus Taxon)', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits
    if all_values:
        y_min = min(all_values) - 0.05
        y_max = max(all_values) + 0.05
        ax.set_ylim(y_min, y_max)
    
    # Add statistics text
    for i, (model, values) in enumerate(data.items()):
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
                        help='List of model names to analyze')
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
            print(f"Collecting fitness data by taxon for models: {', '.join(args.models)}")
        else:
            print("Collecting fitness data by taxon from all models...")
        
        data = collect_fitness_data(model_names=args.models)
        
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
        
        # Save the plot with dynamic filename
        if args.models:
            # Fix filename generation to properly handle model names
            models_str = "-".join(args.models)  # Use hyphen instead of underscore
            output_path = f"fitness_spearman_by_taxon_{models_str}.png"
        else:
            output_path = "fitness_spearman_by_taxon_all_models.png"
        
    elif args.type == 'model':
        if not args.models:
            print("Error: --models parameter is required when plotting by model")
            return
            
        print(f"Collecting fitness data by model for Virus taxon: {', '.join(args.models)}")
        
        data = collect_fitness_data_by_model(model_names=args.models)
        
        # Print summary
        print("\nSummary:")
        for model, values in data.items():
            if values:
                print(f"{model}: {len(values)} experiments, mean |Spearman| = {np.mean(values):.4f}")
            else:
                print(f"{model}: No data found")
        
        # Create and save plot
        print("\nCreating plot by model...")
        fig = create_fitness_plot_by_model(data, model_names=args.models)
        
        # Save the plot with dynamic filename
        # Fix filename generation to properly handle model names
        models_str = "-".join(args.models)  # Use hyphen instead of underscore
        output_path = f"fitness_spearman_by_model_virus_{models_str}.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()