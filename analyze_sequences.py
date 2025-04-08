#!/usr/bin/env python3

import pandas as pd
import sys
import argparse

def analyze_sequences(csv_file):
    """
    Analyze sequences from CSV file to count those with length less than 8192
    and calculate the percentage of the total.
    
    Args:
        csv_file (str): Path to the CSV file
    
    Returns:
        tuple: (count_under_threshold, total_count, percentage)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        if 'length' not in df.columns:
            print(f"Error: 'length' column not found in {csv_file}")
            sys.exit(1)
            
        if 'basescan_contig_id' not in df.columns:
            print(f"Warning: 'basescan_contig_id' column not found in {csv_file}")
        
        # Count total rows
        total_count = len(df)
        
        # Count sequences with length < 8192
        count_under_threshold = len(df[df['length'] < 8192])
        
        # Calculate percentage
        percentage = (count_under_threshold / total_count) * 100 if total_count > 0 else 0
        
        return count_under_threshold, total_count, percentage
    
    except Exception as e:
        print(f"Error analyzing the CSV file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Analyze sequence lengths from a CSV file.')
    parser.add_argument('csv_file', help='Path to the CSV file containing the sequences')
    args = parser.parse_args()
    
    count_under_threshold, total_count, percentage = analyze_sequences(args.csv_file)
    
    print(f"Total number of sequences: {total_count}")
    print(f"Number of sequences with length < 8192: {count_under_threshold}")
    print(f"Percentage of sequences with length < 8192: {percentage:.2f}%")

if __name__ == "__main__":
    main() 