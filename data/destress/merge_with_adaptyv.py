import argparse
import pandas as pd

def merge_files(csv_file: str, tsv_file: str, output_file: str):
    """
    Merges two files based on specific columns: `design_name` from the CSV file
    and `name` from the TSV file. Outputs a merged file with additional columns.

    Args:
        csv_file (str): Path to the input CSV file.
        tsv_file (str): Path to the input TSV file.
        output_file (str): Path to save the merged output file.
    """
    # Load the CSV and TSV files
    csv_data = pd.read_csv(csv_file)
    tsv_data = pd.read_csv(tsv_file, sep='\t')

    # Merge files on specified columns
    merged_data = tsv_data.merge(csv_data, left_on='name', right_on='design_name', how='left')

    # Save the merged data to the output file as tsv
    merged_data.to_csv(output_file, sep='\t', index=False)
    print(f"Successfully merged files and saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a CSV file and a TSV file based on specific columns.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--tsv_file", type=str, required=True, help="Path to the input TSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the merged output file.")

    args = parser.parse_args()
    merge_files(args.csv_file, args.tsv_file, args.output_file)
