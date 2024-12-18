import argparse
import pandas as pd

def merge_files(csv_file: str, tsv_file: str, output_file: str):
    """
    Merges two files based on `common_id` in the TSV file and `design_name` in the CSV file.
    Cleans `design_name` by removing `_chainA` suffix and appends `_binder` to all CSV column names except `design_name`.
    """
    # Load files
    csv_data = pd.read_csv(csv_file)
    tsv_data = pd.read_csv(tsv_file, sep='\t')

    # Verify required columns exist
    if 'design_name' not in csv_data.columns:
        raise ValueError("The column `design_name` is missing in the CSV file.")
    if 'common_id' not in tsv_data.columns:
        raise ValueError("The column `common_id` is missing in the TSV file.")

    # Remove '_chainA' suffix from the `design_name` column
    csv_data['design_name'] = csv_data['design_name'].str.replace('_chainA$', '', regex=True)

    # Append "_binder" to all column names in the CSV except `design_name`
    csv_data = csv_data.rename(columns=lambda col: f"{col}_binder" if col != 'design_name' else col)

    # Merge files on `common_id` (TSV) and `design_name` (CSV)
    merged_data = tsv_data.merge(csv_data, left_on='common_id', right_on='design_name', how='left')

    # Save the merged data to the output file
    merged_data.to_csv(output_file, sep='\t', index=False)
    print(f"Successfully merged files and saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge TSV and CSV files.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--tsv_file", type=str, required=True, help="Path to the input TSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the merged output file.")

    args = parser.parse_args()
    merge_files(args.csv_file, args.tsv_file, args.output_file)
