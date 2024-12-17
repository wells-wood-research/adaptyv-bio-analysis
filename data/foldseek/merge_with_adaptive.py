import pandas as pd
import argparse

def merge_datasets(foldseek_file, additional_dataset_file, output_file):
    # Load Foldseek results
    foldseek_df = pd.read_csv(foldseek_file, sep='\t')

    # Load additional dataset
    additional_df = pd.read_csv(additional_dataset_file, sep=',')

    # Extract 'common_id' from Foldseek results
    foldseek_df['common_id'] = foldseek_df['common_id']

    # Merge datasets on 'common_id'
    merged_df = pd.merge(foldseek_df, additional_df, left_on='common_id', right_on='name', how='outer')

    # Save merged data to output file
    merged_df.to_csv(output_file, sep='\t', index=False)

    print(f"Merged dataset saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge Foldseek results with additional dataset.")
    parser.add_argument("--foldseek", type=str, required=True, help="Path to the Foldseek merged results file.")
    parser.add_argument("--additional", type=str, required=True, help="Path to the additional dataset file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the merged output file.")

    args = parser.parse_args()

    merge_datasets(args.foldseek, args.additional, args.output)

if __name__ == "__main__":
    main()
