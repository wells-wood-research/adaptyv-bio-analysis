import pandas as pd
import argparse

def merge_datasets(foldseek_file, similarity_file, output_file):
    # Load Foldseek merged results
    foldseek_df = pd.read_csv(foldseek_file, sep='\t')

    # Load sequence similarity dataset
    similarity_df = pd.read_csv(similarity_file)

    # Merge datasets on the 'common_id' column
    merged_df = pd.merge(
        foldseek_df, 
        similarity_df, 
        left_on='common_id', 
        right_on='id', 
        how='left'
    )

    # Save the merged dataframe to a file
    merged_df.to_csv(output_file, sep='\t', index=False)

    print(f"Merged dataset saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge Foldseek results with sequence similarity dataset.")
    parser.add_argument("--foldseek", type=str, required=True, help="Path to the Foldseek merged results file.")
    parser.add_argument("--similarity", type=str, required=True, help="Path to the sequence similarity dataset file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the merged output file.")

    args = parser.parse_args()

    merge_datasets(args.foldseek, args.similarity, args.output)

if __name__ == "__main__":
    main()
