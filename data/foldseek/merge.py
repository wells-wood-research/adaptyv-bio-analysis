import os
import pandas as pd
import argparse

def extract_common_identifier(file_name):
    # Extract the common identifier by removing suffixes and extension
    return file_name.rsplit('_', 1)[0]

def process_files(directory):
    # Initialize dictionaries to store processed data
    processed_data = {}
    unmatched_files = []

    # Process each file in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith("_swissprot.tsv") or file_name.endswith("_pdb.tsv"):
            file_path = os.path.join(directory, file_name)

            # Extract common identifier
            common_id = extract_common_identifier(file_name)

            # Read the file into a dataframe
            df = pd.read_csv(file_path, sep='\t', header=None)

            # Extract the first line
            first_row = df.iloc[0].tolist()

            # Determine the suffix (swissprot or pdb)
            suffix = "swissprot" if "_swissprot.tsv" in file_name else "pdb"

            # Add entry to the dictionary
            if common_id not in processed_data:
                processed_data[common_id] = {"swissprot": None, "pdb": None}
            processed_data[common_id][suffix] = first_row

    # Prepare lists for matched and unmatched data
    matched_data = []

    for common_id, data in processed_data.items():
        if data["swissprot"] and data["pdb"]:
            matched_data.append((common_id, data["swissprot"], data["pdb"]))
        else:
            unmatched_files.append(common_id)

    # Define columns
    columns = [
        "query", "target", "fident", "alnlen", "mismatch", 
        "gapopen", "qstart", "qend", "tstart", "tend", "evalue", "bits",
        "alntmscore", "qtmscore", "ttmscore", "lddt", "lddtfull", "prob"
    ]

    # Build the merged dataframe
    rows = []
    for common_id, swissprot_row, pdb_row in matched_data:
        row = {"common_id": common_id}
        row.update({f"{col}_swissprot": value for col, value in zip(columns, swissprot_row)})
        row.update({f"{col}_pdb": value for col, value in zip(columns, pdb_row)})
        rows.append(row)

    merged_df = pd.DataFrame(rows)

    # Save unmatched entries to a file
    if unmatched_files:
        unmatched_file = os.path.join(directory, "unmatched_files.txt")
        with open(unmatched_file, "w") as f:
            f.write("\n".join(unmatched_files))
        print(f"Unmatched files written to {unmatched_file}")

    return merged_df

def main():
    parser = argparse.ArgumentParser(description="Merge Foldseek output files for SwissProt and PDB databases.")
    parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing Foldseek output files.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the merged output file.")

    args = parser.parse_args()

    # Process files and merge data
    merged_df = process_files(args.directory)

    # Save the merged dataframe to a file
    merged_df.to_csv(args.output, sep='\t', index=False)

    print(f"Merged file saved to {args.output}")

if __name__ == "__main__":
    main()
