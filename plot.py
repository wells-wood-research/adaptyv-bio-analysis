import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import plotly.express as px

def plot_kd_vs_bits(data: pd.DataFrame, output_folder: Path):
    """Plot kd vs bits_swissprot and bits_pdb."""
    # Matplotlib plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for design, group in data.groupby("design_type"):
        axes[0].scatter(group["bits_swissprot"], group["kd"], alpha=0.8, label=design, s=50)
    axes[0].set_xlabel("bits_swissprot (%)", fontsize=12)
    axes[0].set_ylabel("kd (log scale)", fontsize=12)
    axes[0].set_xlim(0, 100)
    axes[0].set_yscale("log")
    axes[0].set_title("Log-scale kd vs bits_swissprot", fontsize=14)
    axes[0].legend(title="Design Type", fontsize=10)
    axes[0].grid()

    for design, group in data.groupby("design_type"):
        axes[1].scatter(group["bits_pdb"], group["kd"], alpha=0.8, label=design, s=50)
    axes[1].set_xlabel("bits_pdb (%)", fontsize=12)
    axes[1].set_xlim(0, 100)
    axes[1].set_title("Log-scale kd vs bits_pdb", fontsize=14)
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(output_folder / "kd_vs_bits.pdf")
    plt.close()

    # Plotly interactive plot
    fig_html = px.scatter(
        data, x="bits_swissprot", y="kd", color="design_type",
        title="Log-scale kd vs bits_swissprot", log_y=True, labels={"bits_swissprot": "bits_swissprot (%)", "kd": "kd"}
    )
    fig_html.write_html(str(output_folder / "kd_vs_bits.html"))

def plot_kd_vs_alntmscore(data: pd.DataFrame, output_folder: Path):
    """Plot kd vs alntmscore_swissprot and alntmscore_pdb."""
    # Matplotlib plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for design, group in data.groupby("design_type"):
        axes[0].scatter(group["alntmscore_swissprot"], group["kd"], alpha=0.7, label=design, s=50)
    axes[0].set_xlabel("alntmscore_swissprot (%)", fontsize=12)
    axes[0].set_ylabel("kd (log scale)", fontsize=12)
    axes[0].set_xlim(0, 1)
    axes[0].set_yscale("log")
    axes[0].set_title("Log-scale kd vs alntmscore_swissprot", fontsize=14)
    axes[0].legend(title="Design Type", fontsize=10)
    axes[0].grid()

    for design, group in data.groupby("design_type"):
        axes[1].scatter(group["alntmscore_pdb"], group["kd"], alpha=0.7, label=design, s=50)
    axes[1].set_xlabel("alntmscore_pdb", fontsize=12)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Log-scale kd vs alntmscore_pdb", fontsize=14)
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(output_folder / "kd_vs_alntmscore.pdf")
    plt.close()

    # Plotly interactive plot
    fig_html = px.scatter(
        data, x="alntmscore_swissprot", y="kd", color="design_type",
        title="Log-scale kd vs alntmscore_swissprot", log_y=True, labels={"alntmscore_swissprot": "alntmscore_swissprot (%)", "kd": "kd"}
    )
    fig_html.write_html(str(output_folder / "kd_vs_alntmscore.html"))

def main(args):
    args.input = Path(args.input)
    args.output_folder = Path(args.output_folder)

    assert args.input.exists(), f"Input file {args.input} does not exist"
    args.output_folder.mkdir(parents=True, exist_ok=True)

    # Load and validate the dataset
    data = pd.read_csv(args.input, sep='\t')
    required_columns = {"bits_swissprot", "bits_pdb", "kd", "alntmscore_swissprot", "alntmscore_pdb", "similarity_check"}

    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column {col} in the dataset")


    # Add "design_type" based on "similarity_check"
    data['design_type'] = data['similarity_check'].apply(lambda x: 'de novo' if pd.isna(x) else 'redesign')


    # Generate plots
    plot_kd_vs_bits(data, args.output_folder)
    plot_kd_vs_alntmscore(data, args.output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for kd vs features.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input dataset file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder where plots will be saved.")

    args = parser.parse_args()
    main(args)