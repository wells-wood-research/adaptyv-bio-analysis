import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import plotly.express as px
from scipy.stats import spearmanr, pearsonr


def compute_correlations(data: pd.DataFrame, output_folder: Path):
    """Compute Pearson and Spearman correlations for all floating-point metrics and save to CSV, sorted by Spearman coefficient."""
    correlations = []
    # Select all floating-point columns except 'kd'
    float_metrics = [col for col in data.select_dtypes(include=["float"]).columns if col != "kd"]

    for metric in float_metrics:
        # Remove NaN values
        curr_data = data.dropna(subset=[metric, "kd"])
        # Calculate Pearson correlation and p-value
        pearson_corr, pearson_pval = pearsonr(curr_data[metric], curr_data["kd"])
        # Calculate Spearman correlation and p-value
        spearman_corr, spearman_pval = spearmanr(curr_data[metric], curr_data["kd"])
        # Append results
        correlations.append(
            {
                "metric": metric,
                "pearson_correlation": pearson_corr,
                "pearson_p_value": pearson_pval,
                "spearman_correlation": spearman_corr,
                "spearman_p_value": spearman_pval,
            }
        )

    # Convert to DataFrame and sort by Spearman correlation coefficient
    correlations_df = pd.DataFrame(correlations)
    correlations_df = correlations_df.sort_values(by="spearman_p_value", ascending=True)

    # Save to CSV
    correlations_csv_path = output_folder / "correlations_sorted_by_spearman.csv"
    correlations_df.to_csv(correlations_csv_path, index=False)
    print(f"Correlations saved to {correlations_csv_path}")


def plot_kd_vs_bits(data: pd.DataFrame, output_folder: Path):
    """Plot kd vs bits_swissprot and bits_pdb."""
    # Matplotlib plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for design, group in data.groupby("design_type"):
        axes[0].scatter(
            group["bits_swissprot"], group["kd"], alpha=0.8, label=design, s=50
        )
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
        data,
        x="bits_swissprot",
        y="kd",
        color="design_type",
        title="Log-scale kd vs bits_swissprot",
        log_y=True,
        labels={"bits_swissprot": "bits_swissprot (%)", "kd": "kd"},
    )
    fig_html.write_html(str(output_folder / "kd_vs_bits.html"))


def plot_kd_vs_alntmscore(data: pd.DataFrame, output_folder: Path):
    """Plot kd vs alntmscore_swissprot and alntmscore_pdb."""
    # Matplotlib plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for design, group in data.groupby("design_type"):
        axes[0].scatter(
            group["alntmscore_swissprot"], group["kd"], alpha=0.7, label=design, s=50
        )
    axes[0].set_xlabel("alntmscore_swissprot (%)", fontsize=12)
    axes[0].set_ylabel("kd (log scale)", fontsize=12)
    axes[0].set_xlim(0, 1)
    axes[0].set_yscale("log")
    axes[0].set_title("Log-scale kd vs alntmscore_swissprot", fontsize=14)
    axes[0].legend(title="Design Type", fontsize=10)
    axes[0].grid()

    for design, group in data.groupby("design_type"):
        axes[1].scatter(
            group["alntmscore_pdb"], group["kd"], alpha=0.7, label=design, s=50
        )
    axes[1].set_xlabel("alntmscore_pdb", fontsize=12)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Log-scale kd vs alntmscore_pdb", fontsize=14)
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(output_folder / "kd_vs_alntmscore.pdf")
    plt.close()

    # Plotly interactive plot
    fig_html = px.scatter(
        data,
        x="alntmscore_swissprot",
        y="kd",
        color="design_type",
        title="Log-scale kd vs alntmscore_swissprot",
        log_y=True,
        labels={"alntmscore_swissprot": "alntmscore_swissprot (%)", "kd": "kd"},
    )
    fig_html.write_html(str(output_folder / "kd_vs_alntmscore.html"))


def plot_kd_vs_sequence_similarity(data: pd.DataFrame, output_folder: Path):
    """Plot kd vs sequence similarity with design_type as hue (legend hidden)."""
    # Matplotlib plot
    fig, ax = plt.subplots(figsize=(7, 5))

    for design, group in data.groupby("design_type"):
        ax.scatter(
            group["similarity_check"], group["kd"], alpha=0.7, s=50, label=design
        )
    ax.set_xlabel("Sequence Similarity (%)", fontsize=12)
    ax.set_ylabel("kd (log scale)", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.set_title("Log-scale kd vs Sequence Similarity", fontsize=14)
    ax.grid()
    # Do not show the legend
    ax.legend().remove()

    plt.tight_layout()
    plt.savefig(output_folder / "kd_vs_sequence_similarity.pdf")
    plt.close()

    # Plotly interactive plot
    fig_html = px.scatter(
        data,
        x="similarity_check",
        y="kd",
        color="design_type",
        title="Log-scale kd vs Sequence Similarity",
        log_y=True,
        labels={"similarity_check": "Sequence Similarity (%)", "kd": "kd"},
    )
    # Hide legend in Plotly
    fig_html.update_layout(showlegend=False)
    fig_html.write_html(str(output_folder / "kd_vs_sequence_similarity.html"))



def main(args):
    args.input = Path(args.input)
    args.output_folder = Path(args.output_folder)

    assert args.input.exists(), f"Input file {args.input} does not exist"
    args.output_folder.mkdir(parents=True, exist_ok=True)

    # Load and validate the dataset
    data = pd.read_csv(args.input, sep="\t")
    required_columns = {
        "bits_swissprot",
        "bits_pdb",
        "kd",
        "alntmscore_swissprot",
        "alntmscore_pdb",
        "similarity_check",
    }

    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column {col} in the dataset")

    # Add "design_type" based on "similarity_check"
    data["design_type"] = data["similarity_check"].apply(
        lambda x: "de novo" if pd.isna(x) else "redesign"
    )

    # Compute correlations with p-values
    compute_correlations(data, args.output_folder)
    # Generate plots
    plot_kd_vs_bits(data, args.output_folder)
    plot_kd_vs_alntmscore(data, args.output_folder)
    plot_kd_vs_sequence_similarity(data, args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for kd vs features.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input dataset file."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the output folder where plots will be saved.",
    )

    args = parser.parse_args()
    main(args)
