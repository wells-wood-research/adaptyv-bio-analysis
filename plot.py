import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, pearsonr
import os


def load_and_merge_embeddings(df: pd.DataFrame, embeddings_path: str) -> pd.DataFrame:
    """Load embeddings and merge with the input dataset."""
    embeddings_df = pd.read_csv(embeddings_path)
    df = pd.merge(df, embeddings_df, how="outer", on="sequence")
    # Drop any commonid that is nan
    df = df.dropna(subset=['common_id'])
    return df


def visualize_embeddings(
    df,
    hue_col,
    method="PCA",
    perplexity=30,
    n_components=2,
    n_clusters=None,
    save_dir=".",
    filter_binding=False,
):
    """
    Visualize protein embeddings using PCA or t-SNE and optionally perform clustering.

    Parameters:
    - filter_binding (bool): If True, filter rows where `binding == TRUE`.
    """
    if filter_binding and "binding" in df.columns:
        df = df[df["binding"] == True]

    embeddings_df = df[df["esm2_3b_dim_2560"].notna()]  # Filter rows with embeddings
    embedding_columns = [
        col for col in embeddings_df.columns if col.startswith("esm2_3b")
    ]
    embeddings = embeddings_df[embedding_columns]

    if method == "PCA":
        reducer = PCA(n_components=n_components)
    elif method == "t-SNE":
        reducer = TSNE(
            n_components=n_components, perplexity=perplexity, random_state=42
        )
    else:
        raise ValueError("Invalid method. Choose 'PCA' or 't-SNE'.")

    reduced_result = reducer.fit_transform(embeddings)
    embeddings_df["Component1"] = reduced_result[:, 0]
    embeddings_df["Component2"] = reduced_result[:, 1]

    if n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        embeddings_df["Cluster"] = kmeans.fit_predict(embeddings)

    # Matplotlib scatter plot
    plt.figure(figsize=(7, 7))
    plt.scatter(
        embeddings_df["Component1"],
        embeddings_df["Component2"],
        c=embeddings_df[hue_col] if hue_col in embeddings_df else None,
        cmap="viridis",
        alpha=0.7,
        edgecolor="k",
    )
    plt.title(f"{method} Visualization of Protein Embeddings", fontsize=14)
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.colorbar(label=hue_col)
    plt.grid(alpha=0.3)

    # Save as PDF
    filename_suffix = "_binding_TRUE" if filter_binding else ""
    pdf_path = os.path.join(
        save_dir, f"esm2_3b_{method}_col_by_{hue_col}{filename_suffix}.pdf"
    )
    plt.savefig(pdf_path)
    plt.close()

    print(f"PDF plot saved to {pdf_path}")

    if method == "PCA":
        print(f"Explained variance by components: {reducer.explained_variance_ratio_}")


def visualize_embeddings_interactive(
    df, hue_col, method='PCA', perplexity=30, n_components=2, n_clusters=None, save_dir=".", filter_binding=False
):
    """
    Interactive visualization of protein embeddings with optional filtering and PDF export.

    Parameters:
    - df (DataFrame): Input dataframe containing embeddings and metadata.
    - hue_col (str): Column name to color the plot by.
    - method (str): Dimensionality reduction method ('PCA' or 't-SNE').
    - perplexity (int): Perplexity parameter for t-SNE (only used for t-SNE).
    - n_components (int): Number of reduced dimensions (default: 2).
    - n_clusters (int): Number of clusters for KMeans clustering (default: None).
    - save_dir (str): Directory to save plots.
    - filter_binding (bool): If True, only use rows where 'binding' == 'TRUE'.

    Returns:
    - DataFrame with reduced dimensions and optional clustering.
    """
    if filter_binding:
        df = df[df['binding'] == "TRUE"]

    embeddings_df = df[df['esm2_3b_dim_2560'].notna()]
    embedding_columns = [col for col in embeddings_df.columns if col.startswith('esm2_3b')]
    embeddings = embeddings_df[embedding_columns]

    # Dimensionality reduction
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    else:
        raise ValueError("Invalid method. Choose 'PCA' or 't-SNE'.")

    reduced_result = reducer.fit_transform(embeddings)
    embeddings_df['Component1'] = reduced_result[:, 0]
    embeddings_df['Component2'] = reduced_result[:, 1]

    # Clustering (optional)
    if n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        embeddings_df['Cluster'] = kmeans.fit_predict(embeddings)

    # Save interactive plot (HTML)
    fig = px.scatter(
        embeddings_df,
        x='Component1',
        y='Component2',
        color=hue_col,
        title=f"{method} of ESM2 3B Embeddings",
        labels={"Component1": "Component 1", "Component2": "Component 2"},
        hover_data=["name", "username", "kd"],
        color_continuous_scale='viridis',
    )
    fig.update_traces(marker=dict(size=7, opacity=1))
    fig.update_layout(width=800, height=700)

    html_path = os.path.join(save_dir, f"esm2_3b_{method}_col_by_{hue_col}.html")
    fig.write_html(html_path)
    print(f"Interactive plot saved to {html_path}")

    # Save static plot (PDF)
    plt.figure(figsize=(7, 7))
    plt.scatter(
        embeddings_df['Component1'],
        embeddings_df['Component2'],
        c=embeddings_df[hue_col] if hue_col in embeddings_df else None,
        cmap='viridis', alpha=0.7, edgecolor='k'
    )
    plt.title(f"{method} Visualization of Protein Embeddings", fontsize=14)
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.colorbar(label=hue_col)
    plt.grid(alpha=0.3)

    pdf_path = os.path.join(save_dir, f"esm2_3b_{method}_col_by_{hue_col}.pdf")
    plt.savefig(pdf_path)
    print(f"PDF plot saved to {pdf_path}")
    plt.close()

    if method == 'PCA':
        print(f"Explained variance by components: {reducer.explained_variance_ratio_}")

    return embeddings_df


def plot_kd_vs_metric(
    data: pd.DataFrame, metric: str, output_folder: Path, database: str
):
    """Plot kd vs a specific metric with consideration for database variants."""
    # Check if the metric has SwissProt and/or PDB columns
    has_swissprot = f"{metric}_swissprot" in data.columns
    has_pdb = f"{metric}_pdb" in data.columns

    if has_swissprot or has_pdb:
        # If the metric has database-specific variants
        if database == "both" and has_swissprot and has_pdb:
            _plot_metric_both(data, metric, output_folder)
        else:
            db_label = database
            metric_column = f"{metric}_{db_label}"
            if metric_column in data.columns:
                _plot_metric(data, metric_column, db_label, output_folder)
    else:
        # If the metric is standalone (e.g., similarity_check)
        _plot_single_metric(data, metric, output_folder)


def _determine_x_limit(max_value: float) -> float:
    """Determine x-axis upper limit as either 1 or 100, whichever is closer to the maximum."""
    return 1 if abs(max_value - 1) < abs(max_value - 100) else 100


def compute_correlations(data: pd.DataFrame, output_folder: Path):
    """Compute Pearson and Spearman correlations for all floating-point metrics and save to CSV."""
    correlations = []
    # Select all floating-point columns except 'kd'
    float_metrics = [
        col for col in data.select_dtypes(include=["float"]).columns if col != "kd"
    ]

    for metric in float_metrics:
        # Remove NaN values
        curr_data = data.dropna(subset=[metric, "kd"])
        # Calculate Pearson correlation and p-value
        pearson_corr, pearson_pval = pearsonr(curr_data[metric], curr_data["kd"])
        spearman_corr, spearman_pval = spearmanr(curr_data[metric], curr_data["kd"])
        correlations.append(
            {
                "metric": metric,
                "pearson_correlation": pearson_corr,
                "pearson_p_value": pearson_pval,
                "spearman_correlation": spearman_corr,
                "spearman_p_value": spearman_pval,
            }
        )

    # Sort and save correlations
    correlations_df = pd.DataFrame(correlations).sort_values(
        by="spearman_p_value", ascending=True
    )
    correlations_path = output_folder / "correlations_sorted_by_spearman.csv"
    correlations_df.to_csv(correlations_path, index=False)
    print(f"Correlations saved to {correlations_path}")


def _create_interactive_plot(
    data: pd.DataFrame, x_col: str, y_col: str, output_path: Path, x_max: float
):
    """Create an interactive Plotly scatter plot and save it as an HTML file."""
    fig_html = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color="design_type",
        hover_data=["common_id"],  # Add common_id for hover
        title=f"Log-scale {y_col} vs {x_col}",
        log_y=True,
        labels={x_col: x_col, y_col: y_col},
    )
    fig_html.update_layout(
        xaxis=dict(range=[0, x_max]),  # Set x-axis limits
        showlegend=False,
    )
    fig_html.write_html(str(output_path))


def _plot_single_metric(data: pd.DataFrame, metric: str, output_folder: Path):
    """Generate both Matplotlib and Plotly plots for kd vs a non-database-specific metric."""
    x_max = _determine_x_limit(data[metric].max())

    # Matplotlib plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for design, group in data.groupby("design_type"):
        ax.scatter(group[metric], group["kd"], alpha=0.7, s=50, label=design)

    ax.set_xlabel(metric, fontsize=12)
    ax.set_ylabel("kd (log scale)", fontsize=12)
    ax.set_xlim(0, x_max)
    ax.set_yscale("log")
    ax.set_title(f"Log-scale kd vs {metric}", fontsize=14)
    ax.grid()
    ax.legend(title="Design Type", fontsize=10)

    plt.tight_layout()
    pdf_path = output_folder / f"kd_vs_{metric}.pdf"
    plt.savefig(pdf_path)
    plt.close()

    # Plotly interactive plot
    html_path = output_folder / f"kd_vs_{metric}.html"
    _create_interactive_plot(
        data, x_col=metric, y_col="kd", output_path=html_path, x_max=x_max
    )


def _plot_metric(
    data: pd.DataFrame, metric_column: str, db_label: str, output_folder: Path
):
    """Generate both Matplotlib and Plotly plots for kd vs a database-specific metric."""
    x_max = _determine_x_limit(data[metric_column].max())

    # Matplotlib plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for design, group in data.groupby("design_type"):
        ax.scatter(group[metric_column], group["kd"], alpha=0.7, s=50, label=design)

    ax.set_xlabel(f"{metric_column} ({db_label})", fontsize=12)
    ax.set_ylabel("kd (log scale)", fontsize=12)
    ax.set_xlim(0, x_max)
    ax.set_yscale("log")
    ax.set_title(f"Log-scale kd vs {metric_column} ({db_label})", fontsize=14)
    ax.grid()
    ax.legend(title="Design Type", fontsize=10)

    plt.tight_layout()
    pdf_path = output_folder / f"kd_vs_{metric_column}_{db_label}.pdf"
    plt.savefig(pdf_path)
    plt.close()

    # Plotly interactive plot
    html_path = output_folder / f"kd_vs_{metric_column}_{db_label}.html"
    _create_interactive_plot(
        data, x_col=metric_column, y_col="kd", output_path=html_path, x_max=x_max
    )


def _plot_metric_both(data: pd.DataFrame, metric: str, output_folder: Path):
    """Generate subplots for SwissProt and PDB metrics, and separate interactive Plotly plots."""
    swissprot_column = f"{metric}_swissprot"
    pdb_column = f"{metric}_pdb"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # SwissProt Plot
    if swissprot_column in data.columns:
        x_max_swissprot = _determine_x_limit(data[swissprot_column].max())
        for design, group in data.groupby("design_type"):
            axes[0].scatter(
                group[swissprot_column], group["kd"], alpha=0.7, s=50, label=design
            )
        axes[0].set_xlabel(f"{swissprot_column} (SwissProt)", fontsize=12)
        axes[0].set_xlim(0, x_max_swissprot)
        axes[0].set_ylabel("kd (log scale)", fontsize=12)
        axes[0].set_title(
            f"Log-scale kd vs {swissprot_column} (SwissProt)", fontsize=14
        )
        axes[0].set_yscale("log")
        axes[0].grid()
        axes[0].legend(title="Design Type", fontsize=10)

        # Generate interactive plot for SwissProt
        html_path_swissprot = output_folder / f"kd_vs_{swissprot_column}.html"
        _create_interactive_plot(
            data,
            x_col=swissprot_column,
            y_col="kd",
            output_path=html_path_swissprot,
            x_max=x_max_swissprot,
        )

    # PDB Plot
    if pdb_column in data.columns:
        x_max_pdb = _determine_x_limit(data[pdb_column].max())
        for design, group in data.groupby("design_type"):
            axes[1].scatter(
                group[pdb_column], group["kd"], alpha=0.7, s=50, label=design
            )
        axes[1].set_xlabel(f"{pdb_column} (PDB)", fontsize=12)
        axes[1].set_xlim(0, x_max_pdb)
        axes[1].set_title(f"Log-scale kd vs {pdb_column} (PDB)", fontsize=14)
        axes[1].grid()

        # Generate interactive plot for PDB
        html_path_pdb = output_folder / f"kd_vs_{pdb_column}.html"
        _create_interactive_plot(
            data,
            x_col=pdb_column,
            y_col="kd",
            output_path=html_path_pdb,
            x_max=x_max_pdb,
        )

    plt.tight_layout()
    pdf_path = output_folder / f"kd_vs_{metric}_both.pdf"
    plt.savefig(pdf_path)
    plt.close()


def validate_required_columns(data: pd.DataFrame, database: str, metrics: list):
    """Validate that all required columns for the selected database exist in the dataset."""
    required_columns = ["kd", "design_type"]

    for metric in metrics:
        has_swissprot = f"{metric}_swissprot" in data.columns
        has_pdb = f"{metric}_pdb" in data.columns

        if database in ["swissprot", "both"] and has_swissprot:
            required_columns.append(f"{metric}_swissprot")
        if database in ["pdb", "both"] and has_pdb:
            required_columns.append(f"{metric}_pdb")

        if not has_swissprot and not has_pdb:
            required_columns.append(metric)

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def main(args):
    args.input = Path(args.input)
    args.output_folder = Path(args.output_folder)
    assert args.input.exists(), f"Input file {args.input} does not exist"
    args.output_folder.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, sep="\t")
    data["design_type"] = data["similarity_check"].apply(
        lambda x: "de novo" if pd.isna(x) else "redesign"
    )

    metrics = [
        "bits",
        "alntmscore",
        "evalue",
        "prob",
        "lddt",
        "qtmscore",
        "fident",
        "similarity_check",
    ]

    validate_required_columns(data, args.database, metrics)
    compute_correlations(data, args.output_folder)

    for metric in metrics:
        plot_kd_vs_metric(data, metric, args.output_folder, args.database)

    data = load_and_merge_embeddings(data, args.embeddings)

    # Visualize embeddings (all data)
    # visualize_embeddings_interactive(
    #     df=data,
    #     hue_col="design_type",
    #     method="PCA",
    #     save_dir=args.output_folder,
    # )
    # visualize_embeddings_interactive(
    #     df=data,
    #     hue_col="Cluster" if "Cluster" in data.columns else "design_type",
    #     method="t-SNE",
    #     perplexity=30,
    #     save_dir=args.output_folder,
    # )

    # Visualize embeddings (binding == TRUE)
    visualize_embeddings_interactive(
        df=data,
        hue_col="design_type",
        method="PCA",
        save_dir=args.output_folder,
        filter_binding=True,
    )
    visualize_embeddings_interactive(
        df=data,
        hue_col="Cluster" if "Cluster" in data.columns else "design_type",
        method="t-SNE",
        perplexity=30,
        save_dir=args.output_folder,
        filter_binding=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots and embedding visualizations."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input file.")
    parser.add_argument(
        "--embeddings", type=str, required=True, help="Path to embeddings file."
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Path to output folder."
    )
    parser.add_argument(
        "--database",
        type=str,
        default="both",
        choices=["swissprot", "pdb", "both"],
        help="Specify database for plotting: swissprot, pdb, or both.",
    )
    args = parser.parse_args()
    main(args)
