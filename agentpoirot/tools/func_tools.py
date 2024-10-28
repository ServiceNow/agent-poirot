import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import numpy as np
from agentpoirot import utils as ut
from typing import List


def plot_bar(
    df: pd.DataFrame, plot_column: str, count_column: str, plot_title: str
) -> None:
    """
    Creates a bar plot with `plot_column` on x-axis and `count_column` on y-axis using Seaborn.

    This plot is useful for answering questions like:
    - "What are the counts of each category in the dataset?"
    - "How does the frequency of categories compare?"

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        plot_column (str): Column name for x-axis (categorical data).
        count_column (str): Column name for y-axis (numerical counts).
        plot_title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    b = sns.barplot(data=df, x=plot_column, y=count_column)
    plt.title(plot_title)
    if len(df[plot_column].unique()) > 30:
        b.set(xticklabels=[])
    else:
        plt.xticks(rotation=45)

    plt.tight_layout()
    ut.save_stats_fig(df[[plot_column, count_column]], fig=plt.gcf())


# def plot_pie_chart(
#     df: pd.DataFrame, plot_column: str, count_column: str, plot_title: str
# ) -> None:
#     """
#     Generates a pie chart using Matplotlib to show the proportion of each category.

#     This plot is useful for answering questions like:
#     - "What is the percentage distribution of categories?"
#     - "How much of the total does each category represent?"

#     Args:
#         df (pd.DataFrame): DataFrame containing the data.
#         plot_column (str): Column name for the categories (labels).
#         count_column (str): Column name for the values (numerical data).
#         plot_title (str): Title of the plot.
#     """
#     plt.figure(figsize=(8, 8))
#     plt.pie(df[count_column], labels=df[plot_column], autopct="%1.1f%%", startangle=90)
#     plt.title(plot_title)
#     plt.tight_layout()
#     ut.save_stats_fig(df[[plot_column, count_column]], fig=plt.gcf())


def plot_wordcloud(df: pd.DataFrame, group_by_column: str, plot_column: str) -> None:
    """
    Creates word clouds for each group in `group_by_column` based on text in `plot_column`.

    This plot is useful for answering questions like:
    - "What are the most common words for each group?"
    - "How does word frequency vary across different categories?"

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        group_by_column (str): Column name to group by (e.g., categories).
        plot_column (str): Column name for text data (e.g., reviews, comments).
    """
    grouped_data = df.groupby(group_by_column)[plot_column].apply(list).reset_index()
    for i, row in grouped_data.iterrows():
        wc = WordCloud(width=800, height=400).generate(" ".join(row[plot_column]))
        wc.to_file(f"wordcloud_{row[group_by_column]}.jpg")
    ut.save_stats_fig(df[[group_by_column, plot_column]])


def plot_lines(
    df: pd.DataFrame, x_column: str, plot_columns: List[str], plot_title: str
) -> None:
    """
    Plots multiple lines on a single graph with `x_column` on x-axis and `plot_columns` on y-axis using Seaborn.

    This plot is useful for answering questions like:
    - "How do different variables change over time or across another dimension?"
    - "What trends can be observed between variables?"

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_column (str): Column name for x-axis (e.g., time, date).
        plot_columns (List[str]): List of column names for y-axis (e.g., multiple variables to plot).
        plot_title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    for plot_column in plot_columns:
        sns.lineplot(data=df, x=x_column, y=plot_column, label=plot_column)
    plt.title(plot_title)
    plt.tight_layout()
    ut.save_stats_fig(df[[x_column] + plot_columns], fig=plt.gcf())


def plot_heatmap(
    df: pd.DataFrame, x_column: str, y_column: str, z_column: str, plot_title: str
) -> None:
    """
    Creates a heatmap showing the density of values across `x_column` and `y_column` with `z_column` as intensity using Seaborn.

    This plot is useful for answering questions like:
    - "How do values vary between two dimensions?"
    - "Where are the hotspots or patterns in the data?"

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_column (str): Column name for x-axis.
        y_column (str): Column name for y-axis.
        z_column (str): Column name for intensity values (color scale).
        plot_title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 8))
    pivot_table = df.pivot_table(values=z_column, index=y_column, columns=x_column)
    sns.heatmap(pivot_table, cmap="viridis", annot=True)
    plt.title(plot_title)
    plt.tight_layout()
    ut.save_stats_fig(df[[x_column, y_column, z_column]], fig=plt.gcf())


def plot_boxplot(
    df: pd.DataFrame, x_column: str, y_column: str, plot_title: str
) -> None:
    """
    Generates a box plot using Seaborn to show the distribution of `y_column` values across different categories in `x_column`.

    This plot is useful for answering questions like:
    - "How does the distribution of values differ between categories?"
    - "Are there any outliers or variations in the data?"

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_column (str): Column name for x-axis (categories).
        y_column (str): Column name for y-axis (numerical values).
        plot_title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x_column, y=y_column)
    plt.title(plot_title)
    plt.tight_layout()
    ut.save_stats_fig(df[[x_column, y_column]], fig=plt.gcf())


def plot_violin_plot(
    df: pd.DataFrame, x_column: str, y_column: str, plot_title: str
) -> None:
    """
    Creates a violin plot using Seaborn to visualize the distribution of `y_column` values across different categories in `x_column`.

    This plot is useful for answering questions like:
    - "How does the distribution of values within categories vary?"
    - "What are the density and spread of values within each category?"

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_column (str): Column name for x-axis (categories).
        y_column (str): Column name for y-axis (numerical values).
        plot_title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=x_column, y=y_column, inner="box", scale="width")
    plt.title(plot_title)
    plt.tight_layout()
    ut.save_stats_fig(df[[x_column, y_column]], fig=plt.gcf())


def plot_histogram(df: pd.DataFrame, column: str, plot_title: str) -> None:
    """
    Generates a histogram to show the distribution of the specified column using Seaborn.

    This plot is useful for answering questions like:
    - "What is the distribution of values in this dataset?"
    - "Are there any patterns or outliers in the data distribution?"

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name for histogram data.
        plot_title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(plot_title)
    plt.tight_layout()
    ut.save_stats_fig(df[[column]], fig=plt.gcf())


def plot_correlation_matrix(df: pd.DataFrame, plot_title: str) -> None:
    """
    Creates a heatmap for the correlation matrix of numeric columns using Seaborn.

    This plot is useful for answering questions like:
    - "What is the correlation between different variables in the dataset?"
    - "Are there any strong relationships or dependencies between variables?"

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        plot_title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="viridis")
    plt.title(plot_title)
    plt.tight_layout()
    ut.save_stats_fig(
        df, extra_stats={"correlation_matrix": corr.to_dict()}, fig=plt.gcf()
    )


def plot_pairplot(df: pd.DataFrame, plot_title: str) -> None:
    """
    Generates a pairplot to visualize pairwise relationships between numeric columns using Seaborn.

    This plot is useful for answering questions like:
    - "How do different variables in the dataset relate to each other?"
    - "What are the pairwise correlations and distributions?"

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        plot_title (str): Title of the plot.
    """
    sns.pairplot(df)
    plt.suptitle(plot_title, y=1.02)
    ut.save_stats_fig(df, fig=plt.gcf())


def plot_density_plot(
    df: pd.DataFrame, x_column: str, y_column: str, plot_title: str
) -> None:
    """
    Generates a density plot showing the distribution of values in a 2D space defined by `x_column` and `y_column` using Seaborn.

    This plot is useful for answering questions like:
    - "How are two variables distributed relative to each other?"
    - "Where are the data points concentrated in the 2D space?"

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_column (str): Column name for x-axis.
        y_column (str): Column name for y-axis.
        plot_title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=x_column, y=y_column)
    plt.title(plot_title)
    plt.tight_layout()
    ut.save_stats_fig(df[[x_column, y_column]], fig=plt.gcf())
