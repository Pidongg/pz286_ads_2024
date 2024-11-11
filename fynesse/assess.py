from .config import *
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import osmnx as ox

from . import access


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError


def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


def plot_osm_data(data_frames, labels, north, south, east, west, colours=None):
    fig, ax = plt.subplots()
    graph = ox.graph_from_bbox(north, south, east, west)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)
    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if colours is None:
        colours = ["blue"]*len(data_frames)
    # Plot tourist places
    for i, data_frame in enumerate(data_frames):
        data_frame.plot(
            ax=ax, color=colours[i], alpha=1, markersize=50, label=labels[i])
    plt.legend()
    plt.tight_layout()


def mds_visualisation(distance_matrix, title):
    mds = MDS(n_components=2, dissimilarity='precomputed')  # 2D visualization
    pos = mds.fit_transform(distance_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size as needed

    ax.scatter(pos[:, 0], pos[:, 1])  # Plot the points
    for i, label in enumerate(distance_matrix.index.to_list()):
        ax.annotate(label, (pos[i, 0], pos[i, 1]),
                    xytext=(5, 5),  # Offset the text slightly
                    textcoords='offset points',
                    ha='center', va='bottom')
    plt.title(title)
    plt.show()


def heatmap_plot(data_frame, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_frame, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(title)
    plt.show()
