# This file contains code for suporting addressing questions in the data
import osmnx as ox
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns


def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    pois_counts = {}

    for tag in tags:
        if not tags[tag]:
            continue
        pois = ox.geometries_from_bbox(latitude + distance_km/(2*111), latitude - distance_km/(
            2*111), longitude + distance_km/(2*111), longitude - distance_km/(2*111), tags)
        pois_counts[tag] = pois[tag].notnull().sum()

    return pois_counts


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
