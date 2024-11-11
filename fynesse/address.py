# This file contains code for suporting addressing questions in the data
import osmnx as ox


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
