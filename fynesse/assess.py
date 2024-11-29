from .config import *


# assess.py
import osmnx as ox
import pandas as pd
import numpy as np
from collections import defaultdict
from math import cos, radians
from typing import List, Dict, Optional, Union, Tuple
import matplotlib.pyplot as plt

class AreaComparator:
    def compare_areas(self, 
                     group1_areas: List[str], 
                     group2_areas: List[str], 
                     keys: List[str],
                     radius_km: float = 6.0,
                     group1_label: str = "group1",
                     group2_label: str = "group2") -> Tuple[pd.DataFrame, Dict]:
        """
        Compare features between two groups of areas
        
        Args:
            group1_areas: List of area names for first group
            group2_areas: List of area names for second group
            keys: List of OSM keys to compare (e.g., ['amenity', 'building', 'landuse'])
            radius_km: Radius to search for features around each area center
            group1_label: Label for first group in output
            group2_label: Label for second group in output
            
        Returns:
            Tuple of (comparison DataFrame, raw counts dictionary)
        """
        def get_area_features(area_name: str) -> List[pd.Series]:
            """Get features for a single area within specified radius"""
            try:
                # Get area geometry and center point
                area_gdf = ox.geocode_to_gdf(area_name)
                bounds = area_gdf.total_bounds
                lat = (bounds[1] + bounds[3]) / 2
                lon = (bounds[0] + bounds[2]) / 2
                
                # Calculate bounding box
                lat_offset = radius_km / 111.0
                lon_offset = radius_km / (111.0 * cos(radians(lat)))
                
                north, south = lat + lat_offset, lat - lat_offset
                east, west = lon + lon_offset, lon - lon_offset
                
                # Get features for each key
                results = []
                for key in keys:
                    try:
                        tags = {key: True}
                        bbox = (west, south, east, north)
                        gdf = ox.features_from_bbox(bbox, tags)
                        results.append(gdf[key].value_counts() if key in gdf.columns else pd.Series())
                    except Exception as e:
                        print(f"Error getting {key} for {area_name}: {e}")
                        results.append(pd.Series())
                return results
                
            except Exception as e:
                print(f"Error processing area {area_name}: {e}")
                return [pd.Series() for _ in keys]

        # Initialize count dictionaries
        group1_counts = defaultdict(lambda: defaultdict(int))
        group2_counts = defaultdict(lambda: defaultdict(int))
        
        # Process first group
        print(f"Processing {group1_label} areas...")
        for area in group1_areas:
            counts = get_area_features(area)
            for ind, key in enumerate(keys):
                for k, count in counts[ind].items():
                    group1_counts[key][k] += count
        
        # Process second group
        print(f"Processing {group2_label} areas...")
        for area in group2_areas:
            counts = get_area_features(area)
            for ind, key in enumerate(keys):
                for k, count in counts[ind].items():
                    group2_counts[key][k] += count
        
        # Create comparison dataframe
        all_features = {}
        for key in keys:
            all_features.update({
                f"{key}_{k}": {
                    f'{group1_label}_avg': group1_counts[key][k]/len(group1_areas),
                    f'{group2_label}_avg': group2_counts[key][k]/len(group2_areas)
                }
                for k in set(group1_counts[key].keys()) | set(group2_counts[key].keys())
            })
        
        comparison_df = pd.DataFrame.from_dict(all_features, orient='index')
        comparison_df['difference'] = comparison_df[f'{group1_label}_avg'] - comparison_df[f'{group2_label}_avg']
        comparison_df['ratio'] = comparison_df[f'{group1_label}_avg'] / comparison_df[f'{group2_label}_avg'].replace(0, np.nan)
        
        # Store raw counts for potential further analysis
        raw_counts = {
            group1_label: dict(group1_counts),
            group2_label: dict(group2_counts)
        }
        
        return comparison_df.sort_values('difference', key=abs, ascending=False), raw_counts

    @staticmethod
    def analyze_feature_importance(comparison_df: pd.DataFrame, 
                                 min_difference: float = 0.1,
                                 min_ratio: Optional[float] = None) -> pd.DataFrame:
        """
        Analyze which features show significant differences
        
        Args:
            comparison_df: Output from compare_areas
            min_difference: Minimum absolute difference to consider significant
            min_ratio: Optional minimum ratio to consider significant
            
        Returns:
            DataFrame with significant features and their statistics
        """
        mask = abs(comparison_df['difference']) >= min_difference
        if min_ratio is not None:
            mask &= abs(comparison_df['ratio']) >= min_ratio
            
        significant_features = comparison_df[mask].copy()
        significant_features['abs_difference'] = abs(significant_features['difference'])
        return significant_features.sort_values('abs_difference', ascending=False)

def get_correlations_for_radius(conn, radius_km, features_dict, target_column='percentage'):
    """
    Get POI counts and correlations for a specific radius
    
    Args:
        conn: Database connection object
        radius_km (float): Radius in kilometers to search for POIs
        features_dict (dict): Dictionary of features to count, e.g.,
            {
                'amenity': ['university', 'college', 'school'],
                'building': ['university', 'school'],
                'landuse': ['education']
            }
        target_column (str): Name of the target column to correlate with
        
    Returns:
        tuple: (correlations dict, DataFrame with results)
    """
    coords_query = """
    SELECT n.total_residents, n.full_time_students, n.geography, c.LAT, c.LONG
    FROM nssec_fulltime_students_data n
    JOIN nssec_output_areas_coordinates c ON n.geography = c.OA21CD
    """
    base_data = pd.read_sql(coords_query, conn)
    
    results = []
    chunk_size = 1000
    
    for i in range(0, len(base_data), chunk_size):
        chunk = base_data.iloc[i:i+chunk_size]
        
        chunk_results = []
        for _, row in chunk.iterrows():
            radius_deg = radius_km / 111.0
            
            # Build CASE statements for each feature
            case_statements = []
            for key, values in features_dict.items():
                for value in values:
                    case_statements.append(
                        f"COUNT(CASE WHEN p.{key} = '{value}' THEN 1 END) AS {value}_{key}_count"
                    )
            
            query = f"""
            SELECT {', '.join(case_statements)}
            FROM osm_education_pois p
            WHERE ST_Contains(
                ST_Buffer(
                    Point({row['LONG']}, {row['LAT']}),
                    {radius_deg}
                ),
                p.geometry_col
            )
            """
            poi_counts = pd.read_sql(query, conn)
            chunk_results.append(poi_counts)
            
        if chunk_results:
            combined_counts = pd.concat(chunk_results, ignore_index=True)
            if len(combined_counts) == len(chunk):
                combined_chunk = pd.concat([
                    chunk.reset_index(drop=True), 
                    combined_counts.reset_index(drop=True)
                ], axis=1)
                results.append(combined_chunk)
        
        print(f"Processed chunk {i//chunk_size + 1}")
    
    if not results:
        return None
        
    final_df = pd.concat(results, ignore_index=True)
    final_df[target_column] = final_df['full_time_students'] / final_df['total_residents']
    
    # Calculate correlations for each feature
    correlations = {'radius_km': radius_km}
    
    # Individual correlations
    for key, values in features_dict.items():
        for value in values:
            col_name = f"{value}_{key}_count"
            corr_name = f"{value}_{key}_corr"
            correlations[corr_name] = final_df[col_name].corr(final_df[target_column])
    
    # Total correlation for each key type
    for key in features_dict.keys():
        cols = [f"{value}_{key}_count" for value in features_dict[key]]
        if cols:
            total_col = f"total_{key}_corr"
            correlations[total_col] = final_df[cols].sum(axis=1).corr(final_df[target_column])
    
    return correlations, final_df

def find_optimal_radius(conn, features_dict, radii=[0.5, 1, 2, 3, 4, 5, 7.5, 10], target_column='percentage'):
    """
    Find the optimal radius for feature correlations
    
    Args:
        conn: Database connection object
        features_dict (dict): Dictionary of features to count
        radii (list): List of radii to test in kilometers
        target_column (str): Name of the target column to correlate with
        
    Returns:
        tuple: (DataFrame of results, dict of optimal radii, dict of DataFrames)
    """
    correlation_results = []
    all_dfs = {}

    for radius in radii:
        print(f"\nTesting radius: {radius}km")
        corr, df = get_correlations_for_radius(conn, radius, features_dict, target_column)
        if corr:
            correlation_results.append(corr)
            all_dfs[radius] = df
            print(f"Correlations at {radius}km:", corr)

    corr_df = pd.DataFrame(correlation_results)
    
    # Find optimal radius for each feature
    optimal_radii = {}
    corr_columns = [col for col in corr_df.columns if col.endswith('_corr')]
    
    for col in corr_columns:
        best_radius = corr_df.loc[corr_df[col].abs().idxmax(), 'radius_km']
        best_corr = corr_df[col].abs().max()
        optimal_radii[col] = {'radius': best_radius, 'correlation': best_corr}
    
    return corr_df, optimal_radii, all_dfs

def plot_radius_correlations(corr_df, feature_groups=None):
    """
    Plot correlation results for different radii
    
    Args:
        corr_df (DataFrame): DataFrame containing correlation results
        feature_groups (dict, optional): Dictionary mapping feature groups to plot together
            e.g., {'education': ['university_amenity_corr', 'college_amenity_corr']}
    """
    if feature_groups is None:
        # Plot all correlation columns
        corr_columns = [col for col in corr_df.columns if col.endswith('_corr')]
        plt.figure(figsize=(12, 6))
        for col in corr_columns:
            plt.plot(corr_df['radius_km'], corr_df[col], label=col.replace('_corr', ''))
    else:
        # Plot feature groups separately
        for group_name, features in feature_groups.items():
            plt.figure(figsize=(12, 6))
            for feature in features:
                plt.plot(corr_df['radius_km'], corr_df[feature], 
                        label=feature.replace('_corr', ''))
            plt.title(f'{group_name} Correlations vs. Radius')
            plt.xlabel('Radius (km)')
            plt.ylabel(f'Correlation with Student Percentage')
            plt.legend()
            plt.grid(True)
            plt.show()