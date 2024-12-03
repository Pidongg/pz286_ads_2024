from .config import *
import osmnx as ox
import pandas as pd
import numpy as np
from collections import defaultdict
from math import cos, radians
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


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
                        results.append(gdf[key].value_counts(
                        ) if key in gdf.columns else pd.Series())
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
        comparison_df['difference'] = comparison_df[f'{
            group1_label}_avg'] - comparison_df[f'{group2_label}_avg']
        comparison_df['ratio'] = comparison_df[f'{
            group1_label}_avg'] / comparison_df[f'{group2_label}_avg'].replace(0, np.nan)

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
        significant_features['abs_difference'] = abs(
            significant_features['difference'])
        return significant_features.sort_values('abs_difference', ascending=False)


def get_correlations_for_radius(conn, radius_km, features_dict, table_name, target_column, table_name_2, 
                              geometry_col='geometry', include_distances=False):
    """
    Get POI counts, expected distances, and correlations for a specific radius

    Args:
        conn: Database connection object
        radius_km (float): Radius in kilometers to search for POIs
        features_dict (dict): Dictionary of features to count, e.g.,
            {
                'amenity': ['university', 'college', 'school'],
                'building': ['university', 'school'],
                'landuse': ['education']
            }
        table_name (str): Name of the POI table
        target_column (str): Name of the target column to correlate with
        table_name_2 (str): Name of the table containing target data
        geometry_col (str): Name of the geometry column
        include_distances (bool): Whether to include expected distance features

    Returns:
        tuple: (correlations dict, DataFrame with results)
    """
    coords_query = f"""
    SELECT n.total_residents, n.{target_column}, n.geography, c.LAT, c.LONG
    FROM {table_name_2} n
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
                    # Count-based features
                    case_statements.append(
                        f"COUNT(CASE WHEN p.{key} = '{value}' THEN 1 END) AS {value}_{key}_count"
                    )
                    
                    if include_distances:
                        # Expected distance
                        case_statements.append(f"""
                            AVG(
                                CASE 
                                    WHEN p.{key} = '{value}' THEN 
                                        ST_Distance_Sphere(
                                            Point({row['LONG']}, {row['LAT']}),
                                            ST_Centroid(p.{geometry_col})
                                        )/1000
                                END
                            ) AS {value}_{key}_expected_distance
                        """)

            query = f"""
            SELECT {', '.join(case_statements)}
            FROM {table_name} p
            WHERE ST_Contains(
                ST_Buffer(
                    Point({row['LONG']}, {row['LAT']}),
                    {radius_deg}
                ),
                ST_Centroid(p.{geometry_col})
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

    if not results:
        return None

    final_df = pd.concat(results, ignore_index=True)
    final_df['percentage'] = final_df[target_column] / final_df['total_residents']

    # Calculate correlations for each feature
    correlations = {'radius_km': radius_km}

    # Individual correlations
    for key, values in features_dict.items():
        for value in values:
            # Count correlations
            count_col = f"{value}_{key}_count"
            count_corr = f"{value}_{key}_count_corr"
            correlations[count_corr] = final_df[count_col].corr(final_df['percentage'])
            
            if include_distances:
                # Distance correlations
                dist_col = f"{value}_{key}_expected_distance"
                dist_corr = f"{value}_{key}_distance_corr"
                correlations[dist_corr] = final_df[dist_col].corr(final_df['percentage'])

    # Total correlation for each key type
    for key in features_dict.keys():
        # Total counts correlation
        count_cols = [f"{value}_{key}_count" for value in features_dict[key]]
        if count_cols:
            total_count_corr = f"total_{key}_count_corr"
            correlations[total_count_corr] = final_df[count_cols].sum(
                axis=1).corr(final_df['percentage'])
            
        if include_distances:
            # Total distance correlation
            dist_cols = [f"{value}_{key}_expected_distance" for value in features_dict[key]]
            if dist_cols:
                total_dist_corr = f"total_{key}_distance_corr"
                correlations[total_dist_corr] = final_df[dist_cols].mean(
                    axis=1).corr(final_df['percentage'])

    return correlations, final_df


def find_optimal_radius(conn, features_dict, table_name, target_column, table_name_2, radii=[0.5, 1, 2, 3, 4, 5, 7.5, 10], geometry_col='geometry', include_distances=False):
    """
    Find the optimal radius for feature correlations

    Args:
        conn: Database connection object
        features_dict (dict): Dictionary of features to count
        radii (list): List of radii to test in kilometers
        target_column (str): Name of the target column to correlate with
        table_name_2 (str): Name of the census table

    Returns:
        tuple: (DataFrame of results, dict of optimal radii, dict of DataFrames)
    """
    correlation_results = []
    all_dfs = {}

    for radius in radii:
        print(f"\nTesting radius: {radius}km")
        corr, df = get_correlations_for_radius(
            conn, radius, features_dict, table_name, target_column, table_name_2, geometry_col, include_distances)
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
    Plot correlation results for different radii with improved visualization

    Args:
        corr_df (DataFrame): DataFrame containing correlation results
        feature_groups (dict, optional): Dictionary mapping feature groups to plot together
            e.g., {'community': ['university_amenity', 'social_facility_amenity', 'community_centre_amenity']}
    """
    plt.figure(figsize=(12, 6))

    # Set up the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Radius (km)')
    plt.ylabel('Correlation with Percentage')
    plt.title('Community Features Correlations vs. Radius')

    # Plot each feature with distinct markers and colors
    markers = ['o-', 's-', '^-']  # Different markers for different features
    for i, (feature_name, feature_cols) in enumerate(feature_groups.items()):
        for col in feature_cols:
            plt.plot(corr_df['radius_km'],
                     corr_df[col],
                     markers[i % len(markers)],
                     label=col,
                     linewidth=2,
                     markersize=6)

    # Customize the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    return plt

def transform_df(df, new_tags):
    for tag in new_tags['amenity']:
        df['is_{}'.format(tag)] = df['amenity'] == tag
    merged_df = df.groupby('OA21CD')[f'is_{new_tags["amenity"][0]}'].sum().rename(
        f'is_{new_tags["amenity"][0]}').to_frame()
    print(merged_df)
    # Merge the remaining grouped series one by one
    for tag in new_tags['amenity'][1:]:
        grouped_series = df.groupby(
            'OA21CD')[f'is_{tag}'].sum().rename(f'is_{tag}')
        merged_df = pd.merge(merged_df, grouped_series,
                             on='OA21CD', how='inner')
    merged_df['poi_count'] = merged_df.filter(like='is_').sum(axis=1)
    return merged_df


def plot_correlations(df: pd.DataFrame, x_col: str, y_col: str, kind: str = 'scatter', **kwargs) -> None:
    """
    Plot correlations between two columns

    Args:
        df (pd.DataFrame): DataFrame containing the data
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        kind (str): Type of plot ('scatter', 'line', etc.)
        **kwargs: Additional arguments to pass to plot function
    """
    try:
        plot = df.plot(x=x_col, y=y_col, kind=kind, **kwargs)
        plt.xlabel(x_col)
        plt.ylabel(y_col)

        # Calculate and display correlation
        correlation = df[x_col].corr(df[y_col])
        plt.title(f'Correlation: {correlation:.3f}')

        return plot
    except Exception as e:
        print(f"Error plotting correlations: {e}")


def plot_correlations_normalized(df, feature_name, target='percentage'):
    """
    Plot correlation between a normalized feature (per capita) and target variable.

    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe containing the data
    feature_name : str
        Name of the feature column to analyze
    target : str
        Name of the target column (default='percentage')
    """
    # Calculate normalized feature
    normalized_values = df[feature_name] / df['total_residents']

    # Create plot
    plt.figure(figsize=(10, 6))

    # Plot scatter with transparency
    plt.scatter(normalized_values, df[target],
                alpha=0.3, color='blue', label='Data points')

    # Calculate and display correlation
    correlation = normalized_values.corr(df[target])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=plt.gca().transAxes)

    plt.xlabel(f'{feature_name} per capita')
    plt.ylabel(target)
    plt.title(f'Correlation between {feature_name} per capita and {target}')
    plt.legend()

    plt.show()


def plot_correlation_binned(df, feature_name, target='percentage', bins=30, suffix=''):
    """
    Plot correlation between a feature and target, showing average target value per bin,
    with labeled turning points.

    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
    feature_name : str
        Name of the feature column to analyze
    target : str
        Name of the target column (default='percentage')
    bins : int
        Number of bins to use (default=30)
    """
    # Create bins and calculate mean target value for each bin
    feature_name = feature_name + suffix
    df_grouped = df.groupby(pd.qcut(df[feature_name], bins, duplicates='drop'))[
        target].agg(['mean', 'std', 'count'])
    df_grouped['bin_center'] = df_grouped.index.map(lambda x: x.mid)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot original scatter (with alpha for better visibility)
    plt.scatter(df[feature_name], df[target], alpha=0.1,
                color='lightgray', label='Raw data')

    # Plot mean values
    plt.scatter(df_grouped['bin_center'], df_grouped['mean'],
                color='red', s=100, label='Bin average')

    # Add error bars (±1 standard deviation)
    plt.errorbar(df_grouped['bin_center'], df_grouped['mean'],
                 yerr=df_grouped['std'], color='red', fmt='none', alpha=0.5)

    # Connect average points with a line
    plt.plot(df_grouped['bin_center'], df_grouped['mean'],
             color='red', linestyle='--', alpha=0.5)

    # Find and label turning points
    means = df_grouped['mean'].values
    centers = df_grouped['bin_center'].values

    # Calculate differences between consecutive points
    diffs = np.diff(means)

    # Find where the slope changes significantly (turning points)
    threshold = np.std(diffs) * 1.5  # Adjust threshold as needed
    turning_points = []

    for i in range(1, len(diffs)-1):
        if (abs(diffs[i] - diffs[i-1]) > threshold):
            turning_points.append(i)

    # Label turning points
    for tp in turning_points:
        plt.annotate(f'({centers[tp]:.1f}, {means[tp]:.3f})',
                     xy=(centers[tp], means[tp]),
                     xytext=(10, 10),
                     textcoords='offset points',
                     ha='left',
                     va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5',
                               fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.xlabel(feature_name)
    plt.ylabel(f'Average {target}')
    plt.title(f'Binned correlation between {feature_name} and {target}')
    plt.legend()

    # Print correlation coefficient
    correlation = df[feature_name].corr(df[target])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=plt.gca().transAxes)

    plt.show()

    return df_grouped


def calculate_correlations(df: pd.DataFrame,
                           feature_cols: List[str],
                           target_col: str,
                           method: str = 'pearson') -> pd.Series:
    """
    Calculate correlations between multiple features and a target

    Args:
        df (pd.DataFrame): DataFrame containing the data
        feature_cols (List[str]): List of feature column names
        target_col (str): Target column name
        method (str): Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        pd.Series: Series containing correlations for each feature
    """
    try:
        correlations = {}
        for col in feature_cols:
            corr = df[col].corr(df[target_col], method=method)
            correlations[col] = corr

        return pd.Series(correlations).sort_values(ascending=False)
    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return pd.Series()


def plot_feature_importances(correlations: pd.Series,
                             title: str = 'Feature Correlations',
                             figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot feature correlations as a bar chart

    Args:
        correlations (pd.Series): Series of correlations
        title (str): Plot title
        figsize (tuple): Figure size (width, height)
    """
    try:
        plt.figure(figsize=figsize)
        correlations.plot(kind='bar')
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel('Correlation')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting feature importances: {e}")


def analyze_features(df: pd.DataFrame,
                     features_dict: Dict[str, List[str]],
                     target_col: str,
                     plot: bool = True) -> Dict[str, pd.Series]:
    """
    Analyze multiple feature groups and their correlations with a target

    Args:
        df (pd.DataFrame): DataFrame containing the data
        features_dict (dict): Dictionary mapping feature groups to column lists
        target_col (str): Target column name
        plot (bool): Whether to plot results

    Returns:
        dict: Dictionary of correlation results by feature group
    """
    try:
        results = {}
        for group_name, features in features_dict.items():
            # Calculate correlations
            correlations = calculate_correlations(df, features, target_col)
            results[group_name] = correlations

            if plot:
                # Plot correlations
                plt.figure(figsize=(10, 6))
                plot_feature_importances(
                    correlations,
                    title=f'{group_name} Correlations with {target_col}'
                )

        return results
    except Exception as e:
        print(f"Error analyzing features: {e}")
        return {}


def compare_prices_multi(df, column_name):
    """
    Calculate and visualize average prices for multiple categories
    """
    # Calculate averages for each unique value
    for val in df[column_name].unique():
        df[f'{val}_avg_price'] = df[df[column_name] == val].groupby(
            'date_of_transfer')['price'].transform('mean')

    # Calculate overall statistics for each category
    stats = {}
    for val in df[column_name].unique():
        stats[val] = df[f'{val}_avg_price'].mean()

    # Create box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=column_name, y='price')
    plt.title(f'Price Distribution by {column_name}')
    plt.ylabel('Price (£)')

    # Print summary statistics
    print(f"\nAverage prices by {column_name}:")
    for val, avg in stats.items():
        print(f"{val}: £{avg:,.2f}")


def remove_outliers(df, columns, threshold=2):
    """
    Remove outliers from a DataFrame based on specified columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list
        List of column names to check for outliers.
    threshold : float, optional
        Number of standard deviations to use as the cutoff for outliers (default is 2).

    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed.
    """
    # Calculate mean and standard deviation for each column
    means = df[columns].mean()
    stds = df[columns].std()

    # Filter out rows where any column value is more than `threshold` standard deviations from the mean
    mask = (df[columns] - means).abs() <= threshold * stds
    filtered_df = df[mask.all(axis=1)]

    return filtered_df


def plot_nth_prices(df, n=100, property_type=None, order='highest'):
    """
    Plot the nth highest or lowest prices from the dataset as a line graph

    Parameters:
    df: DataFrame containing price data
    n: Number of prices to show (default 100)
    property_type: Optional filter for specific property type
    order: 'highest' or 'lowest' to determine which end of the price range to plot
    """
    # Input validation
    if order not in ['highest', 'lowest']:
        raise ValueError("order must be either 'highest' or 'lowest'")

    # Filter by property type if specified
    if property_type:
        df = df[df['property_type'] == property_type]

    # Sort prices and get top/bottom n
    if order == 'highest':
        selected_prices = df.nlargest(n, 'price')
        title_prefix = 'Highest'
    else:
        selected_prices = df.nsmallest(n, 'price')
        title_prefix = 'Lowest'

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(n), selected_prices['price'], marker='o')

    # Customize the plot
    plt.title(f'{title_prefix} {n} Prices' +
              (f' for Property Type {property_type}' if property_type else ''))
    plt.xlabel('Rank')
    plt.ylabel('Price (£)')

    # Format y-axis with comma separator for thousands
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show fewer x-axis labels (only show every 10th rank)
    step = max(n // 10, 1)  # Show ~10 labels, but at least 1 step
    plt.xticks(range(0, n, step), range(1, n+1, step))

    plt.show()
