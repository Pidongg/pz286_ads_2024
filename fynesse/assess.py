from branca.colormap import LinearColormap
import folium
from .config import *
import osmnx as ox
import pandas as pd
import numpy as np
from collections import defaultdict
from math import cos, radians
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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
                        f"COUNT(CASE WHEN p.{key} = '{value}' THEN 1 END) AS {
                            value}_{key}_count"
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
    final_df['percentage'] = final_df[target_column] / \
        final_df['total_residents']

    # Calculate correlations for each feature
    correlations = {'radius_km': radius_km}

    # Individual correlations
    for key, values in features_dict.items():
        for value in values:
            # Count correlations
            count_col = f"{value}_{key}_count"
            count_corr = f"{value}_{key}_count_corr"
            correlations[count_corr] = final_df[count_col].corr(
                final_df['percentage'])

            if include_distances:
                # Distance correlations
                dist_col = f"{value}_{key}_expected_distance"
                dist_corr = f"{value}_{key}_distance_corr"
                correlations[dist_corr] = final_df[dist_col].corr(
                    final_df['percentage'])

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
            dist_cols = [f"{value}_{
                key}_expected_distance" for value in features_dict[key]]
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


def calculate_category_sums_and_correlations(df, features_dict, target='price', suffix=''):
    """
    Calculate the sum of features within each category and their correlation with the target variable
    """
    category_sums = {}
    correlations = {}

    for category, feature_list in features_dict.items():
        # Calculate the sum of features within the category
        sum_column_name = f'{category}_sum'
        df[sum_column_name] = df[[x+suffix for x in feature_list]].sum(axis=1)

        # Store the sum in the dictionary
        category_sums[category] = sum_column_name

        # Calculate correlation with the target variable
        correlation = df[sum_column_name].corr(df[target])
        correlations[category] = correlation

    print("\nCategory Sums:")
    print(category_sums)

    print("\nCategory Correlations with Price:")
    for category, correlation in correlations.items():
        print(f"{category}: {correlation:.4f}")


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
    Calculate, visualize, and analyze price distributions for multiple categories
    including variance analysis and statistical measures

    Args:
        df: DataFrame containing price data
        column_name: Column to group by for comparison
    """
    # Calculate statistics for each category
    stats = {}
    for val in df[column_name].unique():
        group_prices = df[df[column_name] == val]['price']
        stats[val] = {
            'mean': group_prices.mean(),
            'median': group_prices.median(),
            'std': group_prices.std(),
            'var': group_prices.var(),
            'cv': group_prices.std() / group_prices.mean(),  # Coefficient of variation
            'iqr': group_prices.quantile(0.75) - group_prices.quantile(0.25),
            'count': len(group_prices)
        }

    # Create subplots for comprehensive visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Box plot
    sns.boxplot(data=df, x=column_name, y='price', ax=ax1)
    ax1.set_title(f'Price Distribution by {column_name}')
    ax1.set_ylabel('Price (£)')

    # Violin plot to show distribution shape
    sns.violinplot(data=df, x=column_name, y='price', ax=ax2)
    ax2.set_title(f'Price Distribution Shape by {column_name}')
    ax2.set_ylabel('Price (£)')

    plt.tight_layout()
    plt.show()

    # Print detailed statistical summary
    print(f"\nDetailed Price Analysis by {column_name}:")
    print("-" * 80)
    print(f"{'Category':<15} {'Mean':>12} {'Median':>12} {
          'Std Dev':>12} {'CV':>8} {'Count':>8}")
    print("-" * 80)

    for val, metrics in stats.items():
        print(f"{str(val):<15} "
              f"£{metrics['mean']:>11,.0f} "
              f"£{metrics['median']:>11,.0f} "
              f"£{metrics['std']:>11,.0f} "
              f"{metrics['cv']:>8.2f} "
              f"{metrics['count']:>8,d}")

    # Variance analysis
    print("\nVariance Analysis:")
    print("-" * 80)
    total_variance = df['price'].var()
    between_group_variance = np.sum([
        stats[val]['count'] *
        (stats[val]['mean'] - df['price'].mean())**2
        for val in stats
    ]) / len(df)
    within_group_variance = np.sum([
        (stats[val]['count'] - 1) * stats[val]['var']
        for val in stats
    ]) / (len(df) - 1)

    print(f"Total Variance:          £{total_variance:,.0f}")
    print(f"Between-group Variance:  £{between_group_variance:,.0f}")
    print(f"Within-group Variance:   £{within_group_variance:,.0f}")
    print(
        f"Variance Ratio (F-stat): {between_group_variance/within_group_variance:.2f}")


def calculate_price_boundaries(reference_df, price_col):
    """
    Calculate price boundaries using a reference dataset

    Parameters:
    -----------
    reference_df : pandas.DataFrame
        Reference dataset to use for calculating boundaries
    price_col : str
        Name of the price column

    Returns:
    --------
    tuple : (lower_bound, upper_bound, mean, std)
    """
    log_price = np.log(reference_df[price_col])
    mean_log_price = log_price.mean()
    std_log_price = log_price.std()
    cutoff_high = mean_log_price + 2*std_log_price
    cutoff_low = mean_log_price - 2*std_log_price
    return cutoff_low, cutoff_high, mean_log_price, std_log_price


def remove_outliers(df_full_match, columns_to_check, df_2011=None, df_2021=None):
    """
    Remove outliers from price columns using reference datasets if provided

    Parameters:
    -----------
    df_full_match : pandas.DataFrame
        DataFrame containing matched properties
    columns_to_check : list
        List of price columns to check for outliers
    df_2011 : pandas.DataFrame, optional
        Reference dataset for 2011 prices
    df_2021 : pandas.DataFrame, optional
        Reference dataset for 2021 prices

    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    cleaned_df = df_full_match.copy()

    for col in columns_to_check:
        # Determine which reference dataset to use
        if col == 'price_2011' and df_2011 is not None:
            ref_df = df_2011
            ref_col = 'price'
        elif col == 'price_2021' and df_2021 is not None:
            ref_df = df_2021
            ref_col = 'price'
        else:
            ref_df = df_full_match
            ref_col = col

        # Calculate boundaries
        cutoff_low, cutoff_high, _, _ = calculate_price_boundaries(
            ref_df, ref_col)

        # Remove outliers
        log_price = np.log(cleaned_df[col])
        cleaned_df = cleaned_df[
            (log_price >= cutoff_low) &
            (log_price <= cutoff_high)
        ]

    return cleaned_df


def analyze_log_price_outliers(df_full_match, cleaned_df, df_2011=None, df_2021=None):
    """
    Analyze price outliers using reference datasets if provided

    Parameters:
    -----------
    df_full_match : pandas.DataFrame
        Original DataFrame with all matched properties
    cleaned_df : pandas.DataFrame
        DataFrame after outlier removal
    df_2011 : pandas.DataFrame, optional
        Reference dataset for 2011 prices
    df_2021 : pandas.DataFrame, optional
        Reference dataset for 2021 prices
    """
    price_cols = ['price_2011', 'price_2021']

    for price_col in price_cols:
        print(f"\nAnalyzing {price_col}...")

        # Determine which reference dataset to use
        if price_col == 'price_2011' and df_2011 is not None:
            ref_df = df_2011
            ref_col = 'price'
        elif price_col == 'price_2021' and df_2021 is not None:
            ref_df = df_2021
            ref_col = 'price'
        else:
            ref_df = df_full_match
            ref_col = price_col

        # Calculate boundaries using reference dataset
        cutoff_low, cutoff_high, mean_log_price, std_log_price = calculate_price_boundaries(
            ref_df, ref_col)

        # Add log prices for plotting
        df_full_match[f'log_{price_col}'] = np.log(df_full_match[price_col])
        cleaned_df[f'log_{price_col}'] = np.log(cleaned_df[price_col])

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # Histogram with outlier boundaries
        sns.histplot(data=df_full_match, x=f'log_{price_col}', bins=100, ax=ax1,
                     alpha=0.5, label='Original')
        sns.histplot(data=cleaned_df, x=f'log_{price_col}', bins=100, ax=ax1,
                     alpha=0.5, label='After cleaning')
        ax1.axvline(cutoff_high, color='r',
                    linestyle='--', label='Upper bound')
        ax1.axvline(cutoff_low, color='r', linestyle='--', label='Lower bound')
        ax1.axvline(mean_log_price, color='g', linestyle='--', label='Mean')
        ax1.set_title(f'Distribution of Log {
                      price_col} Before and After Outlier Removal')
        ax1.legend()

        # Analyze outliers by type and price range
        outliers_analysis = {}
        property_types = sorted(df_full_match['property_type'].unique())

        for ptype in property_types:
            original_data = df_full_match[df_full_match['property_type'] == ptype]
            cleaned_data = cleaned_df[cleaned_df['property_type'] == ptype]

            # Count outliers
            low_outliers = len(
                original_data[original_data[f'log_{price_col}'] < cutoff_low])
            high_outliers = len(
                original_data[original_data[f'log_{price_col}'] > cutoff_high])
            normal = len(cleaned_data)

            outliers_analysis[ptype] = {
                'Low outliers': low_outliers,
                'Normal range': normal,
                'High outliers': high_outliers
            }

        # Create stacked bar chart
        df_plot = pd.DataFrame(outliers_analysis).T
        df_plot_pct = df_plot.div(df_plot.sum(axis=1), axis=0) * 100

        # Plot stacked bars
        bottom_vals = np.zeros(len(property_types))
        # Red for low, Blue for normal, Pink for high
        colors = ['#ff9999', '#66b3ff', '#ff99cc']

        for i, col in enumerate(['Low outliers', 'Normal range', 'High outliers']):
            ax2.bar(property_types, df_plot_pct[col], bottom=bottom_vals,
                    label=col, color=colors[i])
            # Add percentage labels for all segments
            for j, v in enumerate(df_plot_pct[col]):
                ax2.text(j, bottom_vals[j] + v/2, f'{v:.1f}%',
                         ha='center', va='center')
            bottom_vals += df_plot_pct[col]

        ax2.set_title(f'Distribution of {price_col} Ranges by Property Type')
        ax2.set_xlabel('Property Type')
        ax2.set_ylabel('Percentage')
        ax2.legend()

        # Print price boundaries
        print(f"\nPrice Boundaries for {price_col}:")
        print(f"Lower bound: £{np.exp(cutoff_low):,.0f}")
        print(f"Upper bound: £{np.exp(cutoff_high):,.0f}")

        plt.tight_layout()
        plt.show()

        # Print absolute numbers
        print(f"\nAbsolute Numbers by Property Type for {price_col}:")
        print(df_plot)


def analyze_log_price_outliers(merged_df, cleaned_df, price=['price']):
    """
    Analyze the price outliers that were removed, using log transformation

    Parameters:
    -----------
    merged_df : pandas.DataFrame
        Original DataFrame with all data
    cleaned_df : pandas.DataFrame
        DataFrame after outlier removal
    price : list of str, default=['price']
        List of price column names to analyze
    """
    for price_col in price:
        print(f"\nAnalyzing {price_col}...")

        # Add log prices
        merged_df[f'log_{price_col}'] = np.log(merged_df[price_col])
        cleaned_df[f'log_{price_col}'] = np.log(cleaned_df[price_col])

        # Calculate statistics
        mean_log_price = merged_df[f'log_{price_col}'].mean()
        std_log_price = merged_df[f'log_{price_col}'].std()
        cutoff_high = mean_log_price + 2*std_log_price
        cutoff_low = mean_log_price - 2*std_log_price

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # Histogram with outlier boundaries
        sns.histplot(data=merged_df, x=f'log_{price_col}', bins=100, ax=ax1,
                     alpha=0.5, label='Original')
        sns.histplot(data=cleaned_df, x=f'log_{price_col}', bins=100, ax=ax1,
                     alpha=0.5, label='After cleaning')
        ax1.axvline(cutoff_high, color='r',
                    linestyle='--', label='Upper bound')
        ax1.axvline(cutoff_low, color='r', linestyle='--', label='Lower bound')
        ax1.axvline(mean_log_price, color='g', linestyle='--', label='Mean')
        ax1.set_title(f'Distribution of Log {
                      price_col} Before and After Outlier Removal')
        ax1.legend()

        # Analyze outliers by type and price range
        outliers_analysis = {}
        property_types = sorted(merged_df['property_type'].unique())

        for ptype in property_types:
            original_data = merged_df[merged_df['property_type'] == ptype]
            cleaned_data = cleaned_df[cleaned_df['property_type'] == ptype]

            # Count total
            total_original = len(original_data)

            # Count outliers
            low_outliers = len(
                original_data[original_data[f'log_{price_col}'] < cutoff_low])
            high_outliers = len(
                original_data[original_data[f'log_{price_col}'] > cutoff_high])
            normal = len(cleaned_data)

            outliers_analysis[ptype] = {
                'Low outliers': low_outliers,
                'Normal range': normal,
                'High outliers': high_outliers
            }

        # Create stacked bar chart
        df_plot = pd.DataFrame(outliers_analysis).T
        df_plot_pct = df_plot.div(df_plot.sum(axis=1), axis=0) * 100

        # Plot stacked bars
        bottom_vals = np.zeros(len(property_types))
        # Red for low, Blue for normal, Pink for high
        colors = ['#ff9999', '#66b3ff', '#ff99cc']

        for i, col in enumerate(['Low outliers', 'Normal range', 'High outliers']):
            ax2.bar(property_types, df_plot_pct[col], bottom=bottom_vals,
                    label=col, color=colors[i])
            # Add percentage labels for all segments
            for j, v in enumerate(df_plot_pct[col]):
                ax2.text(j, bottom_vals[j] + v/2, f'{v:.1f}%',
                         ha='center', va='center')
            bottom_vals += df_plot_pct[col]

        ax2.set_title(f'Distribution of {price_col} Ranges by Property Type')
        ax2.set_xlabel('Property Type')
        ax2.set_ylabel('Percentage')
        ax2.legend()

        # Print price boundaries
        print(f"\nPrice Boundaries for {price_col}:")
        print(f"Lower bound: £{np.exp(cutoff_low):,.0f}")
        print(f"Upper bound: £{np.exp(cutoff_high):,.0f}")

        plt.tight_layout()
        plt.show()

        # Print absolute numbers
        print(f"\nAbsolute Numbers by Property Type for {price_col}:")
        print(df_plot)


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


def analyze_property_type_distribution(df):
    """
    Visualize the distribution of property types in the transactions
    """
    # Count transactions
    type_counts = df['property_type'].value_counts()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot
    sns.barplot(x=type_counts.index, y=type_counts.values, ax=ax1)
    ax1.set_title('Number of Transactions by Property Type')
    ax1.set_xlabel('Property Type')
    ax1.set_ylabel('Number of Transactions')

    # Add value labels on bars
    for i, v in enumerate(type_counts.values):
        ax1.text(i, v, f'{v:,}', ha='center', va='bottom')

    # Pie chart
    ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    ax2.set_title('Proportion of Property Types')

    plt.tight_layout()

    # Print summary
    print("\nTransaction Counts by Property Type:")
    print("------------------------------------")
    for prop_type, count in type_counts.items():
        print(f"{prop_type} ({get_property_type_name(prop_type)}): {
              count:,} ({count/len(df)*100:.1f}%)")


def get_property_type_name(code):
    """Convert property type code to full name"""
    types = {
        'D': 'Detached',
        'S': 'Semi-detached',
        'T': 'Terraced',
        'F': 'Flat/Maisonette'
    }
    return types.get(code, code)


def analyze_price_changes(merged_df_2011, merged_df, inflation_rate=1):
    """
    Analyze overall price distribution differences between 2011 and 2021.

    Parameters:
    -----------
    merged_df_2011: DataFrame containing 2011 data
    merged_df: DataFrame containing 2021 data
    inflation_rate: Rate to adjust 2021 prices (default=1)
    """
    # Create figure with two subplots if inflation adjustment is needed
    if inflation_rate != 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        # Left plot: Without inflation adjustment
        merged_df['log_price_nominal'] = np.log(merged_df['price'])
        merged_df_2011['log_price'] = np.log(merged_df_2011['price'])

        sns.kdeplot(data=merged_df_2011['log_price'], color='blue', label='2011',
                    fill=True, alpha=0.3, ax=ax1)
        sns.kdeplot(data=merged_df['log_price_nominal'], color='red', label='2021 (Nominal)',
                    fill=True, alpha=0.3, ax=ax1)
        ax1.set_xlabel('Log Price')
        ax1.set_ylabel('Density')
        ax1.set_title(
            'Housing Price Distribution - Nominal Prices (Log Scale)')
        ax1.legend()

        # Add price labels on second x-axis
        ax1_2 = ax1.twiny()
        ax1_2.set_xlim(ax1.get_xlim())
        tick_locations = ax1.get_xticks()
        ax1_2.set_xticks(tick_locations)
        ax1_2.set_xticklabels(
            [f'£{np.exp(x):,.0f}' for x in tick_locations], rotation=45)

        # Right plot: With inflation adjustment
        merged_df['price_2021_inflation_adjusted'] = merged_df['price'] / \
            inflation_rate
        merged_df['log_price_adjusted'] = np.log(
            merged_df['price_2021_inflation_adjusted'])

        sns.kdeplot(data=merged_df_2011['log_price'], color='blue', label='2011',
                    fill=True, alpha=0.3, ax=ax2)
        sns.kdeplot(data=merged_df['log_price_adjusted'], color='red', label='2021 (Inflation Adjusted)',
                    fill=True, alpha=0.3, ax=ax2)
        ax2.set_xlabel('Log Price')
        ax2.set_ylabel('Density')
        ax2.set_title(
            f'Housing Price Distribution - Inflation Adjusted (Log Scale)')
        ax2.legend()

        # Add price labels on second x-axis
        ax2_2 = ax2.twiny()
        ax2_2.set_xlim(ax2.get_xlim())
        tick_locations = ax2.get_xticks()
        ax2_2.set_xticks(tick_locations)
        ax2_2.set_xticklabels(
            [f'£{np.exp(x):,.0f}' for x in tick_locations], rotation=45)

    else:
        # Single plot if no inflation adjustment
        plt.figure(figsize=(10, 6))
        merged_df['price_2021_inflation_adjusted'] = merged_df['price']
        merged_df_2011['log_price'] = np.log(merged_df_2011['price'])
        merged_df['log_price'] = np.log(
            merged_df['price_2021_inflation_adjusted'])

        sns.kdeplot(data=merged_df_2011['log_price'], color='blue', label='2011',
                    fill=True, alpha=0.3)
        sns.kdeplot(data=merged_df['log_price'], color='red', label='2021',
                    fill=True, alpha=0.3)
        plt.xlabel('Log Price')
        plt.ylabel('Density')
        plt.title('Housing Price Distribution (Log Scale)')
        plt.legend()

        # Add price labels on second x-axis
        ax2 = plt.gca().twiny()
        ax2.set_xlim(plt.gca().get_xlim())
        tick_locations = plt.gca().get_xticks()
        ax2.set_xticks(tick_locations)
        ax2.set_xticklabels(
            [f'£{np.exp(x):,.0f}' for x in tick_locations], rotation=45)

    plt.tight_layout()
    plt.show()

    # Print price change statistics
    median_2011 = merged_df_2011['price'].median()
    median_2021_nominal = merged_df['price'].median()
    median_2021_adjusted = merged_df['price_2021_inflation_adjusted'].median()

    print("\nPrice Changes:")
    print(f"Median Price 2011: £{median_2011:,.2f}")

    if inflation_rate != 1:
        print("\nNominal Prices:")
        print(f"Median Price 2021: £{median_2021_nominal:,.2f}")
        print(f"Nominal Price Change: {
              ((median_2021_nominal / median_2011) - 1) * 100:.1f}%")

        print("\nInflation Adjusted Prices:")
        print(f"Median Price 2021 (Adjusted): £{median_2021_adjusted:,.2f}")
        print(f"Real Price Change: {
              ((median_2021_adjusted / median_2011) - 1) * 100:.1f}%")
        print(f"Inflation Rate: {(inflation_rate - 1) * 100:.1f}%")
    else:
        print(f"Median Price 2021: £{median_2021_nominal:,.2f}")
        print(f"Price Change: {
              ((median_2021_nominal / median_2011) - 1) * 100:.1f}%")


def analyze_transaction_counts(merged_df_2011, merged_df):
    """
    Analyze overall transaction count differences between 2011 and 2021.

    Parameters:
    -----------
    merged_df_2011: DataFrame containing 2011 data
    merged_df: DataFrame containing 2021 data
    """
    # Monthly transaction counts
    merged_df_2011['month'] = pd.to_datetime(
        merged_df_2011['date_of_transfer']).dt.month
    merged_df['month'] = pd.to_datetime(merged_df['date_of_transfer']).dt.month

    monthly_counts_2011 = merged_df_2011.groupby('month').size()
    monthly_counts_2021 = merged_df.groupby('month').size()

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot monthly transaction counts
    months = range(1, 13)
    ax1.plot(months, [monthly_counts_2011.get(m, 0)
             for m in months], 'b-', label='2011', marker='o')
    ax1.plot(months, [monthly_counts_2021.get(m, 0)
             for m in months], 'r-', label='2021', marker='o')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Transactions')
    ax1.set_title('Monthly Transaction Counts')
    ax1.set_xticks(months)
    ax1.legend()

    # Add percentage change labels for monthly counts
    for month in months:
        count_2011 = monthly_counts_2011.get(month, 0)
        count_2021 = monthly_counts_2021.get(month, 0)
        if count_2011 > 0:
            pct_change = ((count_2021 - count_2011) / count_2011 * 100)
            y_pos = max(count_2021, count_2011)
            ax1.text(month, y_pos * 1.05,
                     f'{pct_change:+.1f}%', ha='center', va='bottom', fontsize=8)

    # Plot total transaction counts
    total_counts = pd.Series({
        '2011': len(merged_df_2011),
        '2021': len(merged_df)
    })

    bars = ax2.bar(['2011', '2021'], total_counts,
                   color=['blue', 'red'], alpha=0.7)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Total Number of Transactions')
    ax2.set_title('Total Transaction Counts')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}', ha='center', va='bottom')

    # Add percentage change label
    pct_change = (
        (total_counts['2021'] - total_counts['2011']) / total_counts['2011'] * 100)
    ax2.text(0.5, max(total_counts) * 1.1,
             f'Change: {pct_change:+.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Print transaction count statistics
    print("\nTransaction Counts:")
    print(f"Total Transactions 2011: {total_counts['2011']:,}")
    print(f"Total Transactions 2021: {total_counts['2021']:,}")
    print(f"Transaction Count Change: {pct_change:.1f}%")

    # Monthly statistics
    print("\nMonthly Transaction Statistics:")
    print(f"2011 - Average: {monthly_counts_2011.mean():.1f}, Min: {
          monthly_counts_2011.min()}, Max: {monthly_counts_2011.max()}")
    print(f"2021 - Average: {monthly_counts_2021.mean():.1f}, Min: {
          monthly_counts_2021.min()}, Max: {monthly_counts_2021.max()}")


def analyze_property_type_changes(merged_df_2011, merged_df, inflation_rate=1):
    """
    Analyze price changes and transaction count changes by property type between 2011 and 2021.
    """
    # Adjust prices for inflation
    merged_df['price_2021_inflation_adjusted'] = merged_df['price'] / \
        inflation_rate

    # Create figure with subplots for median prices and transaction counts
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Calculate median prices by property type
    property_types = ['D', 'S', 'T', 'F']
    property_type_names = {'D': 'Detached',
                           'S': 'Semi-detached', 'T': 'Terraced', 'F': 'Flat'}
    medians_2011 = [merged_df_2011[merged_df_2011['property_type']
                                   == pt]['price'].median() for pt in property_types]
    medians_2021 = [merged_df[merged_df['property_type'] == pt]
                    ['price_2021_inflation_adjusted'].median() for pt in property_types]

    # Bar plot of median prices by property type
    x = np.arange(len(property_types))
    width = 0.35
    axes[0].bar(x - width/2, medians_2011, width,
                label='2011', color='blue', alpha=0.7)
    axes[0].bar(x + width/2, medians_2021, width,
                label='2021', color='red', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Detached', 'Semi-detached', 'Terraced', 'Flat'])
    axes[0].set_xlabel('Property Type')
    axes[0].set_ylabel('Median Price (£)')
    axes[0].set_yscale('log')
    axes[0].set_title(
        'Median House Prices by Property Type: 2011 vs 2021 (Log Scale)')
    axes[0].legend()

    # Add percentage change labels for median prices
    for i in range(len(property_types)):
        pct_change = (
            (medians_2021[i] - medians_2011[i]) / medians_2011[i] * 100)
        axes[0].text(i, max(medians_2021[i], medians_2021[i]) * 1.1,
                     f'+{pct_change:.1f}%', ha='center', va='bottom')

    # Calculate transaction counts by property type
    counts_2011 = [len(merged_df_2011[merged_df_2011['property_type'] == pt])
                   for pt in property_types]
    counts_2021 = [len(merged_df[merged_df['property_type'] == pt])
                   for pt in property_types]

    # Bar plot of transaction counts by property type
    axes[1].bar(x - width/2, counts_2011, width,
                label='2011', color='blue', alpha=0.7)
    axes[1].bar(x + width/2, counts_2021, width,
                label='2021', color='red', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Detached', 'Semi-detached', 'Terraced', 'Flat'])
    axes[1].set_xlabel('Property Type')
    axes[1].set_ylabel('Transaction Count')
    axes[1].set_title('Transaction Counts by Property Type: 2011 vs 2021')
    axes[1].legend()

    # Add percentage change labels for transaction counts
    for i in range(len(property_types)):
        pct_change = ((counts_2021[i] - counts_2011[i]) / counts_2011[i] * 100)
        axes[1].text(i, max(counts_2021[i], counts_2011[i]) * 1.1,
                     f'{pct_change:+.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Create separate scatter plots for each property type
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 15))
    axes2 = axes2.ravel()

    for idx, (pt_code, pt_name) in enumerate(property_type_names.items()):
        # Filter data for current property type
        pt_data_2011 = merged_df_2011[merged_df_2011['property_type'] == pt_code]
        pt_data_2021 = merged_df[merged_df['property_type'] == pt_code]

        # Align data by using a common index
        # Assuming 'id' is a common identifier
        pt_data_2011 = pt_data_2011.set_index('id')
        pt_data_2021 = pt_data_2021.set_index('id')
        common_index = pt_data_2011.index.intersection(pt_data_2021.index)
        pt_data_2011 = pt_data_2011.loc[common_index]
        pt_data_2021 = pt_data_2021.loc[common_index]

        # Create scatter plot
        axes2[idx].scatter(np.log(pt_data_2011['price']),
                           np.log(
                               pt_data_2021['price_2021_inflation_adjusted']),
                           alpha=0.5,
                           s=20)

        # Add no change line
        max_price = max(np.log(pt_data_2011['price'].max()),
                        np.log(pt_data_2021['price_2021_inflation_adjusted'].max()))
        axes2[idx].plot([0, max_price], [0, max_price],
                        'r--', label='No Change Line')

        # Add labels and title
        axes2[idx].set_title(f'log {pt_name} Prices: 2021 vs 2011')
        axes2[idx].set_xlabel('log 2011 Price (£)')
        axes2[idx].set_ylabel('log 2021 Price (£)')

        # Add correlation coefficient
        corr = np.corrcoef(np.log(pt_data_2011['price']),
                           np.log(pt_data_2021['price_2021_inflation_adjusted']))[0, 1]
        axes2[idx].text(0.05, 0.95, f'Correlation: {corr:.2f}',
                        transform=axes2[idx].transAxes,
                        bbox=dict(facecolor='white', alpha=0.8))

        # Set equal aspect ratio
        axes2[idx].set_aspect('equal')

    plt.tight_layout()
    plt.show()

    # Print summary statistics by property type
    print("\nPrice Change Statistics by Property Type:")
    for i, pt in enumerate(property_types):
        print(f"{pt}: Median 2011: £{medians_2011[i]:,.2f}, "
              f"Median 2021: £{medians_2021[i]:,.2f}, "
              f"Change: {((medians_2021[i] / medians_2011[i]) - 1) * 100:.1f}%")

    # Print transaction count changes
    print("\nTransaction Count Changes by Property Type:")
    for i, pt in enumerate(property_types):
        print(f"{pt}: Count 2011: {counts_2011[i]:,}, "
              f"Count 2021: {counts_2021[i]:,}, "
              f"Change: {((counts_2021[i] / counts_2011[i]) - 1) * 100:.1f}%")


def analyze_immigration_changes(df_full_match, immigration_features):
    """
    Analyze and visualize changes in immigration features between 2011 and 2021.

    Parameters:
    -----------
    df_full_match : pandas.DataFrame
        DataFrame containing matched data for 2011 and 2021
    immigration_features : dict
        Dictionary containing feature names for both years

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added change columns
    """
    # Calculate changes
    for feature in ['uk_passports', 'born_in_uk', 'long_term', 'recent_migrant', 'total']:
        # Calculate absolute change
        change_col = f'{feature}_change'
        df_full_match[change_col] = (df_full_match[immigration_features['2021'][feature]] / df_full_match['total_residents_2021'] -
                                     df_full_match[immigration_features['2011'][feature]] / df_full_match['total_residents_2011'])

        # Calculate percentage change
        pct_change_col = f'{feature}_pct_change'
        df_full_match[pct_change_col] = ((df_full_match[immigration_features['2021'][feature]] / df_full_match['total_residents_2021'] -
                                         df_full_match[immigration_features['2011'][feature]] / df_full_match['total_residents_2011']) /
                                         df_full_match[immigration_features['2011'][feature]] / df_full_match['total_residents_2011']) * 100

    return df_full_match


def plot_immigration_changes_distribution(df_full_match, feature_labels):
    """
    Create distribution plots for immigration changes.

    Parameters:
    -----------
    df_full_match : pandas.DataFrame
        DataFrame containing change calculations
    feature_labels : dict
        Dictionary mapping feature codes to readable labels
    """
    for feature in feature_labels.keys():
        # Get valid data (non-NaN)
        abs_change = df_full_match[f'{feature}_change'].dropna()
        pct_change = df_full_match[f'{feature}_pct_change'].dropna()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f'Changes in {feature_labels[feature]} (2011-2021)', fontsize=14)

        # Plot absolute changes
        sns.histplot(data=abs_change, ax=ax1, bins=30)
        ax1.set_title(f'Absolute Proportion Change (n={len(abs_change):,})')
        ax1.set_xlabel('Change in Proportion')
        ax1.set_ylabel('Frequency')

        # Add mean and median lines
        mean_val = abs_change.mean()
        median_val = abs_change.median()
        ax1.axvline(mean_val, color='r', linestyle='--',
                    label=f'Mean: {mean_val:.0f}')
        ax1.axvline(median_val, color='g', linestyle='--',
                    label=f'Median: {median_val:.0f}')
        ax1.legend()

        # Plot percentage changes
        sns.histplot(data=pct_change, ax=ax2, bins=30)
        ax2.set_title(f'Percentage Change (n={len(pct_change):,})')
        ax2.set_xlabel('Percentage Change (%)')
        ax2.set_ylabel('Frequency')

        # Add mean and median lines
        mean_val = pct_change.mean()
        median_val = pct_change.median()
        ax2.axvline(mean_val, color='r', linestyle='--',
                    label=f'Mean: {mean_val:.1f}%')
        ax2.axvline(median_val, color='g', linestyle='--',
                    label=f'Median: {median_val:.1f}%')
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print(f"\nSummary for {feature_labels[feature]}:")
        print("\nAbsolute Change Statistics:")
        print(abs_change.describe())


def plot_immigration_changes_comparison(df_full_match, feature_labels):
    """
    Create box plot comparing percentage changes across immigration features.

    Parameters:
    -----------
    df_full_match : pandas.DataFrame
        DataFrame containing change calculations
    feature_labels : dict
        Dictionary mapping feature codes to readable labels
    """
    # Prepare data for box plot
    pct_change_data = df_full_match[[
        f'{feat}_change' for feat in feature_labels.keys()]].dropna()

    # Create box plot for percentage changes
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pct_change_data)
    plt.title('Comparison of Changes in Immigration Features (2011-2021)')
    plt.xlabel('Feature')
    plt.ylabel('Change in Proportion')
    plt.xticks(ticks=range(len(feature_labels)),
               labels=[feature_labels[feat] for feat in feature_labels.keys()],
               rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def visualize_oa_sizes_and_changes(df_full_match):
    """
    Visualize the Output Area sizes for 2011 and 2021, and the changes between these years.

    Parameters:
    -----------
    df_full_match : pandas.DataFrame
        DataFrame containing OA data with 'total_residents_2011' and 'total_residents_2021' columns
    """
    # Calculate absolute and percentage changes
    abs_change = df_full_match['total_residents_2021'] - \
        df_full_match['total_residents_2011']
    pct_change = ((df_full_match['total_residents_2021'] - df_full_match['total_residents_2011']) /
                  df_full_match['total_residents_2011'] * 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Output Area Sizes and Changes (2011-2021)', fontsize=16)

    # 2011 Distribution
    sns.histplot(
        data=df_full_match['total_residents_2011'].dropna(), ax=axes[0, 0], bins=30)
    axes[0, 0].set_title(
        f'2011 OA Sizes (n={len(df_full_match["total_residents_2011"].dropna()):,})')
    axes[0, 0].set_xlabel('Population Size')
    axes[0, 0].set_ylabel('Frequency')

    # Add mean and median lines for 2011
    mean_val_2011 = df_full_match['total_residents_2011'].mean()
    median_val_2011 = df_full_match['total_residents_2011'].median()
    axes[0, 0].axvline(mean_val_2011, color='r', linestyle='--',
                       label=f'Mean: {mean_val_2011:.0f}')
    axes[0, 0].axvline(median_val_2011, color='g',
                       linestyle='--', label=f'Median: {median_val_2011:.0f}')
    axes[0, 0].legend()

    # 2021 Distribution
    sns.histplot(
        data=df_full_match['total_residents_2021'].dropna(), ax=axes[0, 1], bins=30)
    axes[0, 1].set_title(
        f'2021 OA Sizes (n={len(df_full_match["total_residents_2021"].dropna()):,})')
    axes[0, 1].set_xlabel('Population Size')
    axes[0, 1].set_ylabel('Frequency')

    # Add mean and median lines for 2021
    mean_val_2021 = df_full_match['total_residents_2021'].mean()
    median_val_2021 = df_full_match['total_residents_2021'].median()
    axes[0, 1].axvline(mean_val_2021, color='r', linestyle='--',
                       label=f'Mean: {mean_val_2021:.0f}')
    axes[0, 1].axvline(median_val_2021, color='g',
                       linestyle='--', label=f'Median: {median_val_2021:.0f}')
    axes[0, 1].legend()

    # Plot absolute changes
    sns.histplot(data=abs_change.dropna(), ax=axes[1, 0], bins=30)
    axes[1, 0].set_title(f'Absolute Change (n={len(abs_change.dropna()):,})')
    axes[1, 0].set_xlabel('Change in Population')
    axes[1, 0].set_ylabel('Frequency')

    # Add mean and median lines for absolute change
    mean_val_abs = abs_change.mean()
    median_val_abs = abs_change.median()
    axes[1, 0].axvline(mean_val_abs, color='r', linestyle='--',
                       label=f'Mean: {mean_val_abs:.0f}')
    axes[1, 0].axvline(median_val_abs, color='g', linestyle='--',
                       label=f'Median: {median_val_abs:.0f}')
    axes[1, 0].axvline(0, color='k', linestyle='-',
                       alpha=0.2, label='No Change')
    axes[1, 0].legend()

    # Plot percentage changes
    sns.histplot(data=pct_change.dropna(), ax=axes[1, 1], bins=30)
    axes[1, 1].set_title(f'Percentage Change (n={len(pct_change.dropna()):,})')
    axes[1, 1].set_xlabel('Percentage Change (%)')
    axes[1, 1].set_ylabel('Frequency')

    # Add mean and median lines for percentage change
    mean_val_pct = pct_change.mean()
    median_val_pct = pct_change.median()
    axes[1, 1].axvline(mean_val_pct, color='r', linestyle='--',
                       label=f'Mean: {mean_val_pct:.1f}%')
    axes[1, 1].axvline(median_val_pct, color='g', linestyle='--',
                       label=f'Median: {median_val_pct:.1f}%')
    axes[1, 1].axvline(0, color='k', linestyle='-',
                       alpha=0.2, label='No Change')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nOA Size Statistics:")
    print("\n2011:")
    print(df_full_match['total_residents_2011'].describe())
    print("\n2021:")
    print(df_full_match['total_residents_2021'].describe())
    print("\nAbsolute Change:")
    print(abs_change.describe())
    print("\nPercentage Change:")
    print(pct_change.describe())

    # Print additional insights
    print(f"\nNumber of OAs with population increase: {
          (abs_change > 0).sum():,} ({(abs_change > 0).mean():.1%})")
    print(f"Number of OAs with population decrease: {
          (abs_change < 0).sum():,} ({(abs_change < 0).mean():.1%})")
    print(f"Number of OAs with no change: {
          (abs_change == 0).sum():,} ({(abs_change == 0).mean():.1%})")


def get_oa_centroids(df_full_match, connection):
    """
    Get centroids for Output Areas from the database.

    Parameters:
    -----------
    df_full_match : pandas.DataFrame
        DataFrame containing OA codes
    connection : mysql.connector.connection.MySQLConnection
        Database connection

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added latitude and longitude columns
    """
    # Get unique OA codes
    oa_codes = tuple(df_full_match['oa_code'].unique())

    # Query to get centroids
    query = """
    SELECT oa_code, 
           ST_Y(ST_Centroid(geometry)) as latitude,
           ST_X(ST_Centroid(geometry)) as longitude
    FROM oa_boundaries
    WHERE oa_code IN {}
    """.format(oa_codes)

    # Execute query and get results
    centroids = pd.read_sql_query(query, connection)

    # Merge centroids with original dataframe
    df_with_coords = df_full_match.merge(centroids, on='oa_code', how='left')

    return df_with_coords


def visualize_immigration_changes_map(df_full_match, feature='total', region_name='Cambridgeshire', connection=None):
    """
    Visualize immigration changes on a map using OpenStreetMap as background.

    Parameters:
    -----------
    df_full_match : pandas.DataFrame
        DataFrame containing OA data with immigration features
    feature : str, default='total'
        Immigration feature to visualize
    region_name : str, default='Cambridgeshire'
        Name of the region for the title
    connection : mysql.connector.connection.MySQLConnection, optional
        Database connection
    """
    # Get coordinates if not already present
    if 'latitude' not in df_full_match.columns or 'longitude' not in df_full_match.columns:
        if connection is None:
            raise ValueError(
                "Database connection required to get OA centroids")
        df_full_match = get_oa_centroids(df_full_match, connection)

    # Calculate percentage change for the selected feature
    pct_change = ((df_full_match[f'{feature}_2021'] - df_full_match[f'{feature}_2011']) /
                  df_full_match[f'{feature}_2011'] * 100)

    # Create a base map centered on the mean coordinates
    center_lat = df_full_match['latitude'].mean()
    center_lon = df_full_match['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=10,
                   tiles='OpenStreetMap')

    # Create color map
    colormap = LinearColormap(
        colors=['red', 'yellow', 'green'],
        vmin=pct_change.quantile(0.1),  # 10th percentile
        vmax=pct_change.quantile(0.9)    # 90th percentile
    )

    # Add the colormap to the map
    colormap.add_to(m)
    colormap.caption = f'Percentage Change in {
        feature.replace("_", " ").title()} (2011-2021)'

    # Add circles for each OA
    for idx, row in df_full_match.iterrows():
        change = ((row[f'{feature}_2021'] - row[f'{feature}_2011']) /
                  row[f'{feature}_2011'] * 100)

        # Create popup text
        popup_text = f"""
        <b>Output Area:</b> {row['oa_code']}<br>
        <b>2011 Value:</b> {row[f'{feature}_2011']:.0f}<br>
        <b>2021 Value:</b> {row[f'{feature}_2021']:.0f}<br>
        <b>Change:</b> {change:.1f}%
        """

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=popup_text,
            color=colormap(change),
            fill=True,
            fillColor=colormap(change),
            fillOpacity=0.7
        ).add_to(m)

    return m


def plot_immigration_changes_maps(df_full_match, region_name='Cambridgeshire', connection=None):
    """
    Create maps for all immigration features.

    Parameters:
    -----------
    df_full_match : pandas.DataFrame
        DataFrame containing OA data
    region_name : str, default='Cambridgeshire'
        Name of the region for the title
    connection : mysql.connector.connection.MySQLConnection, optional
        Database connection
    """
    features = {
        'uk_passports': 'UK Passports',
        'born_in_uk': 'Born in UK',
        'long_term': 'Long-term Residents',
        'recent_migrant': 'Recent Migrants',
        'total': 'Total Population'
    }

    for feature in features:
        print(f"\nGenerating map for {features[feature]}...")
        m = visualize_immigration_changes_map(
            df_full_match, feature, region_name, connection)
        display(m)


def filter_regions(df):
    print(len(df[~df['GeographyCode'].str[0].isin(['E', 'W'])]))
    return df[df['GeographyCode'].str[0].isin(['E', 'W'])].reset_index(drop=True)


def print_unequal_rows(df1, df2):
    # Ensure both dataframes have at least two columns
    if df1.shape[1] < 2 or df2.shape[1] < 2:
        raise ValueError("Both dataframes must have at least two columns.")

    # Create copies with only first two columns and standardized names
    df1_comp = df1.iloc[:, :2].copy()
    df2_comp = df2.iloc[:, :2].copy()

    df1_comp.columns = ['col1', 'col2']
    df2_comp.columns = ['col1', 'col2']

    # Reset index
    df1_comp = df1_comp.reset_index(drop=True)
    df2_comp = df2_comp.reset_index(drop=True)

    # Compare the dataframes
    unequal_mask = (df1_comp != df2_comp).any(axis=1)

    # Print the unequal rows from both original dataframes
    if unequal_mask.any():
        print("Unequal rows in df1:")
        print(df1.iloc[unequal_mask[unequal_mask].index])
        print("\nUnequal rows in df2:")
        print(df2.iloc[unequal_mask[unequal_mask].index])

        # Print the actual differences
        print("\nDifferences:")
        print("df1 values:", df1_comp[unequal_mask].values.tolist())
        print("df2 values:", df2_comp[unequal_mask].values.tolist())
    else:
        print("All rows are equal in the first two columns.")


def null_check(df):
    print(df.isnull().sum().sum())


def create_immigration_map(df, feature_name, feature_label, level='oa', sample_size=5000, n_top=50):
    """
    Create an interactive map showing immigration features by area

    Parameters:
    -----------
    df : pandas DataFrame
        Data containing geographic and immigration features
    feature_name : str
        Name of the feature column to visualize
    feature_label : str
        Label for the feature in the visualization
    level : str, optional (default='oa')
        Geographic level: 'oa' for Output Area or 'lad' for Local Authority District
    sample_size : int, optional (default=5000)
        Number of points to sample for OA-level visualization
    n_top : int, optional (default=50)
        Number of top LADs per cluster to show for LAD-level visualization
    """
    # Create base map centered on England
    m = folium.Map(
        location=[52.5, -1.8],
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    if level == 'oa':
        # Sample the data for OA-level
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

        # Calculate per capita values
        per_capita = df[feature_name] / df['total_residents']
        values = per_capita

    else:  # LAD level
        # Get top LADs from each cluster
        top_lads = pd.concat([
            df[df['cluster'] == 0].nlargest(n_top, 'avg_price'),
            df[df['cluster'] == 1].nlargest(n_top, 'avg_price')
        ])
        df_sample = top_lads
        values = df_sample[feature_name]

    # Use percentiles for color scaling
    vmin = np.percentile(values, 5)
    vmax = np.percentile(values, 95)

    # Create color scale
    colormap = LinearColormap(
        colors=['blue', 'white', 'red'],
        vmin=vmin,
        vmax=vmax,
        caption=f'{feature_label} {
            "(per capita)" if level == "oa" else "Value"}'
    )

    # Add markers for each area
    for idx, row in df_sample.iterrows():
        if level == 'oa':
            value = row[feature_name] / row['total_residents']
            location = [row['latitude'], row['longitude']]
            popup_html = f"{feature_label}: {value:.3f}"

            folium.Circle(
                location=location,
                radius=300,
                color=colormap(np.clip(value, vmin, vmax)),
                fill=True,
                popup=popup_html
            ).add_to(m)

        else:  # LAD level
            coords = get_lad_coordinates(row['lad_name'])
            if coords:
                lat, lon = coords
                value = values[idx]
                value_for_color = np.clip(value, vmin, vmax)
                color = colormap(value_for_color)

                popup_html = f"""
                    <b>{row['lad_name']}</b><br>
                    Cluster: {row['cluster']}<br>
                    {feature_label}: {value:.3f}<br>
                    Avg Price: £{row['avg_price']:,.0f}
                """

                if row['cluster'] == 0:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=8,
                        color=color,
                        weight=2,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7,
                        popup=popup_html,
                        tooltip=row['lad_name']
                    ).add_to(m)
                else:
                    folium.Rectangle(
                        bounds=[[lat-0.02, lon-0.02], [lat+0.02, lon+0.02]],
                        color=color,
                        weight=2,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7,
                        popup=popup_html,
                        tooltip=row['lad_name']
                    ).add_to(m)

    colormap.add_to(m)

    # Add legend for LAD level
    if level == 'lad':
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px; height: 100px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    padding: 10px;
                    border-radius: 5px;
                    ">
            <b>Clusters</b><br>
            Cluster 0: Circles<br>
            Cluster 1: Squares<br>
            <small>Top {n_top} LADs per cluster</small><br>
            <small>Colors indicate {feature_label}</small>
        </div>
        """.format(n_top=n_top, feature_label=feature_label)
        m.get_root().html.add_child(folium.Element(legend_html))

    return m


def get_lad_coordinates(lad_name):
    """
    Get coordinates for a UK Local Authority District

    Parameters:
    -----------
    lad_name : str
        Name of the Local Authority District

    Returns:
    --------
    tuple or None
        (latitude, longitude) if found, None if geocoding fails
    """
    try:
        # Add 'UK' to help with geocoding
        search_name = f"{lad_name}, UK"
        area_gdf = ox.geocode_to_gdf(search_name)
        bounds = area_gdf.total_bounds
        lat = (bounds[1] + bounds[3]) / 2
        lon = (bounds[0] + bounds[2]) / 2
        return lat, lon

    except Exception as e:
        return None


def calculate_immigration_correlations_and_top_lads(lad_summary):
    """
    Calculate correlations between immigration features and house prices,
    and show top LADs by price
    """
    # Calculate correlations
    correlations = {
        'passport_held': lad_summary['avg_uk_passports_ratio'].corr(lad_summary['avg_price']),
        'length_of_residence': {
            'born_in_uk': lad_summary['avg_born_in_uk_ratio'].corr(lad_summary['avg_price']),
            'long_term': lad_summary['avg_long_term_ratio'].corr(lad_summary['avg_price'])
        },
        'migrant_indicator': {
            'uk_migrant': lad_summary['avg_uk_migrant_ratio'].corr(lad_summary['avg_price']),
            'international': lad_summary['avg_international_migrant_ratio'].corr(lad_summary['avg_price']),
            'student': lad_summary['avg_student_address_ratio'].corr(lad_summary['avg_price'])
        }
    }

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot correlations
    corr_data = [
        correlations['passport_held'],
        correlations['length_of_residence']['born_in_uk'],
        correlations['length_of_residence']['long_term'],
        correlations['migrant_indicator']['uk_migrant'],
        correlations['migrant_indicator']['international'],
        correlations['migrant_indicator']['student']
    ]
    feature_names = ['UK\nPassports', 'Born in\nUK', 'Long Term\nResidents',
                     'UK\nMigrants', 'International\nMigrants', 'Student\nAddress']

    bars = ax1.bar(feature_names, corr_data)
    ax1.set_title(
        'Correlation between Immigration Features\nand House Prices', pad=20)
    ax1.set_ylabel('Correlation Coefficient')
    ax1.tick_params(axis='x', rotation=0)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom' if height > 0 else 'top')

    # Plot top LADs
    top_lads = lad_summary.nlargest(10, 'avg_price')
    bars = ax2.bar(top_lads['lad_name'], top_lads['avg_price'] / 1e6)
    ax2.set_title('Top 10 LADs by Average House Price', pad=20)
    ax2.set_ylabel('Average Price (£ millions)')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'£{height:.1f}M',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    return correlations


def analyze_clusters(lad_summary, n_clusters=2):
    """
    Perform clustering analysis on LADs and compare correlations between clusters
    """
    # Select features for clustering
    features = ['avg_uk_passports_ratio', 'avg_born_in_uk_ratio',
                'avg_long_term_ratio', 'avg_uk_migrant_ratio',
                'avg_international_migrant_ratio', 'avg_student_address_ratio']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(lad_summary[features])

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    lad_summary['cluster'] = kmeans.fit_predict(X)

    # Create visualizations
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    # Plot cluster characteristics (existing code)
    cluster_stats = lad_summary.groupby('cluster').agg({
        'avg_price': 'mean',
        **{feature: 'mean' for feature in features}
    })
    cluster_stats_no_price = cluster_stats.drop('avg_price', axis=1)
    cluster_stats_no_price.T.plot(kind='bar', ax=ax1)
    ax1.set_title('Cluster Characteristics', pad=20)
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Average Ratio')
    ax1.legend(['Cluster 0', 'Cluster 1'])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.3f', padding=3)

    # Plot price distribution by cluster (existing code)
    sns.boxplot(data=lad_summary, x='cluster', y='avg_price', ax=ax2)
    ax2.set_title('House Price Distribution by Cluster', pad=20)
    ax2.set_ylabel('Average Price (£)')

    # Calculate and plot correlations for each cluster
    correlations = []
    feature_names = ['UK\nPassports', 'Born in\nUK', 'Long Term\nResidents',
                     'UK\nMigrants', 'International\nMigrants', 'Student\nAddress']
    
    for cluster in [0, 1]:
        cluster_data = lad_summary[lad_summary['cluster'] == cluster]
        cluster_corrs = []
        for feature in features:
            corr = cluster_data[feature].corr(cluster_data['avg_price'])
            cluster_corrs.append(corr)
        correlations.append(cluster_corrs)

    # Plot correlation comparison
    x = np.arange(len(features))
    width = 0.35
    
    ax3.bar(x - width/2, correlations[0], width, label='Cluster 0', color='blue', alpha=0.6)
    ax3.bar(x + width/2, correlations[1], width, label='Cluster 1', color='orange', alpha=0.6)
    
    ax3.set_title('Price Correlations by Cluster', pad=20)
    ax3.set_xlabel('Features')
    ax3.set_ylabel('Correlation with Price')
    ax3.set_xticks(x)
    ax3.set_xticklabels(feature_names, rotation=45, ha='right')
    ax3.legend()

    # Add correlation values as labels
    for i, corrs in enumerate(correlations):
        for j, corr in enumerate(corrs):
            x_pos = j - width/2 if i == 0 else j + width/2
            ax3.text(x_pos, corr + (0.02 if corr >= 0 else -0.02),
                     f'{corr:.2f}',
                     ha='center', va='bottom' if corr >= 0 else 'top')

    plt.tight_layout()
    plt.show()

        # Print cluster analysis results
    print("\nCluster Analysis:")
    for cluster in [0, 1]:
        cluster_lads = lad_summary[lad_summary['cluster'] == cluster]
        
        # Print cluster size
        print(f"\nCluster {cluster} size: {len(cluster_lads)} LADs")
        
        print(f"\nTop 5 LADs in Cluster {cluster}:")
        for _, row in cluster_lads.nlargest(5, 'avg_price').iterrows():
            if row['lad_name'].strip():
                print(f"- {row['lad_name']}: £{row['avg_price']:,.0f}")

        # Print correlation summary
        print(f"\nCorrelations with price in Cluster {cluster}:")
        for feature, corr in zip(feature_names, correlations[cluster]):
            print(f"- {feature}: {corr:.3f}")

    return lad_summary, cluster_stats, correlations


def analyze_feature_pca(X, feature_names=None, n_components_display=3, figsize=(10, 6)):
    """
    Performs PCA analysis on features and visualizes the results.

    Args:
        X (np.ndarray or pd.DataFrame): Feature matrix
        feature_names (list, optional): List of feature names. If None, will use X.columns 
                                      if DataFrame or generate generic names
        n_components_display (int): Number of principal components to display in loadings table
        figsize (tuple): Figure size for the plot

    Returns:
        dict: Dictionary containing:
            - 'pca': Fitted PCA object
            - 'loadings': DataFrame of feature loadings
            - 'explained_variance_ratio': Array of explained variance ratios
            - 'cumulative_variance_ratio': Array of cumulative explained variance ratios
    """
    # Handle feature names
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Calculate variance ratios
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Create loadings DataFrame
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(feature_names))],
        index=feature_names
    )

    # Plotting
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(explained_variance_ratio) + 1),
             cumulative_variance_ratio, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Analysis of Features')
    plt.legend()
    plt.grid(True)

    # Print results
    print("\nFeature Loadings for First {} Components:".format(n_components_display))
    print(loadings.iloc[:, :n_components_display].round(3))

    print("\nExplained Variance Ratio:")
    for i, var in enumerate(explained_variance_ratio):
        print(
            f"PC{i+1}: {var:.3f} ({cumulative_variance_ratio[i]:.3f} cumulative)")

    return {
        'pca': pca,
        'loadings': loadings,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance_ratio
    }
