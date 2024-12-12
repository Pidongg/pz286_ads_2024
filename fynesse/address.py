import osmnx as ox
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


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


def compare_feature_correlations(df, feature_groups, population_col='total_residents', target='percentage'):
    """
    Compare correlations between raw counts and per-capita features with target variable.
    Memory-efficient version.
    """
    all_correlations = []

    # Process each feature group
    for group_name, features in feature_groups.items():
        for feature in features:
            if feature not in df.columns:
                print(f"Warning: {feature} not found in dataframe")
                continue

            # Calculate correlations more efficiently
            raw_corr = df[feature].corr(df[target])

            # Calculate per-capita correlation without creating new column
            per_capita_series = df[feature].div(df[population_col])
            per_capita_corr = per_capita_series.corr(df[target])

            all_correlations.append({
                'group': group_name,
                'feature': feature,
                'raw_correlation': raw_corr,
                'per_capita_correlation': per_capita_corr,
                'difference': abs(per_capita_corr) - abs(raw_corr),
                'recommended': 'per_capita' if abs(per_capita_corr) > abs(raw_corr) else 'raw'
            })

    # Create DataFrame with results
    results_df = pd.DataFrame(all_correlations)

    # Sort by absolute difference in correlations
    results_df = results_df.sort_values('difference', ascending=False)

    # Plot comparison
    plt.figure(figsize=(15, 6))
    x = np.arange(len(results_df))
    width = 0.35

    plt.bar(x - width/2, abs(results_df['raw_correlation']), width,
            label='Raw counts', color='blue', alpha=0.6)
    plt.bar(x + width/2, abs(results_df['per_capita_correlation']), width,
            label='Per capita', color='green', alpha=0.6)

    plt.xlabel('Features')
    plt.ylabel('Absolute Correlation')
    plt.title(
        f'Comparison of Raw vs Per-Capita Feature Correlations with {target}')
    plt.xticks(x, results_df['feature'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results_df


def create_per_capita_features(df, features, population_col='total_residents'):
    for feature in features:
        per_capita_name = f'{feature}_per_capita'
        df[per_capita_name] = df[feature] / df[population_col]


def plot_gaussian_model_results(X, y, feature_names=None, feature_index=0, title=None, alpha=0.05,
                                link_function='log', create_plot=True):
    """
    Fit and plot Gaussian GLM results with specified link function.
    """
    X = np.array(X)  # Features matrix
    y = np.array(y)  # Target variable

    # Define link function
    if link_function == 'log':
        family = sm.families.Gaussian(link=sm.families.links.Log())
    elif link_function == 'inverse':
        family = sm.families.Gaussian(link=sm.families.links.inverse_power())
    elif link_function == 'sqrt':
        family = sm.families.Gaussian(link=sm.families.links.sqrt())
    else:  # default to identity
        family = sm.families.Gaussian(link=sm.families.links.Identity())

    # Fit Gaussian GLM with specified link function
    glm_model = sm.GLM(y, X, family=family)
    glm_results = glm_model.fit()

    # Print model summary
    print(f"\nGaussian GLM Results (with {link_function} link):")
    print("=" * 50)
    print(glm_results.summary())

    if not create_plot:
        return glm_results, None

    # Get feature name
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    feature_name = feature_names[feature_index]

    # Select feature for plotting and sort data
    x_feature = X[:, feature_index]
    sort_idx = np.argsort(x_feature)
    x_feature_sorted = x_feature[sort_idx]
    y_sorted = y[sort_idx]
    X_sorted = X[sort_idx]

    # Get predictions and confidence intervals using sorted data
    predictions = glm_results.get_prediction(
        X_sorted).summary_frame(alpha=alpha)

    # Create plot
    fig = plt.figure(figsize=(12, 6))

    # Plot actual data points
    plt.scatter(x_feature,  # Original unsorted x for scatter plot
                y,          # Original unsorted y for scatter plot
                marker='X',
                color='blue',
                edgecolor='black',
                s=100,
                alpha=0.5,
                zorder=1,
                label='Actual')

    # Plot predicted line and CI with sorted data
    plt.plot(x_feature_sorted,
             predictions['mean'],
             color='red',
             linewidth=3.0,
             zorder=2,
             label=f'Predicted ({link_function} link)')

    plt.fill_between(
        x_feature_sorted,
        predictions['mean_ci_lower'],
        predictions['mean_ci_upper'],
        color='red',
        alpha=0.2,
        label=f'{int((1-alpha)*100)}% CI'
    )

    plt.xlabel(feature_name)
    plt.ylabel('Price')
    if title is None:
        title = f'Gaussian GLM with {
            link_function} link: Price vs {feature_name}'
    plt.title(title)
    plt.legend()

    return glm_results, fig


def fit_linear_model(feature_list, y, alpha=0.05, silent=False):
    """
    Fit a linear model with per-capita features and visualize results.
    """
    X = np.array(feature_list)  # Features matrix
    y = np.array(y)  # Target variable

    # Fit model
    model = sm.OLS(y, X)
    results = model.fit()
    if silent:
        return results, None

    # Sort data by first feature for plotting
    sort_idx = np.argsort(X[:, 0])
    X_sorted = X[sort_idx]
    feature_sorted = X[:, 0][sort_idx]

    # Get predictions using sorted data
    predictions = results.get_prediction(X_sorted).summary_frame(alpha=alpha)

    # Create visualization
    fig = plt.figure(figsize=(12, 6))

    # Plot original data points
    plt.scatter(X[:, 0], y,
                alpha=0.5,
                label='Actual',
                color='blue')

    # Plot predicted line and CI with sorted data
    plt.plot(feature_sorted,
             predictions['mean'],
             color='red',
             label='Predicted')

    plt.fill_between(
        feature_sorted,
        predictions['mean_ci_lower'],
        predictions['mean_ci_upper'],
        alpha=0.2,
        color='red',
        label=f'{int((1-alpha)*100)}% CI'
    )

    plt.xlabel('Features')
    plt.ylabel('Price')
    plt.title(f'Linear Model: Price vs features')
    plt.legend()

    # Print model summary and coefficients
    print("\nModel Summary:")
    print("=" * 50)
    print(results.summary())

    print(results.params)

    return results, fig


def compare_with_baseline(X_full, X_baseline, y, property_type='F', link_function='log', create_plot=True):
    """
    Compare full model with baseline model for predicting house prices.

    Parameters:
    -----------
    X_full : numpy.ndarray
        Feature matrix including all features
    X_baseline : numpy.ndarray
        Feature matrix for baseline model
    y : numpy.ndarray or pandas.Series
        Target variable (prices)
    property_type : str, optional
        Property type label for display purposes
    link_function : str, optional
        Link function to use ('log', 'identity')
    create_plot : bool, optional
        Whether to create comparison plots

    Returns:
    --------
    baseline_results : GLMResults
        Results from baseline model
    full_results : GLMResults
        Results from full model
    fig : matplotlib.figure.Figure or None
        Comparison plot if create_plot=True, else None
    """
    # Get model results
    baseline_results, _ = plot_gaussian_model_results(
        X_baseline, y,
        feature_names=[f'baseline_{i+1}' for i in range(X_baseline.shape[1])],
        link_function=link_function,
        create_plot=False
    )

    full_results, _ = plot_gaussian_model_results(
        X_full, y,
        feature_names=[f'feature_{i+1}' for i in range(X_full.shape[1])],
        link_function=link_function,
        create_plot=False
    )

    # Calculate pseudo R-squared
    pseudo_r2_baseline = 1 - baseline_results.deviance / baseline_results.null_deviance
    pseudo_r2_full = 1 - full_results.deviance / full_results.null_deviance

    # Print comparison
    print(f"\nModel Comparison for Property Type {property_type}:")
    print("=" * 50)
    print(f"Number of observations: {len(y)}")
    print(f"Baseline features: {X_baseline.shape[1]}")
    print(f"Full model features: {X_full.shape[1]}")
    print(f"\nBaseline Model Pseudo R-squared: {pseudo_r2_baseline:.4f}")
    print(f"Full Model Pseudo R-squared: {pseudo_r2_full:.4f}")
    print(
        f"Improvement in Pseudo R-squared: {pseudo_r2_full - pseudo_r2_baseline:.4f}")

    if create_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.scatter(baseline_results.fittedvalues, y,
                    alpha=0.5, label='Actual vs Predicted')
        ax2.scatter(full_results.fittedvalues, y, alpha=0.5,
                    color='red', label='Actual vs Predicted')

        ax1.set_xlabel('Predicted Price')
        ax1.set_ylabel('Actual Price')
        ax2.set_xlabel('Predicted Price')
        ax2.set_ylabel('Actual Price')

        ax1.set_title(f'Property Type {property_type}\nBaseline Model\nPseudo R² = {
                      pseudo_r2_baseline:.4f}')
        ax2.set_title(f'Property Type {property_type}\nFull Model\nPseudo R² = {
                      pseudo_r2_full:.4f}')

        # Add perfect prediction line
        min_val = min(min(y), min(baseline_results.fittedvalues),
                      min(full_results.fittedvalues))
        max_val = max(max(y), max(baseline_results.fittedvalues),
                      max(full_results.fittedvalues))
        ax1.plot([min_val, max_val], [min_val, max_val],
                 'r--', label='Perfect Prediction')
        ax2.plot([min_val, max_val], [min_val, max_val],
                 'r--', label='Perfect Prediction')

        ax1.legend()
        ax2.legend()
        plt.tight_layout()

        return baseline_results, full_results, fig

    return baseline_results, full_results, None


def compare_models_kfold(models_dict, X, y, k=5):
    """
    Compare different models using k-fold cross validation.

    Args:
    models_dict (dict): Dictionary of model names and their model classes (not fitted models)
    X (np.array): Feature matrix
    y (np.array): Target values
    k (int): Number of folds
    """
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Dictionary to store results
    results = {
        'model': [],
        'fold': [],
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2': []
    }

    # For each model
    for model_name, model_class in models_dict.items():
        # For each fold
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create and fit model
            if isinstance(model_class, type(sm.GLM)):
                model = model_class(y_train, X_train)
            else:
                model = model_class(y_train, X_train)

            fitted_model = model.fit()
            y_pred = fitted_model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store results
            results['model'].append(model_name)
            results['fold'].append(fold)
            results['mse'].append(mse)
            results['rmse'].append(rmse)
            results['mae'].append(mae)
            results['r2'].append(r2)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate average metrics for each model
    summary = results_df.groupby('model').agg({
        'mse': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'r2': ['mean', 'std']
    }).round(4)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['mse', 'rmse', 'mae', 'r2']

    for i, metric in enumerate(metrics):
        ax = axes[i//2, i % 2]
        sns.boxplot(data=results_df, x='model', y=metric, ax=ax)
        ax.set_title(f'{metric.upper()} by Model')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    return summary, results_df
