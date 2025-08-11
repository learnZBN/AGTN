import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import numpy as np


# Set font for plots (removed Chinese font setting as labels are now English)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False # This can remain as it handles minus signs

# --- Data Loading and Preprocessing ---
def load_data(file_path):
    """
    Loads data from a CSV file, parses dates, and handles missing values.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['tstamp2'])
        df.set_index('tstamp2', inplace=True)

        if df.isnull().any().any():
            print("Missing values detected. Filling with column mean.")
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# --- Plotting Functions ---
def plot_yearly_trend(df, load_features, save_path='yearly_load_curves.png'):
    """
    Generates a yearly trend plot for each load feature.
    """
    if not df.index.is_unique:
        df = df.resample('H').mean()

    fig, axes = plt.subplots(nrows=len(load_features), ncols=1, figsize=(15, 12), sharex=True)
    if len(load_features) == 1:
        axes = [axes]  # Make it iterable if there's only one subplot

    titles = {
        'KW': 'Annual Electricity Load Curve',
        'CHWTON': 'Annual Cooling Load Curve',
        'HTmmBTU': 'Annual Heating Load Curve'
    }
    y_labels = {
        'KW': 'Electricity (KW)',
        'CHWTON': 'Cooling (CHWTON)',
        'HTmmBTU': 'Heating (HTmmBTU)'
    }
    colors = ['#FF6347', '#4682B4', '#8A2BE2']

    for i, feature in enumerate(load_features):
        if feature in df.columns:
            ax = axes[i]
            ax.plot(df.index, df[feature], label=feature, color=colors[i], linewidth=1)
            ax.set_title(titles.get(feature, f'Yearly Trend of {feature}'), fontsize=14)
            ax.set_ylabel(y_labels.get(feature, feature), fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)

            # Format x-axis for monthly display
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45)
        else:
            print(f"Warning: Feature '{feature}' not found in the data.")

    plt.tight_layout()
    plt.suptitle('Annual Electricity, Cooling, and Heating Load Trend Curves', fontsize=16, y=1.02)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Yearly trend plot saved to {save_path}")


def plot_weekly_trend(df, load_features, days_to_plot=7, save_path='weekly_load_trends.png'):
    """
    Generates a weekly trend plot for a specific time period.
    """
    # Select the last `days_to_plot` days of data
    start_date = df.index.max() - pd.Timedelta(days=days_to_plot)
    weekly_data = df[df.index >= start_date]

    if weekly_data.empty:
        print(f"Warning: Not enough data for a {days_to_plot}-day plot.")
        return

    fig, axes = plt.subplots(nrows=len(load_features), ncols=1, figsize=(15, 12), sharex=True)
    if len(load_features) == 1:
        axes = [axes]

    titles = {
        'KW': 'Last Seven Days Electricity Load Trend',
        'CHWTON': 'Last Seven Days Cooling Load Trend',
        'HTmmBTU': 'Last Seven Days Heating Load Trend'
    }
    y_labels = {
        'KW': 'Electricity (KW)',
        'CHWTON': 'Cooling (CHWTON)',
        'HTmmBTU': 'Heating (HTmmBTU)'
    }
    colors = ['#FF6347', '#4682B4', '#8A2BE2']

    for i, feature in enumerate(load_features):
        if feature in weekly_data.columns:
            ax = axes[i]
            ax.plot(weekly_data.index, weekly_data[feature], label=feature, color=colors[i])
            ax.set_title(titles.get(feature, f'Weekly Trend of {feature}'), fontsize=14)
            ax.set_ylabel(y_labels.get(feature, feature), fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)

            # Format x-axis to show date and time clearly
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            ax.tick_params(axis='x', rotation=45)
        else:
            print(f"Warning: Feature '{feature}' not found in the data.")

    plt.tight_layout()
    plt.suptitle(f'{days_to_plot}-Day Electricity, Cooling, and Heating Load Dynamic Trends', fontsize=16, y=1.02)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Weekly trend plot saved to {save_path}")


# --- Main Execution ---
if __name__ == "__main__":
    # Define the path to the CSV file
    file_path = 'load2023.csv'

    # Load and preprocess the data
    load_data_df = load_data(file_path)

    if load_data_df is not None:
        # Define the load features to be plotted
        load_features_to_plot = ['KW', 'CHWTON', 'HTmmBTU']

        # 1. Plot yearly trend curves
        print("Generating yearly trend plots...")
        plot_yearly_trend(load_data_df, load_features_to_plot)

        # 2. Plot weekly trend curves (e.g., for the last 7 days)
        print("\nGenerating weekly trend plots...")
        plot_weekly_trend(load_data_df, load_features_to_plot, days_to_plot=7)
