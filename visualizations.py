# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:23:27 2026
Updated: added plot_sensor_graph to visually verify the k-NN sensor graph.

@author: usman.anjum
"""

# import matplotlib.pyplot as plt
# import numpy as np


# def plot_sensor_map(coords, source_location):

#     plt.figure(figsize=(6, 6))

#     plt.scatter(
#         coords[:, 0],
#         coords[:, 1],
#         label="Sensors")

#     plt.scatter(
#         source_location[0],
#         source_location[1],
#         marker="*",
#         s=300,
#         label="Source")

#     plt.xlabel("X")
#     plt.ylabel("Y")

#     plt.legend()

#     plt.title("Sensor Locations")

#     plt.show()


# def plot_sensor_graph(coords, adjacency, source_location=None):
#     """
#     Visualizes the k-NN sensor graph: draws an edge between every pair
#     of sensors connected in the adjacency matrix, with sensors and
#     (optionally) the true event source overlaid. Use this to sanity
#     check that the graph looks geometrically reasonable -- no isolated
#     far-flung nodes, no obviously disconnected clusters.
#     """

#     plt.figure(figsize=(6, 6))

#     N = coords.shape[0]
#     for i in range(N):
#         for j in range(i + 1, N):
#             if adjacency[i, j] > 0:
#                 plt.plot(
#                     [coords[i, 0], coords[j, 0]],
#                     [coords[i, 1], coords[j, 1]],
#                     color="gray", linewidth=0.7, zorder=1
#                 )

#     plt.scatter(coords[:, 0], coords[:, 1], label="Sensors", zorder=2)

#     if source_location is not None:
#         plt.scatter(
#             source_location[0], source_location[1],
#             marker="*", s=300, label="Source", zorder=3
#         )

#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.legend()
#     plt.title("Sensor Graph (k-NN)")

#     plt.show()


# def plot_sensor_timeseries(U, sensor_id=0):

#     plt.figure(figsize=(8, 4))

#     plt.plot(U[sensor_id, :, 0])

#     plt.xlabel("Time")

#     plt.ylabel("Signal")

#     plt.title(f"Sensor {sensor_id}")

#     plt.show()


# def plot_event_field(E):

#     plt.figure(figsize=(10, 6))

#     plt.imshow(E[:, :, 0], aspect="auto")

#     plt.colorbar(label="Event Intensity")

#     plt.xlabel("Time")

#     plt.ylabel("Sensor")

#     plt.title("Ground Truth Event Field")

#     plt.show()

# -*- coding: utf-8 -*-
"""
Enhanced Visualization Suite for Spatiotemporal Localization Frameworks.
Optimized for multi-feature geographic sensor overlays and predictive alignment.
Target Venue: IEEE Big Data 2026.
@author: usman.anjum
"""

import matplotlib.pyplot as plt
import numpy as np

# Standardized publication color palette
C_SENSOR = "#1f77b4"     # Sleek Cobalt Blue
C_EDGE = "#94a3b8"       # Clean Muted Slate Gray
C_TRUE = "#22c55e"       # Vibrant Green for True Source
C_PRED = "#ef4444"       # Sharp Red for Predicted Source
FONT_TITLE = {"fontsize": 13, "fontweight": "bold"}
FONT_LABEL = {"fontsize": 11}

# Ordered mapping index to give your plots real environmental column names
FEATURE_NAMES = [
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", 
    "Wind Speed", "Wind Direction", "Temperature", "Pressure", "Precipitation"
]

def plot_sensor_map(coords, source_location, predicted_location=None, title_label="Paris_Validation", save_path=None):
    """
    Plots the spatial coordinate arrangement of the monitoring network grid.
    Dynamically maps True vs. Model Predicted event epicenters if provided.
    """
    plt.figure(figsize=(7, 7), dpi=100)
    
    # 1. Plot the physical station layout array
    plt.scatter(
        coords[:, 1], coords[:, 0], 
        color=C_SENSOR, alpha=0.7, edgecolors="k", s=50, zorder=2, label="Monitoring Stations"
    )
    
    # 2. Plot True Source Origin Epicenter
    plt.scatter(
        source_location[1], source_location[0], 
        marker="*", color=C_TRUE, edgecolors="k", s=350, linewidths=1.5, zorder=4, label="True Hazard Origin"
    )
    
    # 3. Plot Predicted Source Location Node if it is passed out of the evaluation loops
    if predicted_location is not None:
        plt.scatter(
            predicted_location[1], predicted_location[0], 
            marker="X", color=C_PRED, edgecolors="k", s=200, linewidths=1.2, zorder=3, label="Model Predicted Origin"
        )
        # Draw a clean dashed physical tracking error line between the two positions
        plt.plot(
            [source_location[1], predicted_location[1]], 
            [source_location[0], predicted_location[0]], 
            color=C_PRED, linestyle="--", linewidth=1.5, alpha=0.8, zorder=3, label="Localization Distance Gap"
        )

    plt.xlabel("Longitude (°E)", fontdict=FONT_LABEL)
    plt.ylabel("Latitude (°N)", fontdict=FONT_LABEL)
    plt.title(f"Geospatial Localization Footprint\n[{title_label}]", fontdict=FONT_TITLE)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right", frameon=True, shadow=False, facecolor="white", edgecolor="#e2e8f0")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_sensor_graph(coords, adjacency, feature_idx=0, source_location=None, predicted_location=None, title_label="Paris", save_path=None):
    """
    Visualizes the feature-specific k-NN topological sensor network graph.
    """
    plt.figure(figsize=(7, 7), dpi=100)
    N = coords.shape[0]
    
    # 1. DRAW NETWORK CONNECTION LINES (EDGES)
    # Loops through the adjacency slice for your active parameter channel
    edge_drawn = False
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i, j] > 0:
                plt.plot(
                    [coords[i, 1], coords[j, 1]], 
                    [coords[i, 0], coords[j, 0]], 
                    color=C_EDGE, linestyle="-", linewidth=0.8, alpha=0.6, zorder=1
                )
                edge_drawn = True
                
    # Fallback text if a channel dropout forced all lines to completely disconnect
    if not edge_drawn:
        plt.text(
            coords[:, 1].mean(), coords[:, 0].mean(), "Channel Dropout: Isolated Topology", 
            color=C_PRED, fontsize=10, fontstyle="italic", ha="center", va="center", zorder=5
        )

    # 2. PLOT GRID SENSORS
    plt.scatter(
        coords[:, 1], coords[:, 0], 
        color=C_SENSOR, alpha=0.8, edgecolors="k", s=45, zorder=2, label="Grid Stations"
    )
    
    # 3. OVERLAY SOURCE EPICENTERS
    if source_location is not None:
        plt.scatter(
            source_location[1], source_location[0], 
            marker="*", color=C_TRUE, edgecolors="k", s=300, linewidths=1.2, zorder=4, label="True Source"
        )
    if predicted_location is not None:
        plt.scatter(
            predicted_location[1], predicted_location[0], 
            marker="X", color=C_PRED, edgecolors="k", s=180, linewidths=1.0, zorder=3, label="Predicted Source"
        )

    # Dynamically look up the clean physical parameter column name string
    feature_name = FEATURE_NAMES[feature_idx] if feature_idx < len(FEATURE_NAMES) else f"Feature {feature_idx}"

    plt.xlabel("Longitude (°E)", fontdict=FONT_LABEL)
    plt.ylabel("Latitude (°N)", fontdict=FONT_LABEL)
    plt.title(f"Topological Sub-Graph Map: {feature_name}\n[{title_label} Grid Profile]", fontdict=FONT_TITLE)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e2e8f0")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_sensor_timeseries(U, sensor_id=0, feature_idx=0, save_path=None):
    """
    Plots the 100-hour environmental measurement trend line for a specific station.
    """
    plt.figure(figsize=(9, 4), dpi=100)
    
    # Extract the full 100-hour tracking array timeline for this specific sensor channel
    timeline_signal = U[sensor_id, :, feature_idx]
    
    plt.plot(timeline_signal, color=C_SENSOR, linewidth=2, alpha=0.9, label=f"Station {sensor_id} Signal")
    
    feature_name = FEATURE_NAMES[feature_idx] if feature_idx < len(FEATURE_NAMES) else f"Feature {feature_idx}"
    
    plt.xlabel("Time Horizon Progression (Hours)", fontdict=FONT_LABEL)
    plt.ylabel("Normalized Magnitude Scale [0.0 - 1.0]", fontdict=FONT_LABEL)
    plt.title(f"100-Hour Parameter Timeline Analysis: {feature_name} (Sensor {sensor_id})", fontdict=FONT_TITLE)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.xlim(0, U.shape[1] - 1)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper left")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_event_field(E, feature_idx=0, save_path=None):
    """
    Generates a dense spatiotemporal heatmap of the spreading pollutant field.
    """
    plt.figure(figsize=(11, 5), dpi=100)
    
    # Render the dense 2D matrix slice for this active parameter channel
    # Y-Axis = Station indices, X-Axis = Hourly progressions
    heatmap = plt.imshow(E[:, :, feature_idx], aspect="auto", cmap="viridis", interpolation="nearest")
    
    cbar = plt.colorbar(heatmap)
    cbar.set_label("Min-Max Compressed Signal Intensity Scale", rotation=270, labelpad=15, fontsize=10)
    
    feature_name = FEATURE_NAMES[feature_idx] if feature_idx < len(FEATURE_NAMES) else f"Feature {feature_idx}"
    
    plt.xlabel("Spatiotemporal Timeline Progression (Hours)", fontdict=FONT_LABEL)
    plt.ylabel("Discrete Sensor Identification Index Tokens (N)", fontdict=FONT_LABEL)
    plt.title(f"Ground Truth Propagation Wave Matrix: {feature_name} Field Matrix", fontdict=FONT_TITLE)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()