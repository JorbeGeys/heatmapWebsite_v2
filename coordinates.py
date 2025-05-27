image_coordinates = [
    (51.265896000, 4.609984000),#1
    (51.265900909, 4.609978364),#2
    (51.265905818, 4.609972727),#3
    (51.265910727, 4.609967091),#4
    (51.265915636, 4.609961455),#5
    (51.265920545, 4.609955818),#6
    (51.265925455, 4.609950182),#7
    (51.265930364, 4.609944545),#8
    (51.265935273, 4.609938909),#9
    (51.265940182, 4.609933273),#10
    (51.265945091, 4.609927636),#11
    (51.265950000, 4.609922000),#12
    (51.265929000, 4.609904000),#13
    (51.265927091, 4.609902364),#14
    (51.265925182, 4.609900727),#15
    (51.265923273, 4.609899091),#16
    (51.265921364, 4.609897455),#17
    (51.265919455, 4.609895818),#18
    (51.265917545, 4.609894182),#19
    (51.265915636, 4.609892545),#20
    (51.265913727, 4.609890909),#21
    (51.265911818, 4.609889273),#22
    (51.265909909, 4.609887636),#23
    (51.265908000, 4.609886000),#24
]

import numpy as np
import matplotlib.pyplot as plt
import os

def generate_simple_heatmap(image_count_map, coordinates, output_path):
    print("🧪 Generating simple grid heatmap...")

    grid_size = 100  # Controls resolution of heatmap

    # Extract lat/lngs
    lats = [coord[0] for coord in coordinates]
    lngs = [coord[1] for coord in coordinates]
    min_lat, max_lat = min(lats), max(lats)
    min_lng, max_lng = min(lngs), max(lngs)

    # Create empty heatmap grid
    heatmap = np.zeros((grid_size, grid_size))

    def latlng_to_grid(lat, lng):
        # Normalize to grid index
        x = int((lng - min_lng) / (max_lng - min_lng + 1e-9) * (grid_size - 1))
        y = int((lat - min_lat) / (max_lat - min_lat + 1e-9) * (grid_size - 1))
        return y, x

    # Fill grid with counts
    for i, (filename, count) in enumerate(image_count_map):
        lat, lng = coordinates[i]
        y, x = latlng_to_grid(lat, lng)
        heatmap[y, x] += count

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="hot", origin="lower")
    plt.colorbar(label="Detected Objects")
    plt.title("Heatmap of Detections")
    plt.xlabel("Longitude (grid)")
    plt.ylabel("Latitude (grid)")

    heatmap_path = os.path.join(output_path, "heatmap_simple.png")
    plt.savefig(heatmap_path)
    plt.close()

    print(f"✅ Heatmap saved at: {heatmap_path}")
    return heatmap_path
