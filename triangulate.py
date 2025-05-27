# python triangulate.py

import math
import numpy as np
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster

# Constants
IMAGE_WIDTH = 4000
IMAGE_HEIGHT = 2250

def deg2rad(deg):
    return deg * math.pi / 180

def compute_ground_position(annotation, fov_deg, drone_lat, drone_lng, altitude_ft, heading_deg):
    """
    Projects image coordinates into ground plane using drone metadata.
    Assumes constant IMAGE_WIDTH and IMAGE_HEIGHT.
    """
    # Convert altitude to meters
    altitude_m = altitude_ft * 0.3048

    # Get center of bounding box (pixel coords)
    x_center = (annotation["x_min"] + annotation["x_max"]) / 2
    y_center = (annotation["y_min"] + annotation["y_max"]) / 2

    # Convert FOV to radians and calculate angle per pixel
    fov_rad = deg2rad(fov_deg)
    angle_per_pixel = fov_rad / IMAGE_WIDTH

    # Horizontal offset from center (pixels -> radians)
    dx_pixels = x_center - (IMAGE_WIDTH / 2)
    dy_pixels = y_center - (IMAGE_HEIGHT / 2)  # positive down

    angle_x = dx_pixels * angle_per_pixel
    angle_y = dy_pixels * angle_per_pixel  # approximate for vertical

    # Project to ground plane assuming nadir view
    x_offset_m = altitude_m * math.tan(angle_x)
    y_offset_m = altitude_m * math.tan(angle_y)

    # Rotate by heading to get true north-relative position
    heading_rad = deg2rad(heading_deg)
    rotated_x = x_offset_m * math.cos(heading_rad) - y_offset_m * math.sin(heading_rad)
    rotated_y = x_offset_m * math.sin(heading_rad) + y_offset_m * math.cos(heading_rad)

    # Approximate meter per degree (valid near equator)
    meters_per_deg_lat = 111320
    meters_per_deg_lng = 111320 * math.cos(deg2rad(drone_lat))

    # Convert offsets in meters to lat/lng
    lat_offset_deg = rotated_y / meters_per_deg_lat
    lng_offset_deg = rotated_x / meters_per_deg_lng

    estimated_lat = drone_lat + lat_offset_deg
    estimated_lng = drone_lng + lng_offset_deg

    return (estimated_lat, estimated_lng)


def visualize_clusters_on_map(clusters, map_filename="static/clusters_map.html"):
    if not clusters:
        print("No clusters to visualize.")
        return

    # Use average of all points for map center
    all_coords = [(item['lat'], item['lng']) for cluster in clusters for item in cluster]
    avg_lat = sum(coord[0] for coord in all_coords) / len(all_coords)
    avg_lng = sum(coord[1] for coord in all_coords) / len(all_coords)

    # Create base map
    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=18)

    # Optional: use cluster markers
    marker_cluster = MarkerCluster().add_to(m)

    for cluster_id, cluster in enumerate(clusters):
        for point in cluster:
            folium.Marker(
                location=[point['lat'], point['lng']],
                popup=f"Cluster {cluster_id}<br>{point['filename']}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(marker_cluster)

    m.save(map_filename)
    print(f"✅ Cluster map saved to: {map_filename}")
    
    

def cluster_annotations(images_metadata, fov_deg=83, distance_threshold_m=1):
    """
    Projects all bounding boxes into GPS space and clusters nearby detections.
    Assumes constant IMAGE_WIDTH and IMAGE_HEIGHT.
    Each image entry should be:
    {
        "filename": str,
        "drone_lat": float,
        "drone_lng": float,
        "altitude_ft": float,
        "heading_deg": float,
        "bounding_boxes": [ {x_min, y_min, x_max, y_max}, ... ]
    }
    """
    all_projected_points = []
    point_metadata = []

    for image in images_metadata:
        for box in image["bounding_boxes"]:
            lat, lng = compute_ground_position(
                annotation=box,
                fov_deg=fov_deg,
                drone_lat=image["drone_lat"],
                drone_lng=image["drone_lng"],
                altitude_ft=image["altitude_ft"],
                heading_deg=image["heading_deg"]
            )
            all_projected_points.append([lat, lng])
            point_metadata.append({
                "filename": image["filename"],
                "box": box,
                "lat": lat,
                "lng": lng
            })

    # Convert lat/lng to meters (flat projection for clustering)
    coords_meters = np.array([
        [lat * 111320, lng * 111320 * math.cos(deg2rad(lat))]
        for lat, lng in all_projected_points
    ])

    # Cluster with DBSCAN
    db = DBSCAN(eps=distance_threshold_m, min_samples=1).fit(coords_meters)
    labels = db.labels_

    # Organize clusters
    clusters = {}
    for label, meta in zip(labels, point_metadata):
        clusters.setdefault(label, []).append(meta)
    #print("🥵 clusters.values()", list(clusters.values()))
    return list(clusters.values())
