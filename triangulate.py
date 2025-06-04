# python triangulate.py

import math
import numpy as np
from sklearn.cluster import DBSCAN


# Constants
IMAGE_WIDTH = 4000
IMAGE_HEIGHT = 2250

PI = math.pi

def deg2rad(deg):
    return deg * math.pi / 180

def compute_ground_position(annotation, fov_x_deg, fov_y_deg, drone_lat, drone_lng, altitude_ft, heading_deg):
    """
    Projects image coordinates into ground plane using drone metadata.
    Assumes constant IMAGE_WIDTH and IMAGE_HEIGHT.
    """
    # Convert altitude to meters
    altitude_m = altitude_ft * 0.3048

    # Get center of bounding box (pixel coords)
    x_center = (annotation["x_min"] + annotation["x_max"]) / 2
    y_center = (annotation["y_min"] + annotation["y_max"]) / 2

    fov_x_rad = math.radians(fov_x_deg)
    fov_y_rad = math.radians(fov_y_deg)

    #calculate the total meters horizontal en vertical in the img from the drone to one side
    horizontal_meters = math.tan(fov_x_rad/2) * altitude_m  #goedgekeurd door Jeff
    vertical_meters = math.tan(fov_y_rad/2) * altitude_m
    
    #if x_meters_from_drone < 0 it is left side
    #if x_meters_from_drone > 0 it is right side
    x_meters_from_drone = ((x_center - (IMAGE_WIDTH/2))/(IMAGE_WIDTH/2)) * horizontal_meters
    y_meters_from_drone = ((y_center - (IMAGE_HEIGHT/2))/(IMAGE_HEIGHT/2)) * vertical_meters
    

    # Convert heading to radians
    heading_rad = math.radians(heading_deg)
    
    if 0 <= heading_deg < 90: #gechecked door Jeff
        # Rotate the point (x, y) by -heading (to align with North)
        #north_offset_m: how far north (positive) or south (negative) the point is from the drone
        #east_offset_m: how far east (positive) or west (negative)
        north_offset_m_x = - x_meters_from_drone * math.sin(heading_rad)
        north_offset_m_y = - y_meters_from_drone * math.cos(heading_rad)
        
        east_offset_m_x = x_meters_from_drone * math.cos(heading_rad)
        east_offset_m_y = - y_meters_from_drone * math.sin(heading_rad)
        
        
    elif 90 <= heading_deg < 180:
        north_offset_m_x = - x_meters_from_drone * math.sin(PI - heading_rad)
        north_offset_m_y = y_meters_from_drone * math.cos(PI - heading_rad)
        
        east_offset_m_x = - x_meters_from_drone * math.cos(PI - heading_rad)
        east_offset_m_y = - y_meters_from_drone * math.sin(PI - heading_rad)
        
        
    elif 180 <= heading_deg < 270:
        north_offset_m_x = x_meters_from_drone * math.sin(heading_rad - PI)
        north_offset_m_y = y_meters_from_drone * math.cos(heading_rad - PI)
        
        east_offset_m_x = - x_meters_from_drone * math.cos(heading_rad - PI)
        east_offset_m_y = y_meters_from_drone * math.sin(heading_rad - PI)
        
        
    elif 270 <= heading_deg < 360:
        north_offset_m_x = x_meters_from_drone * math.sin((2 * PI) - heading_rad)
        north_offset_m_y = - y_meters_from_drone * math.cos((2 * PI) - heading_rad)
        
        east_offset_m_x = x_meters_from_drone * math.cos((2 * PI) - heading_rad)
        east_offset_m_y = y_meters_from_drone * math.sin((2 * PI) - heading_rad)

    north_offset_m = north_offset_m_x + north_offset_m_y
    east_offset_m = east_offset_m_x + east_offset_m_y
    
    
    drone_lat_rad = math.radians(drone_lat)
    
    # Approximate meters per degree at this latitude
    meters_per_deg_lat = 111194.9                    # dit getal specifieker voor hier.
    meters_per_deg_lng = 111194.9 * math.cos(drone_lat_rad)

    # Convert meter offsets to degree offsets
    delta_lat = north_offset_m / meters_per_deg_lat
    delta_lng = east_offset_m / meters_per_deg_lng

    # Compute new lat/lng
    estimated_lat = drone_lat + delta_lat
    estimated_lng = drone_lng + delta_lng

    return (estimated_lat, estimated_lng)




def cluster_annotations(images_metadata, distance_threshold_m=0.5):
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
                fov_x_deg=70,  # horizontal
                fov_y_deg=55,  # verticalfov_deg=fov_deg,
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

    print("🌎all_prjected_points",all_projected_points)
    # Convert lat/lng to meters (flat projection for clustering)
    coords_meters = np.array([
        [lat * 111320, lng * 111320 * math.cos(deg2rad(lat))]
        for lat, lng in all_projected_points
    ])

    # Cluster with DBSCAN
    db = DBSCAN(eps=distance_threshold_m, min_samples=2).fit(coords_meters)
    labels = db.labels_

    # Organize clusters
    clusters = {}
    for label, meta in zip(labels, point_metadata):
        clusters.setdefault(label, []).append(meta)

    # Filter out singleton clusters
    filtered_clusters = [cluster for cluster in clusters.values() if len(cluster) > 1]

    return filtered_clusters

