# source venv/bin/activate
# docker running in background (whale icon)
# python inference_server.py
# CTRL + C to stop
#(venv) jorbegeys@Jorbes-MacBook-Pro heat map website v2 %
# git push -u origin main

from flask import Flask, request, jsonify, send_from_directory
import inference
import supervision as sv
import cv2
import numpy as np
import os
from PIL import Image, ImageFile # Import ImageFile
import traceback
import subprocess
import shutil
import uuid # Import uuid for generating unique project names
from process_flight_log import extract_filtered_coordinates
from triangulate import cluster_annotations
import math


# Set Pillow's maximum image pixel limit to a higher value
Image.MAX_IMAGE_PIXELS = 250000000



app = Flask(__name__)
UPLOAD_DIR = 'uploads'
MOSAIC_DIR = 'mosaics'
ODM_PROJECTS_BASE_DIR = 'odm_projects'
ANNOTATED_OUTPUT_DIR = 'annotated_images_output'

# Initialize your model (ensure inference.py and API key are correctly set up)
model = inference.get_model("self-trained-garbage-detection/6", api_key="OoQrsMuOLaZZkrcxNM9q")

# The image_coordinates are now solely imported from coordinates.py
# print(f"DEBUG: image_coordinates list has {len(image_coordinates)} entries.") # This print will be outside the function scope, better place it inside generate_heatmap or upload_images for clarity

@app.route('/')
def index():
    """Serves the main HTML page for the application."""
    return send_from_directory('.', 'index.html')

@app.route('/mosaic/<filename>')
def get_mosaic(filename):
    """Serves the generated mosaic images (now including .png) from the MOSAIC_DIR."""
    return send_from_directory(MOSAIC_DIR, filename)

@app.route(f'/{ANNOTATED_OUTPUT_DIR}/<filename>')
def get_annotated_image(filename):
    """Serves the generated annotated images from the ANNOTATED_OUTPUT_DIR."""
    return send_from_directory(ANNOTATED_OUTPUT_DIR, filename)


@app.route('/upload', methods=['POST'])
def upload_images():
    processed_image_urls = []
    print("📥 Received image upload request.")
    uploaded_files = request.files.getlist('images')
    uploaded_csv = request.files.get('csv_file')
    
    
    object_counts = []
    image_count_map = [] # To store (filename, count) pairs
    

    # Ensure necessary directories exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(MOSAIC_DIR, exist_ok=True)
    os.makedirs(ODM_PROJECTS_BASE_DIR, exist_ok=True)
    os.makedirs(ANNOTATED_OUTPUT_DIR, exist_ok=True)

    # Clean up old data in all relevant directories
    for folder in [UPLOAD_DIR, MOSAIC_DIR, ANNOTATED_OUTPUT_DIR, ODM_PROJECTS_BASE_DIR]:
        if os.path.exists(folder):
            for filename_or_dir in os.listdir(folder):
                path_to_delete = os.path.join(folder, filename_or_dir)
                try:
                    if os.path.isfile(path_to_delete):
                        os.remove(path_to_delete)
                    elif os.path.isdir(path_to_delete):
                        shutil.rmtree(path_to_delete)
                    print(f"🧹 Cleaned up old data: {path_to_delete}")
                except Exception as e:
                    print(f"⚠️ Couldn't delete old data {path_to_delete}: {e}")
            

    filtered_csv_data = extract_filtered_coordinates(uploaded_csv)
    
    i = -1
    for file in uploaded_files:
        i += 1
        try:
            image_bytes = file.read()
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, 'wb') as f:
                f.write(image_bytes)
            print(f"✅ Saved uploaded original image: {file.filename}")

            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None or image.shape[0] == 0 or image.shape[1] == 0:
                print(f"Skipping empty or invalid image: {file.filename}")
                continue

            results = model.infer(image)[0]
            detections = sv.Detections.from_inference(results)
            count = len(detections)
            object_counts.append(count)
            print(f"📸 Processed {file.filename}: found {count} objects.")
            
            annotation_boxes = []
            for box in detections.xyxy:
                x_min, y_min, x_max, y_max = map(int, box)
                annotation_boxes.append({
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                })


            box_annotator = sv.BoxAnnotator(thickness=4)
            label_annotator = sv.LabelAnnotator(text_scale=1.5, text_thickness=3)
            annotated = box_annotator.annotate(scene=image, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections)

            annotated_filename = f"annotated_{uuid.uuid4().hex}_{file.filename.replace('.', '_')}.png"
            annotated_save_path = os.path.join(ANNOTATED_OUTPUT_DIR, annotated_filename)
            cv2.imwrite(annotated_save_path, annotated)

            processed_image_urls.append({
                "filename": file.filename,
                "url": f"/{ANNOTATED_OUTPUT_DIR}/{annotated_filename}",
                "drone_lat": filtered_csv_data[i][0],
                "drone_lng": filtered_csv_data[i][1],
                "altitude_ft": filtered_csv_data[i][2],
                "heading_deg": filtered_csv_data[i][3],
                "bounding_boxes": annotation_boxes,
            })
            
            print("🗂️ processed_image_urls", i, processed_image_urls[i])
            # Append filename and count for heatmap generation
            image_count_map.append((file.filename, count))

        except Exception as e:
            print(f"❌ Error processing {file.filename}: {e}")
            traceback.print_exc()
    
    # Call run_odm_stitching to get the actual mosaic data
    mosaic_data = run_odm_stitching()

    global clusterCenters
    clusterCenters = []
    
    global mapShapePoints
    mapShapePoints = []
    clusters = cluster_annotations(processed_image_urls)
    # Pass image_count_map (filenames and counts) and image_coordinates to generate_heatmap
    # You can adjust marker_radius and blur_strength here for desired visual effect
    heatmap_filename = generate_heatmap(
        clusters,
        filtered_csv_data=filtered_csv_data,
        grid_size=3840, #4k resolution
        marker_radius=30,
        blur_strength=25
    )


    return jsonify({
        "annotated_images_meta": processed_image_urls,
        "mosaic": mosaic_data,
        "heatmap": heatmap_filename,
        "object_counts": object_counts,
        "clusterCenters": clusterCenters,
        "mapShapePoints": mapShapePoints,
    })

def run_odm_stitching():
    """
    Executes OpenDroneMap via Docker to stitch images into an orthophoto.
    It expects a TIFF output by default and then converts it to PNG for web display,
    while keeping the full-resolution TIFF available for download.
    """
    print("🧵 Starting OpenDroneMap processing...")
    try:
        odm_image_dir = os.path.abspath(UPLOAD_DIR)
        odm_project_base_dir = os.path.abspath(ODM_PROJECTS_BASE_DIR)
        project_name = str(uuid.uuid4())
        odm_specific_project_output_dir = os.path.join(odm_project_base_dir, project_name)

        project_images_dir_on_host = os.path.join(odm_specific_project_output_dir, "images")
        os.makedirs(project_images_dir_on_host, exist_ok=True)
        print(f"Created ODM project images directory: {project_images_dir_on_host}")

        for img_file_name in os.listdir(odm_image_dir):
            source_path = os.path.join(odm_image_dir, img_file_name)
            destination_path = os.path.join(project_images_dir_on_host, img_file_name)
            if os.path.isfile(source_path):
                shutil.copy(source_path, destination_path)
        print(f"Copied images from {odm_image_dir} to {project_images_dir_on_host}")

        command = [
            "docker", "run", "--rm",
            "-v", f"{odm_project_base_dir}:/datasets",
            "opendronemap/odm",
            project_name,
            "--project-path", "/datasets",
            "--orthophoto-resolution", "0.3" # This is a good compromise for quality and file size
        ]

        print(f"Executing ODM command: {' '.join(command)}")

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("\n--- ODM STDOUT ---")
        print(result.stdout)
        print("--- ODM STDERR ---")
        print(result.stderr)
        print("------------------\n")

        if result.returncode != 0:
            print("❌ ODM processing failed with return code:", result.returncode)
            print("Full ODM error output (if any):", result.stderr)
            print(f"Contents of {odm_image_dir}: {os.listdir(odm_image_dir) if os.path.exists(odm_image_dir) else 'Does not exist'}")
            print(f"Contents of {odm_project_base_dir}: {os.listdir(odm_project_base_dir) if os.path.exists(odm_project_base_dir) else 'Does not exist'}")
            return None

        final_map_path_in_odm_output_tif = os.path.join(odm_specific_project_output_dir, "odm_orthophoto", "odm_orthophoto.tif")

        if os.path.exists(final_map_path_in_odm_output_tif):
            print("✅ ODM orthophoto (TIFF) found. Attempting conversion to PNG...")
            
            original_tif_filename = f"odm_map_fullres_{project_name}.tif"
            web_png_filename = f"odm_map_webres_{project_name}.png"
            
            original_tif_path = os.path.join(MOSAIC_DIR, original_tif_filename)
            web_png_path = os.path.join(MOSAIC_DIR, web_png_filename)

            try:
                shutil.copy(final_map_path_in_odm_output_tif, original_tif_path)
                print(f"✅ Full-resolution TIFF saved to: {original_tif_path}")

                with Image.open(final_map_path_in_odm_output_tif) as img:
                    max_web_dimension = 7680

                    resized_img = img
                    if img.width > max_web_dimension or img.height > max_web_dimension:
                        if img.width > img.height:
                            new_width = max_web_dimension
                            new_height = int(img.height * (max_web_dimension / img.width))
                        else:
                            new_height = max_web_dimension
                            new_width = int(img.width * (max_web_dimension / img.height))
                        
                        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                        print(f"Resized mosaic from {img.width}x{img.height} to {new_width}x{new_height}")
                    else:
                        print(f"Mosaic is already smaller than {max_web_dimension}px, no resizing needed.")

                    if resized_img.mode not in ("RGB", "RGBA"):
                        resized_img = resized_img.convert("RGBA")
                    
                    resized_img.save(web_png_path, format="PNG")
                print(f"✅ Web-resolution PNG successfully created: {web_png_path}")
                
                return {
                    "web_png": os.path.basename(web_png_path),
                    "full_tif": os.path.basename(original_tif_path)
                }

            except Exception as convert_e:
                print(f"❌ Error during mosaic processing (resizing/converting): {convert_e}")
                traceback.print_exc()
                return None
        else:
            print("❌ ODM output (TIFF) not found at expected path:", final_map_path_in_odm_output_tif)
            if os.path.exists(odm_specific_project_output_dir):
                print(f"Contents of ODM project output dir ({odm_specific_project_output_dir}):")
                for root, dirs, files in os.walk(odm_specific_project_output_dir):
                    level = root.replace(odm_specific_project_output_dir, '').count(os.sep)
                    indent = ' ' * 4 * (level)
                    print(f'{indent}{os.path.basename(root)}/')
                    subindent = ' ' * 4 * (level + 1)
                    for f in files:
                        print(f'{subindent}{f}')
            return None

    except Exception as e:
        print("❌ Unexpected exception during ODM processing:", e)
        traceback.print_exc()
        return None


def generate_heatmap(clusters, filtered_csv_data, grid_size, marker_radius, blur_strength):
    print("🔥 Generating geospatial heatmap from clusters...")

    try:
        if not clusters:
            print("⚠️ No cluster data provided for heatmap generation. Returning None.")
            return None

        print("🗂️filtered_csv_data", filtered_csv_data)
        # Extract all lat/lngs from the clusters
        #all_coords = [(item['lat'], item['lng']) for cluster in clusters for item in cluster]
        all_coords = [(row[0], row[1]) for row in filtered_csv_data]

        # Calculate min/max lat/lng to define the bounding box of the map
        lats = [coord[0] for coord in all_coords]
        lngs = [coord[1] for coord in all_coords]

        # Find full coordinates (lat, lng) for each extreme, preserving the pair
        north = max(all_coords, key=lambda c: c[0])  # Highest latitude
        south = min(all_coords, key=lambda c: c[0])  # Lowest latitude
        east  = max(all_coords, key=lambda c: c[1])  # Highest longitude
        west  = min(all_coords, key=lambda c: c[1])  # Lowest longitude
        
        # Now check if the easternmost point is the same as north or south
        if east == north or east == south:
            filteredeast = [coord for coord in all_coords if coord != east]
            east = max(filteredeast, key=lambda c: c[1])
        
        # Now check if the westernmost point is the same as north or south
        if west == north or west == south:
            filteredwest = [coord for coord in all_coords if coord != west]
            west = min(filteredwest, key=lambda c: c[1])
            
            
        print("🌎north", north)
        print("🌎east", east)
        print("🌎south", south)
        print("🌎west", west)
        
        mapShapePoints.append({ "lat": north[0], "lng": north[1] })
        mapShapePoints.append({ "lat": east[0],  "lng": east[1]  })
        mapShapePoints.append({ "lat": south[0], "lng": south[1] })
        mapShapePoints.append({ "lat": west[0],  "lng": west[1]  })
        mapShapePoints.append({ "lat": north[0], "lng": north[1] })  # to close the shape

        # Also extract lat/lng values if needed
        max_lat = north[0]
        min_lat = south[0]
        max_lng = east[1]
        min_lng = west[1]


        print(f"📍 Heatmap Bounding Box: Lat Range ({min_lat:.6f}, {max_lat:.6f}), Lng Range ({min_lng:.6f}, {max_lng:.6f})")

        heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

        def latlng_to_grid_coords(lat, lng):
            lat_range = max_lat - min_lat
            lng_range = max_lng - min_lng

            y = grid_size // 2 if lat_range == 0 else int(((max_lat - lat) / (lat_range + 1e-9)) * (grid_size - 1))
            x = grid_size // 2 if lng_range == 0 else int(((lng - min_lng) / (lng_range + 1e-9)) * (grid_size - 1))

            return max(0, min(y, grid_size - 1)), max(0, min(x, grid_size - 1))

        num_clusters = 0
        for cluster in clusters:
            if not cluster:
                continue
            cluster_center_lat = np.mean([item['lat'] for item in cluster])
            cluster_center_lng = np.mean([item['lng'] for item in cluster])
            y_grid, x_grid = latlng_to_grid_coords(cluster_center_lat, cluster_center_lng)
        
            intensity = float(len(cluster)) * 1
            
            clusterCenters.append([cluster_center_lat, cluster_center_lng, intensity])
            
            cv2.circle(heatmap, (x_grid, y_grid), marker_radius, intensity, thickness=cv2.FILLED)

            num_clusters += 1
            
        print(f"✅ Processed {num_clusters} clusters for heatmap.")
        print(f"📊 Heatmap data (before blur): min={heatmap.min():.2f}, max={heatmap.max():.2f}, non-zero count={np.count_nonzero(heatmap)}")

        if blur_strength > 0 and blur_strength % 2 == 1:
            heatmap = cv2.GaussianBlur(heatmap, (blur_strength, blur_strength), 0)
            print(f"🌀 Applied Gaussian blur with strength {blur_strength}.")

        if heatmap.max() > 0:
            normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            normalized_heatmap = np.uint8(normalized_heatmap)
        else:
            normalized_heatmap = np.zeros((grid_size, grid_size), dtype=np.uint8)

        colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
        
        # --- Draw white lines connecting bounding box corners in order: north → east → south → west → north ---
        north = latlng_to_grid_coords(*north)
        east = latlng_to_grid_coords(*east)
        south = latlng_to_grid_coords(*south)
        west = latlng_to_grid_coords(*west)

        # Define polygon points
        corner_points = [north, east, south, west, north]

        # Draw lines between the corners
        for i in range(len(corner_points) - 1):
            pt1 = (corner_points[i][1], corner_points[i][0])   # x, y
            pt2 = (corner_points[i+1][1], corner_points[i+1][0]) # x, y
            cv2.line(colored_heatmap, pt1, pt2, color=(255, 255, 255), thickness=2)


        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
        save_path = os.path.join(MOSAIC_DIR, heatmap_filename)
        cv2.imwrite(save_path, colored_heatmap)
        print(f"✅ Heatmap saved: {save_path}")

        return heatmap_filename

    except Exception as e:
        print("❌ Error generating heatmap:", e)
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("🚀 Starting C.A.V.A. backend...")
    app.run(port=5500, debug=True)