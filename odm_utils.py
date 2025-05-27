

import os
import uuid
import subprocess

TEMP_DIR = "tempImgODM"
RESULTS_DIR = "results"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def process_odm_images(files):
    project_id = str(uuid.uuid4())
    image_folder = os.path.join(TEMP_DIR, project_id)
    result_folder = os.path.join(RESULTS_DIR, project_id)

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)

    # Save uploaded images
    for file in files:
        filepath = os.path.join(image_folder, file.filename)
        file.save(filepath)

    # Run ODM via Docker
    try:
        subprocess.run([
            "docker", "run", "-it", "--rm",
            "-v", f"{os.path.abspath(image_folder)}:/code/images",
            "-v", f"{os.path.abspath(result_folder)}:/code/odm_orthophoto",
            "opendronemap/odm"
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ODM processing failed: {e}")

    orthophoto_path = os.path.join(result_folder, "odm_orthophoto", "odm_orthophoto.tif")
    return {
        "message": "ODM processing complete.",
        "project_id": project_id,
        "orthophoto_path": orthophoto_path
    }
