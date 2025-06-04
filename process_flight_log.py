import pandas as pd

def extract_filtered_coordinates(csv_file):
    required_columns = ['latitude', 'longitude', 'height_sonar(feet)', 'compass_heading(degrees)', 'isPhoto']

    try:
        # Ensure we're working with a file-like object that can be reset
        if hasattr(csv_file, 'seek'):
            csv_file.seek(0)
        
        # Try reading without skipping
        df = pd.read_csv(csv_file, sep=";")
        df.columns = df.columns.str.strip()

        # Check if all required columns are present
        if not all(col in df.columns for col in required_columns):
            # Reset stream to beginning before re-reading
            if hasattr(csv_file, 'seek'):
                csv_file.seek(0)

            # Retry with skipping the first row
            df = pd.read_csv(csv_file, sep=";", skiprows=1)
            df.columns = df.columns.str.strip()

        # Final validation
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        # Filter and return results
        filtered_df = df[df['isPhoto'] == 1][[
            'latitude', 'longitude', 'height_sonar(feet)', 'compass_heading(degrees)'
        ]]
        data = list(filtered_df.itertuples(index=False, name=None))
        print(f"✅ Extracted {len(data)} rows with isPhoto == 1.")
        return data

    except Exception as e:
        print(f"❌ Failed to process CSV: {e}")
        return []
