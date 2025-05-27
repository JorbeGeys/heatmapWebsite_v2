import pandas as pd

def extract_filtered_coordinates(csv_file):
    """
    Extract coordinates from a semi-colon-delimited CSV with a titled first row and space-padded column headers.
    """
    try:
        df = pd.read_csv(csv_file, sep=";", skiprows=1)

        # Strip whitespace from column headers
        df.columns = df.columns.str.strip()

        required_columns = ['latitude', 'longitude', 'height_sonar(feet)', 'compass_heading(degrees)', 'isPhoto']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        filtered_df = df[df['isPhoto'] == 1][['latitude', 'longitude', 'height_sonar(feet)', 'compass_heading(degrees)']]
        
        data = list(filtered_df.itertuples(index=False, name=None))
        print(f"✅ Extracted {len(data)} rows with isPhoto == 1.")
        
        return data

    except Exception as e:
        print(f"❌ Failed to process CSV: {e}")
        return []
