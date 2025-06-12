import pandas as pd
import csv
import io

def extract_filtered_coordinates(csv_file):
    required_columns = ['latitude', 'longitude', 'height_sonar(feet)', 'compass_heading(degrees)', 'isPhoto']

    try:
        # Reset and read whole content (supporting BytesIO or text streams)
        if hasattr(csv_file, 'seek'):
            csv_file.seek(0)
        content_bytes = csv_file.read()

        # Decode if in bytes
        if isinstance(content_bytes, bytes):
            content = content_bytes.decode('utf-8', errors='replace')
        else:
            content = content_bytes

        # Clean lines: remove blank or metadata lines
        lines = [line for line in content.splitlines() if line.strip() and not line.lower().startswith('record')]

        # Try to sniff from cleaned lines
        sample = "\n".join(lines[:10])  # Only first few lines
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except csv.Error:
            # Fallback: try ; first (DJI often uses it), then ,
            delimiter = ';' if ';' in sample else ','

        # Create a StringIO to pass into pandas
        csv_stream = io.StringIO("\n".join(lines))

        # Try reading
        df = pd.read_csv(csv_stream, delimiter=delimiter)
        df.columns = df.columns.str.strip()

        # If columns are missing, retry skipping one line (possible second header)
        if not all(col in df.columns for col in required_columns):
            csv_stream.seek(0)
            df = pd.read_csv(csv_stream, delimiter=delimiter, skiprows=1)
            df.columns = df.columns.str.strip()

        # Final check
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        # Filter and extract
        filtered_df = df[df['isPhoto'] == 1][[
            'latitude', 'longitude', 'height_sonar(feet)', 'compass_heading(degrees)'
        ]]
        data = list(filtered_df.itertuples(index=False, name=None))
        print(f"✅ Extracted {len(data)} rows with isPhoto == 1.")
        return data

    except Exception as e:
        print(f"❌ Failed to process CSV: {e}")
        return []
