from pathlib import Path
from src.data_processing import process_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def run_pipeline():
    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("Pipeline ready.")
        print("No raw CSV dataset found in data/raw/.")
        return

    source = csv_files[0]
    destination = PROCESSED_DIR / f"{source.stem}_processed.csv"

    process_file(source, destination)

    print("Pipeline completed successfully.")
    print(f"Input: {source}")
    print(f"Output: {destination}")


if __name__ == "__main__":
    run_pipeline()
