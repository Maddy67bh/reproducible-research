from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned = cleaned.drop_duplicates()

    cleaned.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in cleaned.columns
    ]

    return cleaned


def process_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = load_csv(input_path)
    processed = basic_cleaning(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)

    return processed


if __name__ == "__main__":
    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("No CSV file found in data/raw/.")
        print("Add your research dataset to data/raw/ before running the full analysis.")
    else:
        source = csv_files[0]
        destination = PROCESSED_DIR / f"{source.stem}_processed.csv"

        process_file(source, destination)

        print(f"Processed: {source.name}")
        print(f"Saved to: {destination}")
