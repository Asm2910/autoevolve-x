from pathlib import Path

from data_generation.validation import validate_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    """Validate generated project datasets."""

    validate_datasets(
        data_dir=DATA_DIR,
        start_date="2026-01-01",
        end_date="2026-06-30",
    )

if __name__ == "__main__":
    main()