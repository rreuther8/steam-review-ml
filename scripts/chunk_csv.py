import argparse

from steam_recs.data.loaders import chunk_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk a large CSV into smaller CSV files."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Folder to save chunked CSVs",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Number of rows per chunk",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="Columns to keep",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help="Language to filter on (column 'language')",
    )

    args = parser.parse_args()

    chunk_csv(
        input_file=args.input_file,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        columns=args.columns,
        language=args.language,
    )
    print("DONE")


if __name__ == "__main__":
    main()

