import argparse
from pathlib import Path

from dingo.gw.result import Result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--label", type=str)
    parser.add_argument("--extension", type=str)
    parser.add_argument("--result", type=str, nargs="+")
    parser.add_argument("--merge", action="store_true")

    args = parser.parse_args()

    if args.merge:
        print(f"Merging {len(args.result)} parts into complete Result.")
        sub_results = []
        for file_name in args.result:
            sub_results.append(Result(file_name=file_name))

        result = Result.merge(sub_results)
        result.print_summary()

        output_file = Path(args.outdir) / (args.label.replace("_merge", "") + ".hdf5")
        result.to_file(output_file)
