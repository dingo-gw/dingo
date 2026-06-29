"""Write a fixed-GPS time-segments pickle for deterministic ASD generation."""
import argparse
import pickle

# Fixed O1 segment (GWOSC open data), clear of GW150914 (GPS 1126259462.4).
FIXED_GPS_START = 1126257000
SEGMENT_LENGTH = 1024  # must be >= asd_dataset_settings time_psd
DETECTORS = ("H1", "L1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output pickle path.")
    args = parser.parse_args()
    segments = {
        det: [(FIXED_GPS_START, FIXED_GPS_START + SEGMENT_LENGTH)]
        for det in DETECTORS
    }
    with open(args.out, "wb") as f:
        pickle.dump(segments, f)
    print(f"Wrote time segments {segments} to {args.out}")


if __name__ == "__main__":
    main()
