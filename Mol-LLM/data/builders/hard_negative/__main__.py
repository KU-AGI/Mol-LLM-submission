import argparse

from .materialize import materialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Canonical DatasetDict directory')
    parser.add_argument('--pairs', required=True, help='Fixed TRAIN-index selections')
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    result = materialize(args.source, args.pairs, args.output, workers=args.workers)
    print(f"Saved {result['rows']} training rows to {args.output}")


if __name__ == '__main__':
    main()
