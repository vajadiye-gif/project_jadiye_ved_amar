# ==============================================================================
# main.py
# Command-line entry point for training and inference.
#
# Usage:
#   python main.py --predict data/EOSINOPHIL/_0_207.jpg
#   python main.py --predict data/LYMPHOCYTE/img01.jpg data/NEUTROPHIL/img02.jpg
#   python main.py --train
# ==============================================================================

import argparse
from predict import classify_cells
from train   import train_model
from model   import build_model
import config


def main():
    parser = argparse.ArgumentParser(
        description='Blood Cell Classifier — Ved Amar Jadiye'
    )
    parser.add_argument(
        '--predict', nargs='+', metavar='IMAGE_PATH',
        help='Path(s) to image file(s) to classify'
    )
    parser.add_argument(
        '--train', action='store_true',
        help='Run the training loop'
    )
    args = parser.parse_args()

    if args.predict:
        predictions = classify_cells(args.predict)
        for path, label in zip(args.predict, predictions):
            print(f'{path}  →  {label}')

    elif args.train:
        model   = build_model(num_classes=config.NUM_CLASSES)
        history = train_model(model, num_epochs=config.epochs)
        print('Training complete.')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
