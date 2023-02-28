# Copyright (c) Alibaba, Inc. and its affiliates.
from argparse import ArgumentParser

from easycv.predictors.mot_predictor import MOTPredictor


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, help='config file')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder or None')
    args = parser.parse_args()

    model = MOTPredictor(args.checkpoint, args.config, save_path=args.output, score_threshold=0.2)

    track_result_list = model(args.input)
    print(track_result_list)


if __name__ == '__main__':
    main()
