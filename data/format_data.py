import argparse
from pathlib import Path

import pandas as pd


feature_mappings = {
    'left_eye_center': 0,
    'right_eye_center': 1,
    'left_eye_inner_corner': 2,
    'left_eye_outer_corner': 3,
    'right_eye_inner_corner': 4,
    'right_eye_outer_corner': 5,
    'left_eyebrow_inner_end': 6,
    'left_eyebrow_outer_end': 7,
    'right_eyebrow_inner_end': 8,
    'right_eyebrow_outer_end': 9,
    'nose_tip': 10,
    'mouth_left_corner': 11,
    'mouth_right_corner': 12,
    'mouth_center_top_lip': 13,
    'mouth_center_bottom_lip': 14,
}


def assign_class(row, target_class, class_mappings, no_class_index):
    has_class = pd.notna(row[target_class + '_x']) and pd.notna(row[target_class + '_y'])
    if has_class:
        return class_mappings[target_class]
    else:
        return no_class_index


def format_data(args):
    data_file = Path(args.data_file)
    save_path = Path(args.save_path) / (data_file.stem + '_formatted.csv')

    training = pd.read_csv(str(data_file))
    no_class_index = len(feature_mappings)

    for feature in feature_mappings:
        training[feature + '_cls'] = training.apply(lambda row: assign_class(row,
                                                                             feature,
                                                                             feature_mappings,
                                                                             no_class_index), axis=1)
        training[feature + '_x'] = training[feature + '_x'] / args.image_size
        training[feature + '_y'] = training[feature + '_y'] / args.image_size

    training.to_csv(str(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('detr_train')

    parser.add_argument('--data_file', default='data/facial_keypoints/training.csv')
    parser.add_argument('--save_path', default='data/facial_keypoints/')
    parser.add_argument('--image_size', type=int, default=96)
    args = parser.parse_args()

    print('Formatting data...')
    format_data(args)
    print('Done!')

