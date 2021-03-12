import argparse


def train(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser('detr_train')
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    train(args)

