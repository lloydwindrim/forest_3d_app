import glob, os
import argparse


def main(args):
    dataset_path = args.dataset_path
    yolo_root_destination = os.path.join(args.yolo_root,'tmp_yolo','train_data')

    # Percentage of images to be used for the test set
    percentage_test = int(args.split);

    # Create and/or truncate train.txt and test.txt
    file_train = open(os.path.join('tmp_yolo','train.txt'), 'w')
    file_test = open(os.path.join('tmp_yolo','val.txt'), 'w')

    # Populate train.txt and val.txt
    counter = 1
    index_test = round(100 / percentage_test)
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test+1:
            counter = 1
            file_test.write(os.path.join(yolo_root_destination,title + '.jpg') + "\n")
        else:
            file_train.write(os.path.join(yolo_root_destination,title + '.jpg') + "\n")
            counter = counter + 1


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-d', '--dataset_path',
        help='The path of image dataset',
        required=True)
    argparser.add_argument(
        '-y', '--yolo_root',
        help='The path to yolo root directory',
        required=True)
    argparser.add_argument(
        '-s', '--split',
        help='The proportion of validation data',
        required=True)

    args = argparser.parse_args()

    main(args)