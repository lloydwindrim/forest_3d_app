import xml.etree.ElementTree as ET
import argparse
import os

# write your class names in your xml files
#classes = ["class_name"]


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def main(args):
    classes = args.class_names.split(',')

    annot_path = args.dataset_path

    dataset_path = os.path.split(annot_path)[0]

    filenames = os.listdir(annot_path)

    if not os.path.exists(os.path.join(dataset_path, "anno_yolo")):
        os.makedirs(os.path.join(dataset_path, "anno_yolo"))
    for filename in filenames:
        img_id = filename[:-4]
        file_path = os.path.join(annot_path, filename)
        in_file = open(file_path)
        out_file = open(os.path.join(dataset_path, "anno_yolo", img_id + ".txt"), "w")
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text

            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-d', '--dataset_path',
        help='The path of annotation dataset',
        required=True)
    argparser.add_argument(
        '-c', '--class_names',
        help='The class names e.g. "class1,class2,class3"',
        required=True)

    args = argparser.parse_args()

    main(args)
