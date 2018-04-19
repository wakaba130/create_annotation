##
# coding:utf-8
##

import argparse
import glob
import os
import shutil

import chainer
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import utils

from xml.etree import ElementTree
from xml.dom import minidom
from collections import namedtuple

IMAGE_SIZE = namedtuple('IMAGE_SIZE', ('width', 'height', 'depth'))

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default='voc0712')
    parser.add_argument('--output_dir', type=str, default='result1')
    parser.add_argument('--no_copy', help="Do not copy images", action="store_true")
    parser.add_argument('image_dir', type=str)
    return parser.parse_args()


def create_pascalVOC(full_name, img_size, labels, bbox, output_file_name):
    def prettify(elem):
        rough_string = ElementTree.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    top = ElementTree.Element('annotation')

    dir_name, file_name = os.path.split(full_name)
    folder = ElementTree.SubElement(top, 'folder')
    folder.text = str(dir_name)

    filename = ElementTree.SubElement(top, 'filename')
    filename.text = str(file_name)

    path = ElementTree.SubElement(top, 'path')
    path.text = str(full_name)

    source = ElementTree.SubElement(top, 'source')
    source.text = 'Unknown'
    # owner = ElementTree.SubElement(top, 'owner')

    size_s = ElementTree.SubElement(top, 'size')
    w = ElementTree.SubElement(size_s, 'width')
    w.text = str(img_size.width)
    h = ElementTree.SubElement(size_s, 'height')
    h.text = str(img_size.height)
    d = ElementTree.SubElement(size_s, 'depth')
    d.text = str(img_size.depth)

    seg = ElementTree.SubElement(top, 'segmented')
    seg.text = str(0)

    for c in range(len(labels)):
        object = ElementTree.SubElement(top, 'object')

        name = ElementTree.SubElement(object, 'name')
        name.text = str(labels[c])

        pose = ElementTree.SubElement(object, 'pose')
        pose.text = 'Unspecified'

        truncated = ElementTree.SubElement(object, 'truncated')
        truncated.text = str(1)

        difficult = ElementTree.SubElement(object, 'difficult')
        difficult.text = str(0)

        bboxElm = ElementTree.SubElement(object, 'bndbox')
        xmin = ElementTree.SubElement(bboxElm, 'xmin')
        xmin.text = str(int(bbox[c][1]))
        ymin = ElementTree.SubElement(bboxElm, 'ymin')
        ymin.text = str(int(bbox[c][0]))
        xmax = ElementTree.SubElement(bboxElm, 'xmax')
        xmax.text = str(int(bbox[c][3]))
        ymax = ElementTree.SubElement(bboxElm, 'ymax')
        ymax.text = str(int(bbox[c][2]))

    elm = prettify(top)
    with open(output_file_name, 'w') as fp:
        fp.write(elm)

def main():

    args = argparser()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print('output dir is exist.')
        exit()

    print('loading model...')
    chainer.config.train = False
    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    print('loading images...')
    jpg_list = glob.glob(os.path.join(args.image_dir,'*.jpg'))

    for i,jpg_name in enumerate(jpg_list):
        print('{}/{} {}'.format(i+1, len(jpg_list), jpg_name))
        # object detection
        img = utils.read_image(jpg_name, color=True)
        bboxes, labels, scores = model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]
        str_label = [voc_bbox_label_names[n] for n in label]

        # create_output_dir and name
        filename = os.path.split(jpg_name)[1]
        output_dir = os.path.split(jpg_name)[0].replace(args.image_dir,args.output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_name = os.path.join(output_dir, (os.path.splitext(filename)[0] + '.xml'))

        # output xml
        if args.no_copy:
            full_name = os.path.abspath(jpg_name)
        else:
            output_jpg_name = os.path.join(output_dir, filename)
            full_name = os.path.abspath(output_jpg_name)
            shutil.copyfile(jpg_name, output_jpg_name)

        img_size = IMAGE_SIZE(img.shape[1],img.shape[2],img.shape[0])
        create_pascalVOC(full_name, img_size, str_label, bbox, output_name)


if __name__ == '__main__':
    main()