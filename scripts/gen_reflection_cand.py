import os
import xml.etree.ElementTree as ET


CLASS_NAME1 = 'person'
CLASS_NAME2 = 'cat'
# the pascal dataset can be found from here: https://pjreddie.com/projects/pascal-voc-dataset-mirror/
PASCAL_ROOT = '<your pascal root>/datasets'
OUTPUT_DIR = '<your out root>/backdoor_dataset'
NUM_DATASET = 1600

sets = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def check_dir(s_dir):
    if not os.path.exists(s_dir):
        os.makedirs(s_dir)


def get_valid_object(size, box, threshold=0.1):
    w = box[1] - box[0]
    h = box[3] - box[2]

    area_box = w * h
    area_img = size[0] * size[1]
    ret_thres = area_box / area_img
    return ret_thres > threshold


def get_image_label_id(year, img_id):
    in_file = open(os.path.join(PASCAL_ROOT, 'VOCdevkit/VOC%s/Annotations/%s.xml' % (year, img_id)))
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    names = []
    name_ids = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        cls_id = classes.index(cls)

        xml_box = obj.find('bndbox')
        b = (float(xml_box.find('xmin').text), float(xml_box.find('xmax').text), float(xml_box.find('ymin').text),
             float(xml_box.find('ymax').text))
        bb = get_valid_object((w, h), b)

        if bb is True:
            names.append(cls)
            name_ids.append(cls_id)

    if CLASS_NAME1 in names and CLASS_NAME2 in names:
        names = []
    return names


num_class1 = 0
num_class2 = 0

dir_class1 = os.path.join(OUTPUT_DIR, CLASS_NAME1)
dir_class2 = os.path.join(OUTPUT_DIR, CLASS_NAME2)

check_dir(dir_class1)
check_dir(dir_class2)

for year, image_set in sets:
    image_ids = open(os.path.join(PASCAL_ROOT,
                                  'VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set))).read().strip().split()
    for image_id in image_ids:
        label_names = get_image_label_id(year, image_id)
        if CLASS_NAME1 in label_names and num_class1 < NUM_DATASET:
            num_class1 += 1
            os.symlink(os.path.join(PASCAL_ROOT, 'VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (year, image_id)),
                       os.path.join(dir_class1, '%s.jpg' % image_id))
        elif CLASS_NAME2 in label_names and num_class2 < NUM_DATASET:
            num_class2 += 1
            os.symlink(os.path.join(PASCAL_ROOT, 'VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (year, image_id)),
                       os.path.join(dir_class2, '%s.jpg' % image_id))
        else:
            pass

        print('Generating {}: {}/{}\t{}: {}/{}'.format(CLASS_NAME1, num_class1, NUM_DATASET,
                                                       CLASS_NAME2, num_class2, NUM_DATASET))
print('\nDone.')

