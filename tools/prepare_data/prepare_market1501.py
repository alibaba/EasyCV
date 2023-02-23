import argparse
import os
from shutil import copyfile


def generate_query(download_path, save_path):
    # -----------------------------------------
    # query
    query_path = os.path.join(download_path, 'query')
    query_save_path = os.path.join(save_path, 'query')
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = os.path.join(query_path, name)
            dst_path = os.path.join(query_save_path, ID[0])
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, os.path.join(dst_path, name))


def generate_multi_query(download_path, save_path):
    # -----------------------------------------
    # multi-query
    query_path = os.path.join(download_path, 'gt_bbox')
    # for dukemtmc-reid, we do not need multi-query
    if os.path.isdir(query_path):
        query_save_path = os.path.join(save_path, 'multi-query')
        if not os.path.isdir(query_save_path):
            os.mkdir(query_save_path)

        for root, dirs, files in os.walk(query_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg':
                    continue
                ID = name.split('_')
                src_path = os.path.join(query_path, name)
                dst_path = os.path.join(query_save_path, ID[0])
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, os.path.join(dst_path, name))


def generate_gallery(download_path, save_path):
    # -----------------------------------------
    # gallery
    gallery_path = os.path.join(download_path, 'bounding_box_test')
    gallery_save_path = os.path.join(save_path, 'gallery')
    if not os.path.isdir(gallery_save_path):
        os.mkdir(gallery_save_path)

    for root, dirs, files in os.walk(gallery_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = os.path.join(gallery_path, name)
            dst_path = os.path.join(gallery_save_path, ID[0])
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, os.path.join(dst_path, name))


def generate_train_val(download_path, save_path):
    label_dict = dict()
    # ---------------------------------------
    # train_all
    train_path = os.path.join(download_path, 'bounding_box_train')
    train_save_path = os.path.join(save_path, 'train_all')
    txt_base = os.path.join(save_path, 'meta')
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
    if not os.path.isdir(txt_base):
        os.mkdir(txt_base)
    train_txt_path = open(os.path.join(txt_base, 'train_all.txt'), 'w')
    label_map_path = open(os.path.join(txt_base, 'label_map.txt'), 'w')

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = os.path.join(train_path, name)
            dst_path = os.path.join(train_save_path, ID[0])
            if ID[0] not in label_dict:
                label_dict[ID[0]] = str(len(label_dict))
                label_map_path.write(ID[0] + ' ' + label_dict[ID[0]] + '\n')
            train_txt_path.write(dst_path + '/' + name + ' ' +
                                 label_dict[ID[0]] + '\n')
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, os.path.join(dst_path, name))

    # ---------------------------------------
    # train_val
    train_path = os.path.join(download_path, 'bounding_box_train')
    train_save_path = os.path.join(save_path, 'train')
    val_save_path = os.path.join(save_path, 'val')
    train_txt_path = open(os.path.join(txt_base, 'train.txt'), 'w')
    val_txt_path = open(os.path.join(txt_base, 'val.txt'), 'w')
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
        os.mkdir(val_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = os.path.join(train_path, name)
            train_dst_path = os.path.join(train_save_path, ID[0])
            val_dst_path = os.path.join(val_save_path, ID[0])
            if ID[0] not in label_dict:
                label_dict[ID[0]] = str(len(label_dict))
                label_map_path.write(ID[0] + ' ' + label_dict[ID[0]] + '\n')
            if not os.path.isdir(
                    val_dst_path):  # first image is used as val image
                os.mkdir(val_dst_path)
                copyfile(src_path, os.path.join(val_dst_path, name))
                val_txt_path.write(val_dst_path + '/' + name + ' ' +
                                   label_dict[ID[0]] + '\n')
            else:
                if not os.path.isdir(train_dst_path):
                    os.mkdir(train_dst_path)
                copyfile(src_path, os.path.join(train_dst_path, name))
                train_txt_path.write(train_dst_path + '/' + name + ' ' +
                                     label_dict[ID[0]] + '\n')


def main(download_path):
    if not os.path.isdir(download_path):
        print('please change the download_path')

    save_path = download_path + '/pytorch'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    generate_gallery(download_path, save_path)
    generate_multi_query(download_path, save_path)
    generate_query(download_path, save_path)
    generate_train_val(download_path, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='prepare market1501 datasets')
    parser.add_argument(
        '--download-path',
        type=str,
        default='./data/Market1501',
        help='You only need to change this line to your dataset download path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args.download_path)
