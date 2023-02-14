import os
from shutil import copyfile

# You only need to change this line to your dataset download path
download_path = '/apsarapangu/disk2/yunji.cjy/Market1501'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
# -----------------------------------------
# query
query_path = download_path + '/query'
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = query_path + '/' + name
        dst_path = query_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# -----------------------------------------
# multi-query
query_path = download_path + '/gt_bbox'
# for dukemtmc-reid, we do not need multi-query
if os.path.isdir(query_path):
    query_save_path = download_path + '/pytorch/multi-query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

# -----------------------------------------
# gallery
gallery_path = download_path + '/bounding_box_test'
gallery_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

label_dict = dict()
# ---------------------------------------
# train_all
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train_all'
txt_base = download_path + '/pytorch/meta'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
if not os.path.isdir(txt_base):
    os.mkdir(txt_base)
train_txt_path = open(txt_base + '/train_all.txt', 'w')
label_map_path = open(txt_base + '/label_map.txt', 'w')

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if ID[0] not in label_dict:
            label_dict[ID[0]] = str(len(label_dict))
            label_map_path.write(ID[0] + ' ' + label_dict[ID[0]] + '\n')
        train_txt_path.write(dst_path + '/' + name + ' ' + label_dict[ID[0]] +
                             '\n')
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# ---------------------------------------
# train_val
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'
train_txt_path = open(txt_base + '/train.txt', 'w')
val_txt_path = open(txt_base + '/val.txt', 'w')
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        train_dst_path = train_save_path + '/' + ID[0]
        val_dst_path = val_save_path + '/' + ID[0]
        if ID[0] not in label_dict:
            label_dict[ID[0]] = str(len(label_dict))
            label_map_path.write(ID[0] + ' ' + label_dict[ID[0]] + '\n')
        if not os.path.isdir(val_dst_path):  # first image is used as val image
            os.mkdir(val_dst_path)
            copyfile(src_path, val_dst_path + '/' + name)
            val_txt_path.write(val_dst_path + '/' + name + ' ' +
                               label_dict[ID[0]] + '\n')
        else:
            if not os.path.isdir(train_dst_path):
                os.mkdir(train_dst_path)
            copyfile(src_path, train_dst_path + '/' + name)
            train_txt_path.write(train_dst_path + '/' + name + ' ' +
                                 label_dict[ID[0]] + '\n')
