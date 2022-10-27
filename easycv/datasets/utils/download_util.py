# coding: utf-8
# Copyright (c) Alibaba, Inc. and its affiliates.
import os, glob, wget
from easycv.utils.constant import CACHE_DIR

# The location where downloaded data is stored


'''
    { key : value key: value, key: value, ..... } 
    parameter:
        key  : str
        value: tuple
            explain ：[links, cmd, condition, data_home, default]
                links: list , collection of data download links
                cmd: str, Data decompression command
                condition: bool, whether to create data_name path, need if True else not need
                data_home: The location where the data is stored after decompression,
                default: The default train or val file
'''
DATASETS = {

    "voc2007": (
                    ["http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"],
                    "tar -xvf ",
                    False,
                    "VOCdevkit/VOC2007"
                ),
    "voc2012": (
                    ["http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"],
                    "tar -xvf ",
                    False,
                    "VOCdevkit/VOC2012/"
                ),
    'coco2017': (
                    [
                        'http://images.cocodataset.org/zips/train2017.zip',
                        'http://images.cocodataset.org/zips/val2017.zip',
                        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                    ],
                    "unzip -d",
                    True,
                    "annotations",
                    dict(detection=dict(
                            train='instances_train2017.json',
                            val='instances_val2017.json',
                        ),
                        train_dataset='train2017',
                        val_dataset='val2017',
                        pose=dict(
                            train='person_keypoints_train2017.json',
                            val='person_keypoints_val2017.json'
                        )
                    ) # default
                ),
    'lvis': (
        [
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip',
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip',
            'http://images.cocodataset.org/zips/train2017.zip',
            'http://images.cocodataset.org/zips/val2017.zip'

        ],
        "unzip -d",
        True,
        'annotations',
        dict(train='lvis_v1_train.json', val='lvis_v1_val.json', train_dataset='train2017', val_dataset='val2017') # default
    )
}


class DownLoadDataFile(object):

    def __init__(self) -> None:

        '''
        data_name: download file of name
        dataset_home: data root path
        split: return train or val
        tmp_data_path: Name of the data root directory
        task: Distinguish between tasks [detection, cls, pose]
        '''

        self.data_name = None
        self.dataset_home = CACHE_DIR
        self.tmp_data_path = None
        self.split = 'train'
        self.task = 'detection'

    def get_voc_path(self, data_name, dataset_home=CACHE_DIR, split='train', task='detection'):
        self.init_Parameters_must_be(data_name, dataset_home, split, task)
        self.is_mkdir(self.dataset_home)
        self.voc_download()
        return self.return_voc_path()

    def get_coco_path(self, data_name, dataset_home=CACHE_DIR, split='train', task='detection'):
        self.init_Parameters_must_be(data_name, dataset_home, split, task)
        self.is_mkdir(self.dataset_home)
        self.coco_dowload()
        return self.return_coco_path()

    def init_Parameters_must_be(self, data_name, dataset_home=CACHE_DIR, split='train', task='detection'):
        self.data_name = data_name.lower()
        # Using a new path
        if dataset_home:
            self.dataset_home = dataset_home
        self.tmp_data_path =  DATASETS[self.data_name][3]
        self.split = split
        self.task = task
        assert self.split in ["train", 'val'], f"{self.split} not in [train, val]"
        assert self.data_name in DATASETS.keys(), f"{self.data_name} is not right down link"
        assert self.task in ['cls', 'detection', 'pose'], f"{self.task} is not in ['cls', 'detection', 'pose']"

    def voc_download(self):

        # Check whether the path exists
        if os.path.exists(os.path.join(self.dataset_home, self.tmp_data_path)):
            return self.return_voc_path()

        # Start the download
        down_list = self.check_file()

        # Began to unpack
        self.extract_files(down_list)

    def coco_dowload(self):
        # Check whether the path exists
        if os.path.exists(os.path.join(self.dataset_home, self.data_name.upper())):
            return self.return_coco_path()
        # Start the download
        down_list = self.check_file()
        # Start the download
        self.extract_files(down_list)
        # Path of normalization
        self.regularization_path('coco')

    def check_file(self):

        download_finished = list()
        for tmp_link in DATASETS[self.data_name][0]:
            file_name = wget.filename_from_url(tmp_link)
            download_finished.append(file_name)
            # Check whether the compressed package exists. If no, download the compressed package
            if not os.path.exists(os.path.join(self.dataset_home, file_name)):
                self.download_files(tmp_link, file_name)

            # The prevention of Ctrol + C
            if not os.path.exists(os.path.join(self.dataset_home, file_name)):
                exit()

        return download_finished

    def is_mkdir(self, data_path):
        os.makedirs(data_path, exist_ok=True)

    def check_path_exists(self, map_path):
        for value in map_path.values():
            assert os.path.exists(value), f"{value} is not exists"
        return map_path

    def download_files(self, link, file_name):
        try:
            print(f"{file_name} is start downlaod........")
            print(link)
            file_name = wget.download(link, out=self.dataset_home)
            print(f"{file_name} is download finished\n")
        except:
            print(f"{file_name} is download fail")
            exit()

    def extract_files(self,file_list):
        for file_name in file_list:
            if DATASETS[self.data_name][2]:
                self.save_dir = os.path.join(self.dataset_home, self.data_name.upper())
                os.makedirs(self.save_dir, exist_ok=True)
                if file_name.endswith('zip'):
                    cmd = f"{DATASETS[self.data_name][1]} {self.save_dir} {os.path.join(self.dataset_home, file_name)}"
                else:
                    cmd = f"{DATASETS[self.data_name][1]} {os.path.join(self.dataset_home, file_name)} -C {self.save_dir}"
            else:
                cmd = f"{DATASETS[self.data_name][1]} {os.path.join(self.dataset_home, file_name)} -C {self.dataset_home}"

            print("begin Unpack.....................")
            os.system(cmd)
            print("Unpack is finished.....................")

    def regularization_path(self, type='coco'):
        if type == 'coco':
            file_list = glob.glob(os.path.join(self.dataset_home, self.data_name, "*.json"))
            annotations_dir = os.path.join(self.save_dir, self.tmp_data_path)
            for tmp in file_list:
                os.makedirs(annotations_dir, exist_ok=True)
                cmd = f'mv {tmp} {annotations_dir}'
                os.system(cmd)

    def return_coco_path(self):

        root_path = os.path.join(self.dataset_home, self.data_name.upper())
        annotations_path = os.path.join(root_path, self.tmp_data_path)
        map_path = dict()
        map_path['ann_file'] = os.path.join(annotations_path, DATASETS[self.data_name][4][self.task][self.split])
        map_path['img_prefix'] =  os.path.join(root_path, DATASETS[self.data_name][4][self.split+'_dataset'])

        return self.check_path_exists(map_path)

    def return_voc_path(self):
        if self.split == "train":
            path = os.path.join(os.path.join(self.dataset_home, self.tmp_data_path), "ImageSets/Main/train.txt")
        else:
            path = os.path.join(os.path.join(self.dataset_home, self.tmp_data_path), "ImageSets/Main/val.txt")
        return self.check_path_exists({'path': path})
