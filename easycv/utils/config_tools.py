import copy
import os.path as osp
import platform
import sys
import tempfile
import warnings
from importlib import import_module

from mmcv import Config, import_modules_from_strings

import easycv
from easycv.file import io
from easycv.framework.errors import IOError, KeyError, ValueError
from .user_config_params_utils import check_value_type

if platform.system() == 'Windows':
    import regex as re
else:
    import re


def traverse_replace(d, key, value):
    if isinstance(d, (dict, Config)):
        for k, v in d.items():
            if k == key:
                d[k] = value
            else:
                traverse_replace(v, key, value)
    elif isinstance(d, (list, tuple, set)):
        for v in d:
            traverse_replace(v, key, value)


class WrapperConfig(Config):
    """A facility for config and config files.
    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.
    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    @staticmethod
    def _substitute_predefined_vars(filename,
                                    temp_config_name,
                                    first_order_params=None):
        """
        Override Config._substitute_predefined_vars.
        Supports first-order parameter reuse to avoid rebuilding custom config.py templates.
        Args:
            filename (str): Original script file.
            temp_config_name (str): Template script file.
            first_order_params (dict): first-order parameters.
        Returns:
            No return value.
        """
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)

        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            left_match, right_match = '{([', '])}'
            match_list = []
            line_list = []
            first_order_params_traced = None
            for line in f:
                # Push and pop control regular item matching
                if not line.strip().startswith('#'):
                    for single_str in line:
                        if single_str in left_match:
                            match_list.append(single_str)
                        if single_str in right_match:
                            match_list.pop()
                match_length = len(match_list)

                key = line.split('=')[0].strip()

                # Check whether it is a first-order parameter
                if first_order_params_traced is None and len(key) > 0:
                    first_order_params_traced = key

                if first_order_params_traced is not None:
                    if first_order_params_traced in first_order_params:
                        if match_length == 0:
                            value = first_order_params[
                                first_order_params_traced]
                            # repr() is used to convert the data into a string form (in the form of a Python expression) suitable for the interpreter to read
                            line = ' '.join(
                                [first_order_params_traced, '=',
                                 repr(value)]) + '\n'
                            line_list.append(line)
                            first_order_params_traced = None
                    else:
                        line_list.append(line)
                        if match_length == 0:
                            first_order_params_traced = None

            config_file = ''.join(line_list)

        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)


def check_base_cfg_path(base_cfg_name='configs/base.py',
                        father_cfg_name=None,
                        easycv_root=None):
    """
    Concatenate paths by parsing path rules.
    for example(pseudo-code):
        1. 'configs' in base_cfg_name or 'benchmarks' in base_cfg_name:
        base_cfg_name = easycv_root + base_cfg_name
        2. 'configs' not in base_cfg_name and 'benchmarks' not in base_cfg_name:
        base_cfg_name = father_cfg_name + base_cfg_name
    """
    parse_base_cfg = base_cfg_name.split('/')
    if parse_base_cfg[0] == 'configs' or parse_base_cfg[0] == 'benchmarks':
        if easycv_root is not None:
            base_cfg_name = osp.join(easycv_root, base_cfg_name)
    else:
        if father_cfg_name is not None:
            _parse_base_path_list = base_cfg_name.split('/')
            parse_base_path_list = copy.deepcopy(_parse_base_path_list)
            parse_ori_path_list = father_cfg_name.split('/')
            parse_ori_path_list.pop()
            for filename in _parse_base_path_list:
                if filename == '.':
                    parse_base_path_list.pop(0)
                elif filename == '..':
                    parse_base_path_list.pop(0)
                    parse_ori_path_list.pop()
                else:
                    break
            base_cfg_name = '/'.join(parse_ori_path_list +
                                     parse_base_path_list)

    return base_cfg_name


# Read config without __base__
def mmcv_file2dict_raw(filename, first_order_params=None):
    fileExtname = osp.splitext(filename)[1]
    if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
        raise IOError('Only py/yml/yaml/json type are supported now!')

    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix=fileExtname)
        if platform.system() == 'Windows':
            temp_config_file.close()
        temp_config_name = osp.basename(temp_config_file.name)
        if first_order_params is not None:
            WrapperConfig._substitute_predefined_vars(filename,
                                                      temp_config_file.name,
                                                      first_order_params)
        else:
            Config._substitute_predefined_vars(filename, temp_config_file.name)
        if filename.endswith('.py'):
            temp_module_name = osp.splitext(temp_config_name)[0]
            sys.path.insert(0, temp_config_dir)
            Config._validate_py_syntax(filename)
            mod = import_module(temp_module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
            # delete imported module
            del sys.modules[temp_module_name]
        elif filename.endswith(('.yml', '.yaml', '.json')):
            import mmcv
            cfg_dict = mmcv.load(temp_config_file.name)
        # close temp file
        temp_config_file.close()
    cfg_text = filename + '\n'
    with open(filename, 'r', encoding='utf-8') as f:
        # Setting encoding explicitly to resolve coding issue on windows
        cfg_text += f.read()

    return cfg_dict, cfg_text


# Reac config with __base__
def mmcv_file2dict_base(ori_filename,
                        first_order_params=None,
                        easycv_root=None):
    cfg_dict, cfg_text = mmcv_file2dict_raw(ori_filename, first_order_params)

    BASE_KEY = '_base_'
    if BASE_KEY in cfg_dict:
        base_filename = cfg_dict.pop(BASE_KEY)
        base_filename = base_filename if isinstance(base_filename,
                                                    list) else [base_filename]

        cfg_dict_list = list()
        cfg_text_list = list()
        for f in base_filename:
            base_cfg_path = check_base_cfg_path(
                f, ori_filename, easycv_root=easycv_root)
            _cfg_dict, _cfg_text = mmcv_file2dict_base(
                base_cfg_path, first_order_params, easycv_root=easycv_root)
            cfg_dict_list.append(_cfg_dict)
            cfg_text_list.append(_cfg_text)

        base_cfg_dict = dict()
        for c in cfg_dict_list:
            if len(base_cfg_dict.keys() & c.keys()) > 0:
                raise KeyError('Duplicate key is not allowed among bases')
            base_cfg_dict.update(c)

        base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = base_cfg_dict

        # merge cfg_text
        cfg_text_list.append(cfg_text)
        cfg_text = '\n'.join(cfg_text_list)

    return cfg_dict, cfg_text


def grouping_params(user_config_params):
    first_order_params, multi_order_params = {}, {}
    for full_key, v in user_config_params.items():
        key_list = full_key.split('.')
        if len(key_list) == 1:
            first_order_params[full_key] = v
        else:
            multi_order_params[full_key] = v

    return first_order_params, multi_order_params


def adapt_pai_params(cfg_dict):
    """
    Args:
        cfg_dict (dict): All parameters of cfg.
    Returns:
        cfg_dict (dict): Add the cfg of export and oss.
    """
    # oss config
    cfg_dict['oss_sync_config'] = dict(
        other_file_list=['**/events.out.tfevents*', '**/*log*'])
    cfg_dict['oss_io_config'] = dict(
        ak_id='your oss ak id',
        ak_secret='your oss ak secret',
        hosts='oss-cn-zhangjiakou.aliyuncs.com',
        buckets=['your_bucket_2'])
    return cfg_dict


def init_path(ori_filename):
    easycv_root = osp.dirname(easycv.__file__)  # easycv package root path
    if not osp.exists(osp.join(easycv_root, 'configs')):
        if osp.exists(osp.join(osp.dirname(easycv_root), 'configs')):
            easycv_root = osp.dirname(easycv_root)
        else:
            raise ValueError('easycv root does not exist!')
    parse_ori_filename = ori_filename.split('/')
    if parse_ori_filename[0] == 'configs' or parse_ori_filename[
            0] == 'benchmarks':
        if osp.exists(osp.join(easycv_root, ori_filename)):
            ori_filename = osp.join(easycv_root, ori_filename)

    return ori_filename, easycv_root


# gen mmcv.Config
def mmcv_config_fromfile(ori_filename):
    ori_filename, easycv_root = init_path(ori_filename)

    cfg_dict, cfg_text = mmcv_file2dict_base(
        ori_filename, easycv_root=easycv_root)

    if cfg_dict.get('custom_imports', None):
        import_modules_from_strings(**cfg_dict['custom_imports'])

    return Config(cfg_dict, cfg_text=cfg_text, filename=ori_filename)


def pai_config_fromfile(ori_filename,
                        user_config_params=None,
                        model_type=None):
    ori_filename, easycv_root = init_path(ori_filename)

    if user_config_params is not None:
        # grouping params
        first_order_params, multi_order_params = grouping_params(
            user_config_params)
    else:
        first_order_params, multi_order_params = None, None

    # replace first-order parameters
    cfg_dict, cfg_text = mmcv_file2dict_base(
        ori_filename, first_order_params, easycv_root=easycv_root)

    # export config
    if cfg_dict.get('export', None) is None:
        cfg_dict['export'] = dict(export_neck=True)
        cfg_dict['checkpoint_sync_export'] = True

    # Add export and oss ​​related configuration to adapt to pai platform
    if model_type:
        cfg_dict = adapt_pai_params(cfg_dict)

    if cfg_dict.get('custom_imports', None):
        import_modules_from_strings(**cfg_dict['custom_imports'])

    cfg = Config(cfg_dict, cfg_text=cfg_text, filename=ori_filename)

    # replace multi-order parameters
    if multi_order_params:
        cfg.merge_from_dict(multi_order_params)
    return cfg


# get the true value for ori_key in cfg_dict
def get_config_class_value(cfg_dict, ori_key, dict_mem_helper):
    if ori_key in dict_mem_helper:
        return dict_mem_helper[ori_key]

    k_list = ori_key.split('.')
    t = cfg_dict
    for at_k in k_list:
        if isinstance(t, (list, tuple)):
            t = t[int(at_k)]
        else:
            t = t[at_k]
    dict_mem_helper[ori_key] = t
    return t


def config_dict_edit(ori_cfg_dict, cfg_dict, reg, dict_mem_helper):
    """
    edit ${configs.variables} in config dict to solve dependicies in config
    ori_cfg_dict: to find the true value of ${configs.variables}
    cfg_dict: for find leafs of dict by recursive
    reg: Regular expression pattern for find all ${configs.variables} in leafs of dict
    dict_mem_helper: to store the true value of ${configs.variables} which have been found
    """
    if isinstance(cfg_dict, dict):
        for key, value in cfg_dict.items():
            if isinstance(value, str):
                var_set = set(reg.findall(value))
                if len(var_set) > 0:
                    n_value = value
                    is_arithmetic_exp = True
                    is_exp = True
                    for var in var_set:
                        # get the true value for ori_key in cfg_dict
                        var_value = get_config_class_value(
                            ori_cfg_dict, var[2:-1], dict_mem_helper)

                        # while ${variables}, replace by true value directly
                        if var == value:
                            is_exp = False
                            cfg_dict[key] = var_value
                            break
                        else:
                            if isinstance(var_value, str):
                                # str expression, like '${data_root_path}/images/'
                                is_arithmetic_exp = False
                            # arithmetic expression, like '-${img_size}//2'
                            n_value = n_value.replace(var, str(var_value))
                    if is_exp:
                        cfg_dict[key] = eval(
                            n_value) if is_arithmetic_exp else n_value

            # recursively find the leafs of dict
            config_dict_edit(ori_cfg_dict, value, reg, dict_mem_helper)

    elif isinstance(cfg_dict, list):
        for i, value in enumerate(cfg_dict):
            if isinstance(value, dict):
                # recursively find the leafs of dict
                config_dict_edit(ori_cfg_dict, value, reg, dict_mem_helper)
            elif isinstance(value, list):
                # recursively find the leafs of list
                config_dict_edit(ori_cfg_dict, value, reg, dict_mem_helper)
            elif isinstance(value, str):
                var_set = set(reg.findall(value))
                if len(var_set) > 0:
                    n_value = value
                    is_arithmetic_exp = True
                    is_exp = True
                    for var in var_set:
                        # get the true value for ori_key in cfg_dict
                        var_value = get_config_class_value(
                            ori_cfg_dict, var[2:-1], dict_mem_helper)
                        # while '${variables}', replace by true value directly
                        if var == value:
                            # is not an expression, directly replace
                            is_exp = False
                            cfg_dict[i] = var_value
                            break
                        else:  # arithmetic expression
                            if isinstance(var_value, str):
                                # str expression, like '${data_root_path}/images/'
                                is_arithmetic_exp = False
                            # arithmetic expression, like '-${img_size}//2'
                            n_value = n_value.replace(var, str(var_value))
                    if is_exp:
                        cfg_dict[i] = eval(
                            n_value) if is_arithmetic_exp else n_value


def rebuild_config(cfg, user_config_params):
    """
    # rebuild config by user config params,
    modify config by user config params & replace ${configs.variables} by true value
    return: Config
    """
    print(user_config_params)
    assert len(user_config_params
               ) % 2 == 0, 'user_config_params must be setted as --key value'

    new_text = cfg.text + '\n\n#user config params\n'

    for ori_args_k, args_v in zip(user_config_params[0::2],
                                  user_config_params[1::2]):
        assert ori_args_k.startswith('--')
        args_k = ori_args_k[2:]
        t = cfg
        attr_list = args_k.split('.')
        # find the location of key
        for at_k in attr_list[:-1]:
            if isinstance(t, (list, tuple)):
                try:
                    t = t[int(at_k)]
                except:
                    raise IndexError('%s is out of index' % at_k)
            else:
                try:
                    t = getattr(t, at_k)
                except:
                    raise KeyError('%s is not in dict' % at_k)
        # replace the value of key bey user config params
        if isinstance(t, (list, tuple)):  # set value of list[index]
            # convert str_type to value_type
            formatted_v = check_value_type(args_v, t[int(attr_list[-1])])
            try:
                t[int(attr_list[-1])] = formatted_v
            except:
                raise IndexError('%s is out of index' % attr_list[-1])
        else:  # set value of dict[map]
            # convert str_type to value_type
            formatted_v = check_value_type(args_v, getattr(t, attr_list[-1]))
            try:
                setattr(t, attr_list[-1], formatted_v)
            except:
                raise KeyError('set %s is error' % attr_list[-1])

        new_text += '%s %s\n' % (ori_args_k, args_v)
    # reg example: '-${a.b.ca}+${a.b.ca}+${d.c}'
    reg = re.compile('\$\{[a-zA-Z_][a-zA-Z0-9_\.]*\}')
    dict_mem_helper = {}

    # edit ${configs.variables} in config dict to solve dependicies in config
    config_dict_edit(cfg._cfg_dict, cfg._cfg_dict, reg, dict_mem_helper)

    return Config(
        cfg_dict=cfg._cfg_dict, cfg_text=new_text, filename=cfg._filename)


# validate config for export, which may be from training config, disable pretrained
def validate_export_config(cfg):
    cfg_copy = copy.deepcopy(cfg)
    pretrained = getattr(cfg_copy.model, 'pretrained', None)
    if pretrained is not None:
        setattr(cfg_copy.model, 'pretrained', None)
    backbone = getattr(cfg_copy.model, 'backbone', None)
    if backbone is not None:
        pretrained = getattr(backbone, 'pretrained', None)
        if pretrained is not None:
            setattr(backbone, 'pretrained', False)
    return cfg_copy


CONFIG_TEMPLATE_ZOO = {

    # cls
    'CLASSIFICATION_RESNET':
    'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py',
    'CLASSIFICATION_RESNEXT':
    'configs/classification/imagenet/resnext/imagenet_resnext50-32x4d_jpg.py',
    'CLASSIFICATION_HRNET':
    'configs/classification/imagenet/hrnet/imagenet_hrnetw18_jpg.py',
    'CLASSIFICATION_VIT':
    'configs/classification/imagenet/vit/imagenet_vit_base_patch16_224_jpg.py',
    'CLASSIFICATION_SWINT':
    'configs/classification/imagenet/swint/imagenet_swin_tiny_patch4_window7_224_jpg.py',
    'CLASSIFICATION_M0BILENET':
    'configs/classification/imagenet/mobilenet/mobilenetv2.py',
    'CLASSIFICATION_INCEPTIONV4':
    'configs/classification/imagenet/inception/inceptionv4_b32x8_100e.py',
    'CLASSIFICATION_INCEPTIONV3':
    'configs/classification/imagenet/inception/inceptionv3_b32x8_100e.py',

    # metric learning
    'METRICLEARNING':
    'configs/metric_learning/imagenet_timm_softmaxbased_jpg.py',
    'MODELPARALLEL_METRICLEARNING':
    'configs/metric_learning/imagenet_timm_modelparallel_softmaxbased_jpg.py',

    # detection
    'YOLOX': 'configs/config_templates/yolox.py',
    'YOLOX_ITAG': 'configs/config_templates/yolox_itag.py',
    'YOLOX_ITAG_EASY':
    'configs/detection/yolox/yolox_s_8xb16_300e_coco_pai.py',
    'YOLOX_COCO_EASY': 'configs/detection/yolox/yolox_s_8xb16_300e_coco.py',
    'FCOS_ITAG_EASY': 'configs/detection/fcos/fcos_r50_torch_1x_pai.py',
    'FCOS_COCO_EASY': 'configs/detection/fcos/fcos_r50_torch_1x_coco.py',

    # segmentation
    'FCN_SEG': 'configs/segmentation/fcn/fcn_r50-d8_512x512_8xb4_60e_voc12.py',
    'UPERNET_SEG':
    'configs/segmentation/upernet/upernet_r50_512x512_8xb4_60e_voc12.py',
    'SEGFORMER_SEG': 'configs/segmentation/segformer/segformer_b5_coco.py',

    # ssl
    'MOCO_R50_TFRECORD': 'configs/config_templates/moco_r50_tfrecord.py',
    'MOCO_R50_TFRECORD_OSS':
    'configs/config_templates/moco_r50_tfrecord_oss.py',
    'MOCO_TIMM_TFRECORD': 'configs/config_templates/moco_timm_tfrecord.py',
    'MOCO_TIMM_TFRECORD_OSS':
    'configs/config_templates/moco_timm_tfrecord_oss.py',
    'SWAV_R50_TFRECORD': 'configs/config_templates/swav_r50_tfrecord.py',
    'SWAV_R50_TFRECORD_OSS':
    'configs/config_templates/swav_r50_tfrecord_oss.py',
    'MOBY_TIMM_TFRECORD_OSS':
    'configs/config_templates/moby_timm_tfrecord_oss.py',
    'DINO_TIMM': 'configs/config_templates/dino_timm.py',
    'DINO_TIMM_TFRECORD_OSS':
    'configs/config_templates/dino_timm_tfrecord_oss.py',
    'DINO_R50_TFRECORD_OSS':
    'configs/config_templates/dino_rn50_tfrecord_oss.py',
    'MAE': 'configs/config_templates/mae_vit_base_patch16.py',

    # edge models
    'YOLOX_EDGE': 'configs/config_templates/yolox_edge.py',
    'YOLOX_EDGE_ITAG': 'configs/config_templates/yolox_edge_itag.py',

    # pose
    'TOPDOWN_HRNET': 'configs/config_templates/topdown_hrnet_w48_udp.py',
    'TOPDOWN_LITEHRNET': 'configs/config_templates/topdown_litehrnet_30.py',

    # video_classification
    'X3D_XS': 'configs/video_recognition/x3d/x3d_xs.py',
    'X3D_M': 'configs/video_recognition/x3d/x3d_m.py',
    'X3D_L': 'configs/video_recognition/x3d/x3d_l.py',
    'VIDEO_SWIN_T': 'configs/video_recognition/swin/video_swin_tiny.py',
    'VIDEO_SWIN_S': 'configs/video_recognition/swin/video_swin_s.py',
    'VIDEO_SWIN_B': 'configs/video_recognition/swin/video_swin_b.py',
    'SWIN_BERT': 'configs/video_recognition/clipbert/clipbert_multilabel.py',
}
