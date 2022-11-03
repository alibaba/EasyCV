import copy
import os.path as osp
import platform
import sys
import tempfile
from argparse import Action
from importlib import import_module

from mmcv import Config, import_modules_from_strings

from easycv.framework.errors import IOError, KeyError, ValueError
from .user_config_params_utils import check_value_type

if platform.system() == 'Windows':
    import regex as re
else:
    import re


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple: The expanded list or tuple from the string.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


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


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


class EasyCVConfig(Config):

    @staticmethod
    def _substitute_predefined_vars(filename,
                                    temp_config_name,
                                    first_order_params=None):
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
            line_list = []
            for line in f:
                key = line.split('=')[0].strip()

                # replace first order params
                if first_order_params and key in first_order_params:
                    value = first_order_params[key]
                    if is_number(value):
                        line = ' '.join([key, '=', value])
                    else:
                        line = ' '.join([key, '=', repr(value)])

                line_list.append(line)
            config_file = '\n'.join(line_list)

        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)


# To find base cfg in 'easycv/configs/', base_cfg_name should be 'configs/xx/xx.py'
# TODO: reset the api, keep the same way as mmcv `Config.fromfile`
def check_base_cfg_path(base_cfg_name='configs/base.py', ori_filename=None):

    if base_cfg_name == '../../base.py':
        # To becompatible with previous config
        base_cfg_name = 'configs/base.py'

    base_cfg_dir_1 = osp.abspath(osp.dirname(
        osp.dirname(__file__)))  # easycv_package_root_path
    base_cfg_path_1 = osp.join(base_cfg_dir_1, base_cfg_name)
    print('Read base config from', base_cfg_path_1)
    if osp.exists(base_cfg_path_1):
        return base_cfg_path_1

    base_cfg_dir_2 = osp.dirname(base_cfg_dir_1)  # upper level dir
    base_cfg_path_2 = osp.join(base_cfg_dir_2, base_cfg_name)
    print('Read base config from', base_cfg_path_2)
    if osp.exists(base_cfg_path_2):
        return base_cfg_path_2

    # relative to ori_filename
    ori_cfg_dir = osp.dirname(ori_filename)
    base_cfg_path_3 = osp.join(ori_cfg_dir, base_cfg_name)
    base_cfg_path_3 = osp.abspath(osp.expanduser(base_cfg_path_3))
    if osp.exists(base_cfg_path_3):
        return base_cfg_path_3

    raise ValueError('%s not Found' % base_cfg_name)


# Read config without __base__
def mmcv_file2dict_raw(ori_filename, first_order_params=None):
    filename = osp.abspath(osp.expanduser(ori_filename))
    if not osp.isfile(filename):
        if ori_filename.startswith('configs/'):
            # read configs/config_templates/detection_oss.py
            filename = check_base_cfg_path(ori_filename)
        else:
            raise ValueError('%s and %s not Found' % (ori_filename, filename))

    fileExtname = osp.splitext(filename)[1]
    if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
        raise IOError('Only py/yml/yaml/json type are supported now!')

    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix=fileExtname)
        if platform.system() == 'Windows':
            temp_config_file.close()
        temp_config_name = osp.basename(temp_config_file.name)
        EasyCVConfig._substitute_predefined_vars(filename,
                                                 temp_config_file.name,
                                                 first_order_params)
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
def mmcv_file2dict_base(ori_filename, first_order_params=None):
    cfg_dict, cfg_text = mmcv_file2dict_raw(ori_filename, first_order_params)

    BASE_KEY = '_base_'
    if BASE_KEY in cfg_dict:
        # cfg_dir = osp.dirname(filename)
        base_filename = cfg_dict.pop(BASE_KEY)
        base_filename = base_filename if isinstance(base_filename,
                                                    list) else [base_filename]

        cfg_dict_list = list()
        cfg_text_list = list()
        for f in base_filename:
            base_cfg_path = check_base_cfg_path(f, ori_filename)
            _cfg_dict, _cfg_text = mmcv_file2dict_base(base_cfg_path,
                                                       first_order_params)
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


# gen mmcv.Config
def mmcv_config_fromfile(ori_filename, user_config_params=None):
    # grouping params
    first_order_params, multi_order_params = grouping_params(
        user_config_params)

    # replace first-order parameters
    cfg_dict, cfg_text = mmcv_file2dict_base(ori_filename, first_order_params)

    if cfg_dict.get('custom_imports', None):
        import_modules_from_strings(**cfg_dict['custom_imports'])

    cfg = Config(cfg_dict, cfg_text=cfg_text, filename=ori_filename)

    # replace multi-order parameters
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

    # detection
    'YOLOX': 'configs/config_templates/yolox.py',
    'YOLOX_ITAG': 'configs/config_templates/yolox_itag.py',

    # cls
    'CLASSIFICATION': 'configs/config_templates/classification.py',
    'CLASSIFICATION_OSS': 'configs/config_templates/classification_oss.py',
    'CLASSIFICATION_TFRECORD_OSS':
    'configs/config_templates/classification_tfrecord_oss.py',

    # metric learning
    'METRICLEARNING_TFRECORD_OSS':
    'configs/config_templates/metric_learning/softmaxbased_tfrecord_oss.py',
    'MODELPARALLEL_METRICLEARNING':
    'configs/config_templates/metric_learning/modelparallel_softmaxbased_tfrecord_oss.py',

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
}
