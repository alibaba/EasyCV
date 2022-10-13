import configparser
import os

import jinja2


def try_parse(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def parse_config(config_path):
    """config_path is the path.

    such as :
    config_path content:
        val/img_tag_tags_mean_average_precision=1
    Returns dict:
        {"val/img_tag_tags_mean_average_precision":1}
    """
    assert os.path.exists(config_path)
    config = {}
    with open(config_path, 'r') as fin:
        for line_str in fin:
            line_str = line_str.strip()
            if len(line_str) == 0:
                continue
            if line_str[0] == '#':
                continue
            if '=' in line_str:
                tmp_id = line_str.find('=')
                key = line_str[:tmp_id].strip()
                val = try_parse(line_str[(tmp_id + 1):].strip())
                config[key] = val
    return config


class MyConf(configparser.ConfigParser):
    # the origin configParser try to convert the key to lower
    def optionxform(self, optionstr):
        return optionstr

    def as_dict(self):
        dict_section = dict(self._sections)
        for key in dict_section:
            dict_section[key] = dict(dict_section[key])

            # ini val is string, so need the parse it
            for sub_key in dict_section[key]:
                dict_section[key][sub_key] = try_parse(
                    dict_section[key][sub_key])
        return dict_section


def parse_ini(file_path):
    config = MyConf()
    config.read(file_path, encoding='utf8')
    dict_section = config.as_dict()
    return dict_section


def render_config(filepath):
    dir = os.path.dirname(filepath)
    filename = 'render_' + os.path.basename(filepath)
    with open(filepath, 'r') as f:
        context = f.read()
        template = jinja2.Template(context)
        config = template.render(exp_id='1')

    dst_filepath = os.path.join(dir, filename)
    print('write the render config from ', filepath, 'to ', dst_filepath)

    with open(dst_filepath, 'w') as f:
        f.write(config)
    return dst_filepath
