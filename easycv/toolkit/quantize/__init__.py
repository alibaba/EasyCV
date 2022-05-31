try:
    from .quantize_utils import calib, quantize_config_check
except ImportError as e:
    print(e)
