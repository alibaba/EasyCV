try:
    from .quantize_utils import calib, quantize_config
except ImportError as e:
    print(e)
