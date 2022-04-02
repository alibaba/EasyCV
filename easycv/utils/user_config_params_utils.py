VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}


def check_value_type(replacement, original):
    """
    convert replacement's type to original's type,
    support converting str to int or float or list or tuple
    """
    original_type = type(original)
    replacement_type = type(replacement)

    if replacement_type == original_type:
        return replacement
    if (replacement is None and original_type in VALID_TYPES) or \
        (original is None and replacement_type in VALID_TYPES):
        return replacement

    try:
        if original_type in [list, tuple, bool]:
            replacement = eval(replacement)
        else:
            replacement = original_type(replacement)
    except:
        raise TypeError(
            f'cannot convert {replacement_type} to {original_type}')

    return replacement
