from .log import init_logger, pretty_print, print_once


def default(val, default_val):
    return val if val is not None else default_val
