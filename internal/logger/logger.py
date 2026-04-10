import logging

def init_logger(level, debug_mode):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level.upper(), format=fmt)
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)