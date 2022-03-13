import logging
import re
import os
import datetime

__all__ = ['initLog']

CRITICAL_STYLE = '\033[1;91m' # bold, red
FATAL_STYLE = '\033[1;91m' # bold, red
ERROR_STYLE = '\033[3;31m' # italic, red
WARNING_STYLE = '\033[93m' # yellow
INFO_STYLE = '\033[92m' # green
DEBUG_STYLE = '\033[94m' # blue

_style_LUT = {
    'CRITICAL': CRITICAL_STYLE,
    'ERROR': ERROR_STYLE,
    'WARNING': WARNING_STYLE,
    'INFO': INFO_STYLE,
    'DEBUG': DEBUG_STYLE
}

_filter_LUT = {
    'CRITICAL': lambda rec: rec.levelname=='CRITICAL',
    'ERROR': lambda rec: rec.levelname=='ERROR',
    'WARNING': lambda rec: rec.levelname=='WARNING',
    'INFO': lambda rec: rec.levelname=='INFO',
    'DEBUG': lambda rec: rec.levelname=='DEBUG'    
}

def initLog(name, log_path=None):
    # get main logger
    logger_ = logging.getLogger(name)
    logger_.setLevel(logging.DEBUG)
    
    # clear handler
    logger_.handlers = []
    # base format
    fmt = '%(asctime)s[%(module)s][%(funcName)s][%(levelname)s]%(message)s'
    
    # set stream handler
    hdlr = []
    for k, v in _style_LUT.items():
        hdlr = logging.StreamHandler()
        hdlr.set_name('stream_hdlr_{}'.format(k))
        hdlr.setFormatter(logging.Formatter('{}{}\033[0m'.format(v, fmt)))
        hdlr.addFilter(_filter_LUT[k])
        logger_.addHandler(hdlr)
            
    # set file handler
    if log_path:
        # case log_path is given
        if os.path.isdir(log_path):
            # case log_path is valid
            if os.access(log_path, os.W_OK):
                # case have access to write
                # set file name
                log_fn = 'Main.{}.log'.format(datetime.datetime.now().strftime('%y%m%d%H%M%S'))
                log_fullfn = '/'.join([log_path, log_fn])
                # set file handler
                f_hdlr = logging.FileHandler(log_fullfn)
                f_hdlr.set_name('file_hdlr')
                f_hdlr.setFormatter(logging.Formatter(fmt))
                # assign file handler
                logger_.addHandler(f_hdlr)
            else:
                logger_.warning('Cannot write to {} Ignore to save the log file.'.format(log_path))
        else:
            logger_.warning('{} is not a valid path. Ignore to save the log file.'.format(log_path))
    
    return logger_