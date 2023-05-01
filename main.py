import sys
import os
import argparse
import datetime
import logging
from utils.argact import CudaAction, FileAction
from utils.initLog import initLog
from utils.cfgParser import CfgParser
from utils.initModel import initModel
from utils.initFetcher import initFetcher

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)
    
def disable_logfile(logger):
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
            os.remove(h.baseFilename)
            logger.removeHandler(h)
        
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, required=True, action=FileAction,
                        help='path to solver file.')
    parser.add_argument('--config', type=str, required=True, action=FileAction,
                        help='path to config file.',)
    parser.add_argument('--cuda', type=str, default='-1', action=CudaAction,
                        help='If -1, use cpu; 2 for single GPU(2), '
                        '2,3,4 for multi GPUS(2,3,4). '
                        'default=-1')
    parser.add_argument('-no-log', action='store_true')
    return parser.parse_args()

def main():
    # parse arguments from CLI
    arg = arg_parser()
    
    # check if log file is needed
    # set logger
    if arg.no_log:
        logger = initLog('DL_log')
    else:
        logger = initLog('DL_log', log_path='./')
    
    # parse config file
    cfg = CfgParser(arg.config, logger)
    if cfg is None:
        logger.fatal('error getting config')
        return
    
    # init data fetcher
    fetcher = initFetcher(cfg)

    # init model
    model = initModel(cfg)

    # parse solver file
    # training and testing would be implemented in different solver
    #sol = 

    # run sol
    #solver(cfg, logger)
    
    
if __name__ == '__main__':
    main()

