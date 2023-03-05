import os
import yaml

from initLog import initLog

__all__ = ['CfgParser']

def CfgPrintAll(out_dict, logger):
    def CfgPrint(d, logger, ind=0):
        # d is a dict
        for k, v in d.items():
            if isinstance(v, dict):
                logger.info('{}{}:'.format(' '*ind, k))
                CfgPrint(v, logger, ind=ind+2)
            else:
                logger.info('{}{}: {}'.format(' '*ind, k, v))
    
    logger.info('{i:#^16} {c: ^16} {i:#^16}'.format(c='START CONFIG', i=''))    
    if isinstance(out_dict, dict):
        CfgPrint(out_dict, logger)
    else:
        CfgPrint({'DEFAULT_KEY': out_dict}, logger)
    logger.info('{i:#^16} {c: ^16} {i:#^16}'.format(c='END CONFIG', i=''))

def CfgParser(cfg_file, logger):
    def parse_config(con_):
        con = con_.copy()
        for k, v in con.items():
            if isinstance(v, dict):
                temp = parse_config(v)
                con[k] = temp
        return ConfigClass(con)
    
    cfg = None
    logger.info('parsing config file... {}'.format(cfg_file))
    try:
        with open(cfg_file, 'r') as f:
            #cfg = yaml.load(f.read(), Loader=yaml.CLoader)
            cfg = yaml.full_load(f.read())
        CfgPrintAll(cfg, logger)
        cfg = parse_config(cfg)
    except Exception as e:
        logger.error(e)
        print('')
        logger.error('failed to parse the config file')
        logger.error('Please check')
        cfg = None
    finally:
        return cfg
    
    
class ConfigClass():
    def __init__(self, d):
        self._data = d.copy()
            
    def __getattr__(self, x):
        return self._data.get(x, None)
    
    def __setattr__(self, k, v):
        if k == '_data':
            return super().__setattr__(k, v)
        else:
            return None

if __name__ == '__main__':
    logger = initLog('cfgParser_TEST')
    filepath = os.path.dirname(os.path.abspath(__file__))
#     filepath = '/'.join(__file__.split('/')[:-1])
#     print(filepath)
    cfg = CfgParser('{}/../config/config.yml'.format(filepath), logger)
