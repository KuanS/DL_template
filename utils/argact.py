import argparse
import os
from .initLog import initLog

class CudaAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self.logger = initLog('ArgParser')
        super().__init__(option_strings, dest, **kwargs)
        
    def setCUDA(self, cuda_):
        self.logger.info('set CUDA_VISIBLE_DEVICES = {}'.format(cuda_))
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_
        
    def __call__(self, parser, namespace, values, option_string=None):
        CUDA = ','.join(list(map(lambda x: x.strip(), values.split(','))))
        setattr(namespace, self.dest, CUDA)
        self.setCUDA(CUDA)
        
class FileAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self.logger = initLog('ArgParser')
        super().__init__(option_strings, dest, **kwargs)

    def checkFile(self, fn):
        val = os.path.isfile(fn)
        if not val:
            self.logger.error('{}: No such file'.format(fn))
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        if self.checkFile(values):
            #fn = os.path.abspath(values)
            fn = values
        else:
            fn = ''
        setattr(namespace, self.dest, fn)

