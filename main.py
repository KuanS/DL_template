from utils.initLog import initLog


def main_function():
  logger = initLog('dev', '/home/kuan/DL_template')

  logger.critical('CRITICAL')
  logger.fatal('FATAL')
  logger.error('ERROR')
  logger.warning('WARNING')
  logger.info('INFO')
  logger.debug('DEBUG')


if __name__ == '__main__':
  main_function()
