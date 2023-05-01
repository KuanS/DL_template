# import fetcher
#from dataset import XXX as fetcher 

def initFetcher(cfg):
    return fetcher(cfg.DATASET)
