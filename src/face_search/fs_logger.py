import logging
import datetime
import sys
import os

def logger_init(logger_fname=None, level = logging.INFO):
    if logger_fname is None:
        logger_root = os.path.curdir 
        logger_root = os.path.join(logger_root, 'runs')
        os.makedirs(logger_root,exist_ok=True)
        now = datetime.datetime.now()
        method = list(filter(lambda x:".py" in x, sys.argv))
        if len(method):
            method= os.path.split(method[-1])[-1]
        else:
            method = 'log'
        filename = f'{now:%Y%m%d.%H%M%S}.{method}.log'
        logger_fname = os.path.join(logger_root, filename)

    logger = logging.getLogger()
    # get the root logger
    stdformattter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s %(message)s', datefmt='%m-%d %H:%M:%S')
    fformatter = logging.Formatter('%(asctime)s %(filename)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    logging.basicConfig(level=level, format='%(levelname)-8s %(message)s')

    # Add a handler to log to stdout
    stdout_handler = logger.handlers[0] #logging.StreamHandler()
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(stdformattter)
    #logger.addHandler(stdout_handler)

    # Add a handler to log to a file
    file_handler = logging.FileHandler(logger_fname)
    file_handler.setLevel(level)
    file_handler.setFormatter(fformatter)
    logger.addHandler(file_handler)



