import timeit
from typing import Callable, Mapping, Any

import logging
import datetime
import os

import pathlib
__dir__ = pathlib.Path(__file__).parent

class MyFormatter(logging.Formatter):
    converter=datetime.datetime.fromtimestamp
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s

def measure_execute_time(func: Callable, args: Mapping[str, Any], loop: int):
    result = timeit.timeit(lambda: func(**args), number=loop)

    return result/loop

def build_logger(log_dir, log_name):
    log_dir = (__dir__/log_dir).__str__()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # set handler
    handler = logging.FileHandler(f"{log_dir}/{log_name}.log", encoding="utf-8")
    formatter = MyFormatter("%(asctime)s %(name)s %(message)s", datefmt='%d-%m-%Y %H:%M:%S.%f')
    handler.setFormatter(formatter)

    # create logger
    logger = logging.getLogger(log_name)
    logger.propagate = False
    
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger