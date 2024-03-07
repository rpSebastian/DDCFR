import logging

from sacred import Experiment

ex = Experiment("default")
logger = logging.getLogger("mylogger")
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(levelname).1s] %(filename)s:%(lineno)d - %(message)s ",
    "%Y-%m-%d %H:%M:%S",
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

ex.logger = logger
