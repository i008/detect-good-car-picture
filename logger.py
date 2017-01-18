import logging
import sys

from settings import DEBUG


def get_logger(lvl=logging.DEBUG, log_file_name=None, logger_name='default', elk_host=None, elk_port=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(lvl)
    stdout_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if elk_host and elk_port:
        import logstash
        logstash_handler = logstash.LogstashHandler(elk_host, int(elk_port))
        logstash_handler.setLevel(lvl)
        logger.addHandler(logstash_handler)

    if log_file_name:
        hdlr = logging.FileHandler(log_file_name)
        hdlr.setFormatter(stdout_formatter)
        hdlr.setLevel(lvl)
        logger.addHandler(hdlr)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(lvl)
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)

    if not DEBUG:
        logger.disabled = True

    return logger

logger = get_logger()
