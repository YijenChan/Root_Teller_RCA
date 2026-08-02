#encoding = utf-8

import logging


class Logger():
    def __init__(self, logname, loglevel=logging.DEBUG, loggername=None):
        """Create a file logger with an optional console handler."""
        # Create the logger.
        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(loglevel)
        # Create a handler that writes to the requested log file.
        fh = logging.FileHandler(logname)
        fh.setLevel(loglevel)
        if not self.logger.handlers:
            # Create a second handler for console output.
            ch = logging.StreamHandler()
            ch.setLevel(loglevel)
            formatter = logging.Formatter(
                '[%(levelname)s]%(asctime)s %(filename)s:%(lineno)d: %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def getlog(self):
        return self.logger
