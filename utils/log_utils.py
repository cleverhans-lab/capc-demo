import os
import logging


def create_logger(save_path='', file_type='', level='debug'):
    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO
    else:
        raise Exception(f"Unsupported loggin level: {level}.")

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='a')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    logger = create_logger(save_path='logs', file_type=__file__)
    logger.info('logging message')
