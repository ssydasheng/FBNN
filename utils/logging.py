import logging
import os
import time
import shutil

def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def get_logger(name, logpath, filepath, package_files=[],
               displaying=True, saving=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = logpath + name + time.strftime("-%Y%m%d-%H%M%S")
    makedirs(log_path)
    if saving:
        info_file_handler = logging.FileHandler(log_path)
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    logger.info(filepath)
    with open(filepath, 'r') as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


def manage_folder_tree(args, str_):
    if os.path.exists('results/' + str_) and not args.restore:
        if args.run == 'debug':
            print('Folder already exists, in debug mode, deleting it')
            shutil.rmtree('results/' + str_)  # RM folder, use with caution
        else:
            print(str_)
            raise ValueError('path already exists, select another name for run')
    if not args.restore:
        os.makedirs('results/' + str_)
    print(args)