import logging
import numpy as np
import os

class CachingDictionary:
    def __init__(self, dir_path, load_func, ext='.npy'):
        os.makedirs(dir_path, exist_ok=True)

        self.dir_path = dir_path
        self.load_func = load_func
        self.ext = ext

    def get(self, key):
        file_path = self.get_file_path(key)

        if os.path.exists(file_path):
            return np.load(file_path, allow_pickle=True)
        return None

    def set(self, key, *args):
        value = self.load_func(*args)
        file_path = self.get_file_path(key)
        np.save(file_path, value, allow_pickle=True)
        logging.debug(f'caching value for {key}')

        return value

    def try_get(self, key, *args):
        file_path = self.get_file_path(key)

        if os.path.exists(file_path):
            logging.debug(f'reading value for {key}')

            return np.load(file_path, allow_pickle=True)
        return self.set(key, *args)

    def get_file_path(self, key):
        return os.path.join(
            self.dir_path,
            'cache_' + '_'.join([str(i) for i in key]) + self.ext
        )