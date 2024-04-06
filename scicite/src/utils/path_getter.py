import inspect
from pathlib import Path


class PathGetter:
    @staticmethod
    def get_src_directory() -> Path:
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        dir_path = Path(filename).absolute().parent.parent.parent
        return dir_path

    @staticmethod
    def get_root_directory() -> Path:
        src_dir = PathGetter.get_src_directory()
        return src_dir.parent

    @classmethod
    def get_data_directory(cls) -> Path:
        return cls.get_root_directory() / 'data'

