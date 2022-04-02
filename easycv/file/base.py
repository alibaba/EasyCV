# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import hashlib
import os
import re
import shutil
import time
from datetime import datetime
from functools import lru_cache
from typing import List, Union


class IOBase:

    @staticmethod
    def register(options):
        pass

    def open(self, path: str, mode: str = 'r'):
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def move(self, src: str, dst: str):
        raise NotImplementedError

    def copy(self, src: str, dst: str):
        raise NotImplementedError

    def copytree(self, src: str, dst: str):
        raise NotImplementedError

    def makedirs(self, path: str, exist_ok=True):
        raise NotImplementedError

    def remove(self, path: str):
        raise NotImplementedError

    def rmtree(self, path: str):
        raise NotImplementedError

    def listdir(self,
                path: str,
                recursive=False,
                full_path=False,
                contains=None):
        raise NotImplementedError

    def isdir(self, path: str) -> bool:
        raise NotImplementedError

    def isfile(self, path: str) -> bool:
        raise NotImplementedError

    def abspath(self, path: str) -> str:
        raise NotImplementedError

    def last_modified(self, path: str) -> datetime:
        raise NotImplementedError

    def last_modified_str(self, path: str) -> str:
        raise NotImplementedError

    def size(self, path: str) -> int:
        raise NotImplementedError

    def md5(self, path: str) -> str:
        hash_md5 = hashlib.md5()
        with self.open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    re_remote = re.compile(r'(oss|https?)://')

    def islocal(self, path: str) -> bool:
        return not self.re_remote.match(path.lstrip())

    def is_writable(self, path):
        new_dir = ''
        if self.islocal(path) and not self.exists(path):
            new_dir = path
            while True:
                parent = os.path.dirname(new_dir)
                if self.exists(parent):
                    break
                new_dir = parent
            self.makedirs(path)
        flag = self._is_writable(path)
        if new_dir and self.exists(new_dir):
            self.remove(new_dir)
        return flag

    @lru_cache(maxsize=8)
    def _is_writable(self, path):
        import oss2
        try:
            tmp_file = os.path.join(path, f'.tmp.{time.time()}')
            with self.open(tmp_file, 'w') as f:
                f.write('test line.')
            self.remove(tmp_file)
        except (OSError, oss2.exceptions.RequestError,
                oss2.exceptions.ServerError):
            return False
        return True


class IOLocal(IOBase):
    __name__ = 'IOLocal'

    def _check_path(self, path):
        if not self.islocal(path):
            raise RuntimeError(
                'OSS Credentials must be provided by `easycv.io.access_oss`. ')

    def open(self, path, mode='r'):
        self._check_path(path)
        path = self.abspath(path)
        return open(path, mode=mode)

    def exists(self, path):
        self._check_path(path)
        path = self.abspath(path)
        return os.path.exists(path)

    def move(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src)
        dst = self.abspath(dst)
        if src == dst:
            return
        shutil.move(src, dst)

    def copy(self, src, dst):
        self._check_path(src)
        self._check_path(dst)

        if os.path.isdir(dst):
            if not os.path.exists(dst):
                os.makedirs(dst)
            file_name = os.path.split(src)[-1]
            dst = os.path.join(dst, file_name)

        src = self.abspath(src)
        dst = self.abspath(dst)
        shutil.copyfile(src, dst)

        return dst

    def copytree(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src).rstrip('/')
        dst = self.abspath(dst).rstrip('/')
        if src == dst:
            return
        self.makedirs(dst)
        created_dir = {dst}
        for file in self.listdir(src, recursive=True):
            src_file = os.path.join(src, file)
            dst_file = os.path.join(dst, file)
            dst_dir = os.path.dirname(dst_file)
            if dst_dir not in created_dir:
                self.makedirs(dst_dir)
                created_dir.add(dst_dir)
            self.copy(src_file, dst_file)

    def makedirs(self, path, exist_ok=True):
        self._check_path(path)
        path = self.abspath(path)
        os.makedirs(path, exist_ok=exist_ok)

    def remove(self, path):
        self._check_path(path)
        path = self.abspath(path)
        if os.path.isdir(path):
            self.rmtree(path)
        else:
            os.remove(path)

    def rmtree(self, path):
        shutil.rmtree(path)

    def listdir(self,
                path,
                recursive=False,
                full_path=False,
                contains: Union[str, List[str]] = None):
        self._check_path(path)
        path = self.abspath(path)

        if recursive:
            files = [
                os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn
            ]
            if not full_path:
                prefix_len = len(path.rstrip('/')) + 1
                files = [file[prefix_len:] for file in files]
        else:
            files = os.listdir(path)
            if full_path:
                files = [os.path.join(path, file) for file in files]

        if not contains:
            return files

        if isinstance(contains, str):
            contains = [contains]
        files = [
            file for file in files
            if any(keyword in file for keyword in contains)
        ]
        return files

    def isdir(self, path):
        self._check_path(path)
        return os.path.isdir(path)

    def isfile(self, path):
        self._check_path(path)
        return os.path.isfile(path)

    def glob(self, path):
        self._check_path(path)
        return glob.glob(path)

    def abspath(self, path):
        self._check_path(path)
        return os.path.abspath(path)

    def last_modified(self, path):
        return datetime.fromtimestamp(float(self.last_modified_str(path)))

    def last_modified_str(self, path):
        self._check_path(path)
        return str(os.path.getmtime(path))

    def size(self, path: str) -> int:
        return os.stat(path).st_size
