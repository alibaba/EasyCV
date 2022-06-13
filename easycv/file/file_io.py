# Copyright (c) Alibaba, Inc. and its affiliates.
import fnmatch
import logging
import os
import re
import time
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from typing import List, Union

from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from .base import IOLocal
from .utils import (OSS_PREFIX, create_namedtuple, get_oss_config, is_oss_path,
                    mute_stderr, oss_progress)


def set_oss_env(ak_id: str, ak_secret: str, hosts: Union[str, List[str]],
                buckets: Union[str, List[str]]):
    if isinstance(buckets, str):
        buckets = [buckets]
    if isinstance(hosts, str):
        hosts = [hosts]
    assert (len(buckets) == len(hosts))

    os.environ['OSS_ACCESS_KEY_ID'] = ak_id
    os.environ['OSS_ACCESS_KEY_SECRET'] = ak_secret
    os.environ['OSS_ENDPOINTS'] = ','.join(hosts)
    os.environ['OSS_BUCKETS'] = ','.join(buckets)


class IO(IOLocal):
    """
    IO module to support both local and oss io.
    If access oss file, you need to authorize OSS, please refer to `IO.access_oss`.
    """

    __name__ = 'IO'

    def __init__(self, max_retry=10):

        super(IO, self).__init__()

        self.oss_pattern = re.compile(r'oss://([^/]+)/(.+)')
        self.buckets_map = {}
        self.auth = None
        self.oss_config = None
        self.max_retry = max_retry

        if self._is_oss_env_prepared():
            self.access_oss(
                ak_id=os.environ.get('OSS_ACCESS_KEY_ID'),
                ak_secret=os.environ.get('OSS_ACCESS_KEY_SECRET'),
                hosts=[
                    i.strip()
                    for i in os.environ.get('OSS_ENDPOINTS').split(',')
                ],
                buckets=[
                    i.strip() for i in os.environ.get('OSS_BUCKETS').split(',')
                ])

    def _is_oss_env_prepared(self):
        return (os.environ.get('OSS_ACCESS_KEY_ID')
                and os.environ.get('OSS_ACCESS_KEY_SECRET')
                and os.environ.get('OSS_ENDPOINTS')
                and os.environ.get('OSS_BUCKETS'))

    def _get_default_oss_config(self):
        oss_config = get_oss_config()
        ak_id = oss_config['ak_id']
        ak_secret = oss_config['ak_secret']
        hosts = oss_config['hosts']
        buckets = oss_config['buckets']
        return ak_id, ak_secret, hosts, buckets

    def access_oss(self,
                   ak_id: str = '',
                   ak_secret: str = '',
                   hosts: Union[str, List[str]] = '',
                   buckets: Union[str, List[str]] = ''):
        """If access oss file, you need to authorize OSS as follows:

        Method1:
        from easycv.file import io
        io.access_oss(
            ak_id='your_accesskey_id',
            ak_secret='your_accesskey_secret',
            hosts='your endpoint' or ['your endpoint1', 'your endpoint2'],
            buckets='your bucket' or ['your bucket1', 'your bucket2'])
        Method2:
            Add oss config to your local file `~/.ossutilconfig`, as follows:
            More oss config information, please refer to: https://help.aliyun.com/document_detail/120072.html
            ```
            [Credentials]
                language = CH
                endpoint = your endpoint
                accessKeyID = your_accesskey_id
                accessKeySecret = your_accesskey_secret
            [Bucket-Endpoint]
                bucket1 = endpoint1
                bucket2 = endpoint2
            ```
            Then run the following command, the config file will be read by default to authorize oss.

            from easycv.file import io
            io.access_oss()
        """
        from oss2 import Auth
        if not (ak_id and ak_secret):
            try:
                ak_id, ak_secret, hosts, buckets = \
                    self._get_default_oss_config()
            except Exception as e:
                logging.error(e)
                raise ValueError(
                    'Please provide oss config or create `~/.ossutilconfig` file!'
                )
        self.auth = Auth(ak_id, ak_secret)
        if isinstance(buckets, str):
            buckets = [buckets]
        if isinstance(hosts, str):
            hosts = [hosts for _ in range(len(buckets))]
        else:
            assert len(hosts) == len(buckets), \
                'number of hosts and number of buckets should be the same'

        self.buckets_map = {
            bucket_name: self._create_bucket_obj(host, bucket_name)
            for host, bucket_name in zip(hosts, buckets)
        }

        self.oss_config = create_namedtuple(
            ak_id=ak_id, ak_secret=ak_secret, hosts=hosts, buckets=buckets)

        if not self._is_oss_env_prepared():
            set_oss_env(
                ak_id=ak_id, ak_secret=ak_secret, hosts=hosts, buckets=buckets)

    def _create_bucket_obj(self, host, bucket_name):
        import oss2
        from oss2 import Bucket

        try:
            return Bucket(self.auth, host, bucket_name)
        except oss2.exceptions.ClientError as e:
            logging.error(f'Invalid bucket name "{bucket_name}"')
            raise oss2.exceptions.ClientError(e)

    def _get_bucket_obj_and_path(self, path):
        m = self.oss_pattern.match(path)
        if not m:
            raise IOError(
                f'invalid oss path: "{path}", should be "oss://<bucket_name>/path"'
            )
        bucket_name, path = m.groups()
        path = path.replace('//', '/')
        bucket_name = bucket_name.split('.')[0]

        bucket = self.buckets_map.get(bucket_name, None)

        if not bucket:
            raise IOError(
                f'Bucket {bucket_name} not registered in oss_io_config')
        return bucket, path

    def _read_oss(self, full_path, mode):
        assert mode in ['r', 'rb']
        path_exists = self.exists(full_path)
        if not path_exists:
            raise FileNotFoundError(full_path)
        bucket, path = self._get_bucket_obj_and_path(full_path)
        num_retry = 0
        data = None
        while num_retry < self.max_retry:
            try:
                obj = bucket.get_object(path)
                # auto cache large files to avoid memory issues
                if obj.content_length > 200 * 1024**2:  # 200M
                    with tqdm(
                            total=obj.content_length,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                            leave=False,
                            desc='reading ' +
                            os.path.basename(full_path)) as t:
                        obj = CallbackIOWrapper(t.update, obj, 'read')
                        data = obj.read()
                else:
                    data = obj.read()
                break
            except Exception as e:
                num_retry += 1
                logging.warning(
                    f'read exception occur, sleep 3s to retry num_retry/max_retry {num_retry}/{self.max_retry}\n {e}'
                )
                time.sleep(3)

        if data is None:
            raise ValueError('Read file error: %s!' % full_path)

        if mode == 'rb':
            return NullContextWrapper(BytesIO(data))
        else:
            return NullContextWrapper(StringIO(data.decode()))

    def _write_oss(self, full_path, mode):
        assert mode in ['w', 'wb']
        bucket, path = self._get_bucket_obj_and_path(full_path)
        path_exists = self.exists(full_path)
        if path_exists:
            bucket.delete_object(path)
        if mode == 'wb':
            return BinaryOSSFile(bucket, path)
        return OSSFile(bucket, path)

    def _append_oss(self, full_path):
        path_exists = self.exists(full_path)
        bucket, path = self._get_bucket_obj_and_path(full_path)
        position = bucket.head_object(path).content_length \
            if path_exists else 0
        return OSSFile(bucket, path, position=position)

    def open(self, full_path, mode='r'):
        """
        Same usage as the python build-in `open`.
        Support local path and oss path.

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            # Write something to a oss file.
            with io.open('oss://bucket_name/demo.txt', 'w') as f:
                f.write("test")

            # Read from a oss file.
            with io.open('oss://bucket_name/demo.txt', 'r') as f:
                print(f.read())
        Args:
            full_path: absolute oss path
        """
        if not is_oss_path(full_path):
            return super().open(full_path, mode)

        if 'w' in mode:
            return self._write_oss(full_path, mode)
        elif mode == 'a':
            return self._append_oss(full_path)
        elif 'r' in mode:
            return self._read_oss(full_path, mode)
        else:
            raise ValueError('invalid mode: %s' % mode)

    def exists(self, path):
        """
        Whether the file exists, same usage as `os.path.exists`.
        Support local path and oss path.

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            ret = io.exists('oss://bucket_name/dir')
            print(ret)
        Args:
            path: oss path or local path
        """
        if not is_oss_path(path):
            return super().exists(path)

        bucket, _path = self._get_bucket_obj_and_path(path)
        # TODO: 网络不稳时try，其他错误直接抛出
        if not path.endswith('/'):
            # if file exists
            exists = self._obj_exists(bucket, _path)
            # if bucket exists
            if not exists:
                _path = self._correct_oss_dir(_path)
                exists = self._obj_exists(bucket, _path)
        else:
            # if bucket exists
            exists = self._obj_exists(bucket, _path)

        return exists

    def _obj_exists(self, bucket, path):
        return bucket.object_exists(path)

    def move(self, src, dst):
        """Move src to dst,  same usage as shutil.move.
        Support local path and oss path.

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            # move oss file to local
            io.move('oss://bucket_name/file.txt', '/your/local/path/file.txt')
            # move oss file to oss
            io.move('oss://bucket_name/file.txt', 'oss://bucket_name/file.txt')
            # move local file to oss
            io.move('/your/local/file.txt', 'oss://bucket_name/file.txt')
            # move directory
            io.move('oss://bucket_name/dir1', 'oss://bucket_name/dir2')
        Args:
            src: oss path or local path
            dst: oss path or local path
        """
        if not is_oss_path(src) and not is_oss_path(dst):
            return super().move(src, dst)
        if src == dst:
            return
        if self.isfile(src):
            self.copy(src, dst)
        else:
            self.copytree(src, dst)
        self.remove(src)

    def safe_copy(self, src, dst, try_max=3):
        """ oss always bug, we need safe_copy!
        """
        try_flag = True
        try_idx = 0
        while try_flag and try_idx < try_max:
            try_idx += 1
            try:
                self.copy(src, dst)
                try_flag = False
            except:
                pass

        if try_flag:
            print('oss copy from %s to %s failed!!! TRIGGER safe exit!' %
                  (src, dst))
        return

    def _download_oss(self, src, dst):
        target_dir, _ = os.path.split(dst)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        bucket, src = self._get_bucket_obj_and_path(src)
        obj = bucket.get_object(src)
        if obj.content_length > 100 * 1024**2:  # 100M
            with oss_progress('downloading') as callback:
                bucket.get_object_to_file(src, dst, progress_callback=callback)
        else:
            bucket.get_object_to_file(src, dst)

        return

    def _upload_oss(self, src, dst):
        bucket, dst = self._get_bucket_obj_and_path(dst)
        self._create_oss_dirs(bucket, os.path.split(dst)[0])
        src_size = os.stat(src).st_size
        if src_size > 5 * 1024**3:  # 5G
            raise RuntimeError(
                f'A file > 5G cannot be uploaded to OSS. Please split your file first.\n{src}'
            )
        if src_size > 100 * 1024**2:  # 100M
            with oss_progress('uploading') as callback:
                logging.info(f'upload {src} to {dst}')
                bucket.put_object_from_file(
                    dst, src, progress_callback=callback)
        else:
            bucket.put_object_from_file(dst, src)

        return

    def _copy_oss(self, src, dst):
        bucket, dst = self._get_bucket_obj_and_path(dst)
        # copy between oss paths
        src_bucket, src = self._get_bucket_obj_and_path(src)
        total_size = src_bucket.head_object(src).content_length
        if src_bucket.get_bucket_location(
        ).location != bucket.get_bucket_location().location:
            import tempfile
            local_tmp = os.path.join(tempfile.gettempdir(), src)
            self.copy(f'{OSS_PREFIX}{src_bucket.bucket_name}/{src}', local_tmp)
            self.copy(local_tmp, f'{OSS_PREFIX}{bucket.bucket_name}/{dst}')
            self.remove(local_tmp)
            return

        if total_size < 1024**3 or src_bucket != bucket:  # 1GB
            bucket.copy_object(src_bucket.bucket_name, src, dst)
        else:
            # multipart copy
            from oss2.models import PartInfo
            from oss2 import determine_part_size
            part_size = determine_part_size(
                total_size, preferred_size=100 * 1024)
            upload_id = bucket.init_multipart_upload(dst).upload_id
            parts = []

            part_number = 1
            offset = 0
            while offset < total_size:
                num_to_upload = min(part_size, total_size - offset)
                byte_range = (offset, offset + num_to_upload - 1)

                result = bucket.upload_part_copy(bucket.bucket_name, src,
                                                 byte_range, dst, upload_id,
                                                 part_number)
                parts.append(PartInfo(part_number, result.etag))

                offset += num_to_upload
                part_number += 1

            bucket.complete_multipart_upload(dst, upload_id, parts)

    def copy(self, src, dst):
        """
        Copy a file from src to dst. Same usage as `shutil.copyfile`.
        If you want to copy a directory, please use `easycv.io.copytree`

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            # Copy a file from local to oss:
            io.copy('/your/local/file.txt', 'oss://bucket/dir/file.txt')

            # Copy a oss file to local:
            io.copy('oss://bucket/dir/file.txt', '/your/local/file.txt')

            # Copy a file from oss to oss::
            io.copy('oss://bucket/dir/file.txt', 'oss://bucket/dir/file2.txt')

        Args:
            src: oss path or local path
            dst: oss path or local path
        """
        cloud_src = is_oss_path(src)
        cloud_dst = is_oss_path(dst)
        if not cloud_src and not cloud_dst:
            return super().copy(src, dst)

        if src == dst:
            return
        # download
        if cloud_src and not cloud_dst:
            self._download_oss(src, dst)
            return
        # upload
        if cloud_dst and not cloud_src:
            self._upload_oss(src, dst)
            return

        self._copy_oss(src, dst)

    def copytree(self, src, dst):
        """
        Copy files recursively from src to dst. Same usage as `shutil.copytree`.
        If you want to copy a file, please use `easycv.io.copy`.

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            # copy files from local to oss
            io.copytree(src='/your/local/dir1', dst='oss://bucket_name/dir2')
            # copy files from oss to local
            io.copytree(src='oss://bucket_name/dir2', dst='/your/local/dir1')
            # copy files from oss to oss
            io.copytree(src='oss://bucket_name/dir1', dst='oss://bucket_name/dir2')

        Args:
            src: oss path or local path
            dst: oss path or local path
        """
        cloud_src = is_oss_path(src)
        cloud_dst = is_oss_path(dst)
        if not cloud_src and not cloud_dst:
            return super().copytree(src, dst)

        self.makedirs(dst)
        created_dir = {dst}
        src_files = self.listdir(src, recursive=True)
        max_len = min(max(map(len, src_files)), 50)
        with tqdm(src_files, desc='copytree', leave=False) as progress:
            for file in progress:
                src_file = os.path.join(src, file)
                dst_file = os.path.join(dst, file)
                dst_dir = os.path.dirname(dst_file)
                if dst_dir not in created_dir:
                    self.makedirs(dst_dir)
                    created_dir.add(dst_dir)
                progress.set_postfix({'file': f'{file:-<{max_len}}'[:max_len]})
                self.copy(src_file, dst_file)

    def listdir(self,
                path,
                recursive=False,
                full_path=False,
                contains: Union[str, List[str]] = None):
        """
        List all objects in path. Same usage as `os.listdir`.

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            ret = io.listdir('oss://bucket/dir', recursive=True)
            print(ret)
        Args:
            path: local file path or oss path.
            recursive: If False, only list the top level objects.
                If True, recursively list all objects.
            full_path: if full path, return files with path prefix.
            contains: substr to filter list files.

        return: A list of path.
        """
        from oss2 import ObjectIterator

        if not is_oss_path(path):
            return super().listdir(path, recursive, full_path, contains)

        bucket, path = self._get_bucket_obj_and_path(path)
        path = self._correct_oss_dir(path)
        files = [
            obj.key for obj in ObjectIterator(
                bucket, prefix=path, delimiter='' if recursive else '/')
        ]
        if path in files:
            files.remove(path)

        if not files and not self._obj_exists(bucket, path):
            raise FileNotFoundError(
                f'No such directory: {OSS_PREFIX}{bucket.bucket_name}/{path}')

        if full_path:
            files = [
                f'{OSS_PREFIX}{bucket.bucket_name}/{file}' for file in files
            ]
        else:
            files = [file[len(path):] for file in files]

        if not contains:
            return files

        if isinstance(contains, str):
            contains = [contains]
        files = [
            file for file in files
            if any(keyword in file for keyword in contains)
        ]
        return files

    def _remove_obj(self, path):
        bucket, path = self._get_bucket_obj_and_path(path)
        with mute_stderr():
            bucket.delete_object(path)

    def remove(self, path):
        """
        Remove a file or a directory recursively.
        Same usage as `os.remove` or `shutil.rmtree`.

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            # Remove a oss file
            io.remove('oss://bucket_name/file.txt')

            # Remove a oss directory
            io.remove('oss://bucket_name/dir/')

        Args:
            path: local or oss path, file or directory
        """
        if not is_oss_path(path):
            return super().remove(path)

        if self.isfile(path):
            self._remove_obj(path)
        else:
            return self.rmtree(path)

    def _correct_oss_dir(self, path):
        if not path.endswith('/'):
            path += '/'

        return path

    def rmtree(self, path):
        """
        Remove directory recursively, same usage as `shutil.rmtree`

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            io.remove('oss://bucket_name/dir_name')
            # Or
            io.remove('oss://bucket_name/dir_name/')
        Args:
            path: oss path
        """
        if not is_oss_path(path):
            return super().rmtree(path)
        # have to delete its content first before delete the directory itself
        for file in self.listdir(path, recursive=True, full_path=True):
            logging.info(f'delete {file}')
            self._remove_obj(file)
        if self.exists(path):
            # remove the directory itself
            path = self._correct_oss_dir(path)
            self._remove_obj(path)

    def makedirs(self, path, exist_ok=True):
        """
        Create directories recursively, same usage as `os.makedirs`

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            io.makedirs('oss://bucket/new_dir/')
        Args:
            path: local or oss dir path
        """
        del exist_ok
        # there is no need to create directory in oss
        if not is_oss_path(path):
            return super().makedirs(path)
        else:
            bucket, _path = self._get_bucket_obj_and_path(path)
            self._create_oss_dirs(bucket, _path)

    def _create_oss_dirs(self, bucket, dir_path):
        """
        Fix cannot access middle folder created by `bucket.put_object_from_file`
        by put null object to middle dir to create dir object.
        """
        dirs = dir_path.split(os.sep)
        cur_d = ''
        for d in dirs:
            cur_d = os.path.join(cur_d, d)
            cur_d = self._correct_oss_dir(cur_d)
            if not self._obj_exists(bucket, cur_d):
                bucket.put_object(cur_d, '')

    def isdir(self, path):
        """
        Return whether a path is directory, same usage as `os.path.isdir`

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            io.isdir('oss://bucket/dir/')
        Args:
            path: local or oss path
        Return: bool, True or False.
        """
        if not is_oss_path(path):
            return super().isdir(path)
        path = self._correct_oss_dir(path)
        return self.exists(path)

    def isfile(self, path):
        """
        Return whether a path is file object, same usage as `os.path.isfile`

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            io.isfile('oss://bucket/file.txt')
        Args:
            path: local or oss path
        Return: bool, True or False.
        """
        if not is_oss_path(path):
            return super().exists(path) and not super().isdir(path)
        return self.exists(path) and not self.isdir(path)

    def glob(self, file_path):
        """
        Return a list of paths matching a pathname pattern.
        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            io.glob('oss://bucket/dir/*.txt')
        Args:
            path: local or oss file pattern
        Return: list, a list of paths.
        """
        if not is_oss_path(file_path):
            return super().glob(file_path)

        dir_name = self._correct_oss_dir(os.path.split(file_path)[0])

        file_list = self.listdir(dir_name)
        reture_list = []
        for l in file_list:
            name = dir_name + l
            if fnmatch.fnmatch(name, file_path):
                reture_list.append(name)
        return reture_list

    def abspath(self, path):
        if not is_oss_path(path):
            return super().abspath(path)
        return path

    def authorize(self, path):
        if not is_oss_path(path):
            raise ValueError('Only oss path can use "authorize"')
        import oss2
        bucket, path = self._get_bucket_obj_and_path(path)
        bucket.put_object_acl(path, oss2.OBJECT_ACL_PUBLIC_READ)

    def last_modified(self, path):
        if not is_oss_path(path):
            return super().last_modified(path)
        return datetime.strptime(
            self.last_modified_str(path),
            r'%a, %d %b %Y %H:%M:%S %Z') + timedelta(hours=8)

    def last_modified_str(self, path):
        if not is_oss_path(path):
            return super().last_modified_str(path)
        bucket, path = self._get_bucket_obj_and_path(path)
        return bucket.get_object_meta(path).headers['Last-Modified']

    def size(self, path: str) -> int:
        """
        Get the size of file path, same usage as `os.path.getsize`

        Example:
            from easycv.file import io
            io.access_oss(your oss config) # only oss file need, refer to `IO.access_oss`
            size = io.size('oss://bucket/file.txt')
            print(size)
        Args:
            path: local or oss path.
        Return: size of file in bytes
        """
        if not is_oss_path(path):
            return super().size(path)
        bucket, path = self._get_bucket_obj_and_path(path)
        return int(bucket.get_object_meta(path).headers['Content-Length'])


class OSSFile:

    def __init__(self, bucket, path, position=0):
        self.position = position
        self.bucket = bucket
        self.path = path
        self.buffer = StringIO()

    def write(self, content):
        # without a "with" statement, the content is written immediately without buffer
        # when writing a large batch of contents at a time, this will be quite slow
        import oss2
        buffer = self.buffer.getvalue()
        if buffer:
            content = buffer + content
            self.buffer.close()
            self.buffer = StringIO()
        try:
            result = self.bucket.append_object(self.path, self.position,
                                               content)
            self.position = result.next_position
        except oss2.exceptions.PositionNotEqualToLength:
            raise RuntimeError(
                f'Race condition detected. It usually means multiple programs were writing to the same file'
                f'{OSS_PREFIX}{self.bucket.bucket_name}/{self.path} (Error 409: PositionNotEqualToLength)'
            )
        except (oss2.exceptions.RequestError,
                oss2.exceptions.ServerError) as e:
            self.buffer.write(content)
            logging.error(
                str(e) +
                f'when writing to {OSS_PREFIX}{self.bucket.bucket_name}/{self.path}. Content buffered.'
            )

    def flush(self, retry=0):
        import oss2
        try:
            self.bucket.append_object(self.path, self.position,
                                      self.buffer.getvalue())
        except oss2.exceptions.RequestError as e:
            if 'timeout' not in str(e) or retry > 2:
                raise
            # retry if timeout
            logging.error('| OSSIO timeout. Retry uploading...')
            import time
            time.sleep(5)
            self.flush(retry + 1)
        except oss2.exceptions.ObjectNotAppendable as e:
            from . import io
            logging.error(str(e) + '\nTrying to recover..\n')
            full_path = f'{OSS_PREFIX}{self.bucket.bucket_name}/{self.path}'
            with io.open(full_path) as f:
                prev_content = f.read()
            io.remove(full_path)
            self.position = 0
            content = self.buffer.getvalue()
            self.buffer.close()
            self.buffer = StringIO()
            self.write(prev_content)
            self.write(content)

    def close(self):
        self.flush()

    def seek(self, position):
        self.position = position

    def __enter__(self):
        return self.buffer

    def __exit__(self, *args):
        self.close()


class BinaryOSSFile:

    def __init__(self, bucket, path):
        self.bucket = bucket
        self.path = path
        self.buffer = BytesIO()

    def __enter__(self):
        return self.buffer

    def __exit__(self, *args):
        value = self.buffer.getvalue()
        if len(value) > 100 * 1024**2:  # 100M
            with oss_progress('uploading') as callback:
                self.bucket.put_object(
                    self.path, value, progress_callback=callback)
        else:
            self.bucket.put_object(self.path, value)


class NullContextWrapper:

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __iter__(self):
        return self._obj.__iter__()

    def __next__(self):
        return self._obj.__next__()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
