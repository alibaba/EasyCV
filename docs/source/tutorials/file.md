# file tutorial

The file module of easycv supports operations both on local and oss files, oss introduction please refer to: https://www.aliyun.com/product/oss .

If you operate oss files, you need refer to [access_oss](#access_oss) to authorize oss first.

## Support operations

### access_oss

Authorize oss.

**Method1:**

```python
from easycv.file import io
io.access_oss(
ak_id='your_accesskey_id',
ak_secret='your_accesskey_secret',
hosts='your endpoint' or ['your endpoint1', 'your endpoint2'],
buckets='your bucket' or ['your bucket1', 'your bucket2'])
```

**Method2:**

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

```python
from easycv.file import io
io.access_oss()
```

**Method3:**

Set environment variables as follow,  EasyCV will automatically parse environment variables for authorization:

```python
import os

os.environ['OSS_ACCESS_KEY_ID'] = 'your_accesskey_id'
os.environ['OSS_ACCESS_KEY_SECRET'] = 'your_accesskey_secret'
os.environ['OSS_ENDPOINTS'] = 'your endpoint1,your endpoint2'
os.environ['OSS_BUCKETS'] = 'your bucket1,your bucket2'
```

### open

Support w,wb, a, r, rb modes on oss path. Local path is the same usage as the python build-in `open`.

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')

# Write something to a oss file.
with io.open('oss://bucket_name/demo.txt', 'w') as f:
	f.write("test")

# Read from a oss file.
with io.open('oss://bucket_name/demo.txt', 'r') as f:
  print(f.read())
```

**Example for local:**

```python
from easycv.file import io

# Write something to a oss file.
with io.open('/your/local/path/demo.txt', 'w') as f:
	f.write("test")

# Read from a oss file.
with io.open('/your/local/path/demo.txt', 'r') as f:
  print(f.read())
```

### exists

Whether the file exists, same usage as `os.path.exists`. Support local path and oss path.

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')

ret = io.exists('oss://bucket_name/dir')
print(ret)
```

**Example for Local:**

```python
from easycv.file import io

ret = io.exists('oss://bucket_name/dir')
print(ret)
```

### move

Move src to dst,  same usage as shutil.move. Support local path and oss path.

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')

# move oss file to local
io.move('oss://bucket_name/file.txt', '/your/local/path/file.txt')
# move oss file to oss
io.move('oss://bucket_name/dir1/file.txt', 'oss://bucket_name/dir2/file.txt')
# move local file to oss
io.move('/your/local/file.txt', 'oss://bucket_name/file.txt')
# move directory
io.move('oss://bucket_name/dir1/', 'oss://bucket_name/dir2/')
```

**Example for local:**

```python
from easycv.file import io

# move local file to local
io.move('/your/local/path1/file.txt', '/your/local/path2/file.txt')
# move local dir to local
io.move('/your/local/dir1', '/your/local/dir2')
```

### copy

Copy a file from src to dst. Same usage as `shutil.copyfile`.If you want to copy a directory, please refert to [copytree](# copytree).

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')

# Copy a file from local to oss:
io.copy('/your/local/file.txt', 'oss://bucket/dir/file.txt')
# Copy a oss file to local:
io.copy('oss://bucket/dir/file.txt', '/your/local/file.txt')
# Copy a file from oss to oss::
io.copy('oss://bucket/dir/file.txt', 'oss://bucket/dir/file2.txt')
```

**Example for local:**

```python
from easycv.file import io

# Copy a file from local to local:
io.copy('/your/local/path1/file.txt', '/your/local/path2/file.txt'')
```

### copytree

Copy files recursively from src to dst. Same usage as `shutil.copytree`.

If you want to copy a file, please use [copy](# copy).

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')

# copy files from local to oss
io.copytree(src='/your/local/dir1', dst='oss://bucket_name/dir2')
# copy files from oss to local
io.copytree(src='oss://bucket_name/dir2', dst='/your/local/dir1')
# copy files from oss to oss
io.copytree(src='oss://bucket_name/dir1', dst='oss://bucket_name/dir2')
```

**Example for local:**

```python
from easycv.file import io

# copy files from local to local
io.copytree(src='/your/local/dir1', dst='/your/local/dir2')
```

### listdir

List all objects in path. Same usage as `os.listdir`.

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')

ret = io.listdir('oss://bucket/dir', recursive=True)
print(ret)
```

**Example for local:**

```python
from easycv.file import io

ret = io.listdir('oss://bucket/dir', recursive=True)
print(ret)
```

### remove

Remove a file or a directory recursively. Same usage as `os.remove` or `shutil.rmtree`.

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')

# Remove a oss file
io.remove('oss://bucket_name/file.txt')
# Remove a oss directory
io.remove('oss://bucket_name/dir/')
```

**Example for local:**

```python
from easycv.file import io

# Remove a local file
io.remove('/your/local/path/file.txt')
# Remove a local directory
io.remove('/your/local/dir/')
```

### rmtree

Remove directory recursively, same usage as `shutil.rmtree`.

 **Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')

io.remove('oss://bucket_name/dir_name/')
```

 **Example for local:**

```python
from easycv.file import io

io.remove('/your/local/dir/')
```

### makedirs

Create directories recursively, same usage as `os.makedirs`.

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')

io.makedirs('oss://bucket/new_dir/')
```

**Example for local:**

```python
from easycv.file import io

io.makedirs('/your/local/new_dir/')
```

### isdir

Return whether a path is directory, same usage as `os.path.isdir`.

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config') # only oss file need, refer to `IO.access_oss`
ret = io.isdir('oss://bucket/dir/')
print(ret)
```

**Example for local:**

```python
from easycv.file import io

ret = io.isdir('your/local/dir/')
print(ret)
```

### isfile

Return whether a path is file object, same usage as `os.path.isfile`.

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')
ret = io.isfile('oss://bucket/file.txt')
print(ret)
```

**Example for local:**

```python
from easycv.file import io

ret = io.isfile('/your/local/path/file.txt')
print(ret)
```

### glob

Return a list of paths matching a pathname pattern.

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')

ret = io.glob('oss://bucket/dir/*.txt')
print(ret)
```

**Example for local:**

```python
from easycv.file import io

ret = io.glob('/your/local/dir/*.txt')
print(ret)
```

### size

Get the size of file path, same usage as `os.path.getsize`.

**Example for oss:**

`io.access_oss` please refer to [access_oss](# access_oss).

```python
from easycv.file import io

io.access_oss('your oss config')

size = io.size('oss://bucket/file.txt')
print(size)
```

**Example for local:**

```python
from easycv.file import io

size = io.size('/your/local/path/file.txt')
print(size)
```
