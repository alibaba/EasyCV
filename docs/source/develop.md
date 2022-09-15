# Develop

## 1. Code Style
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following toolsseed isortseed isortseed isort for linting and formatting:
- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [yapf](https://github.com/google/yapf): formatter
- [isort](https://github.com/timothycrosley/isort): sort imports

Style configurations of yapf and isort can be found in [setup.cfg](../../setup.cfg).
We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `seed-isort-config`, `isort`, `trailing whitespaces`,
 fixes `end-of-files`, sorts `requirments.txt` automatically on every commit.
 The config for a pre-commit hook is stored in [.pre-commit-config](../../.pre-commit-config.yaml).
 After you clone the repository, you will need to install initialize pre-commit hook.
 ```bash
 pip install -r requirements/tests.txt
 ```
 From the repository folder
 ```bash
 pre-commit install
 ```

 After this on every commit check code linters and formatter will be enforced.

 If you want to use pre-commit to check all the files, you can run
 ```bash
pre-commit run --all-files
 ```

 If you only want to format and lint your code, you can run
 ```bash
 sh scripts/linter.sh
 ```

 ## 2. Test
 ### 2.1 Unit test
 ```bash
bash scripts/ci_test.sh
 ```


### 2.2 Test data storage

As we need a lot of data for testing, including images, models. We use git lfs
to store those large files.

1. install git-lfs(version>=2.5.0)

for mac

```bash
brew install git-lfs
git lfs install
```

for centos, please download rpm from git-lfs github release [website](https://github.com/git-lfs/git-lfs/releases/tag/v3.2.0)
```bash
wget http://101374-public.oss-cn-hangzhou-zmf.aliyuncs.com/git-lfs-3.2.0-1.el7.x86_64.rpm
sudo rpm -ivh git-lfs-3.2.0-1.el7.x86_64.rpm
git lfs install
```

for ubuntu
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

2. track your data type using git lfs, for example, to track png files
```bash
git lfs track "*.png"
```

3. add your test files to `data/test/` folder, you can make directories if you need.
```bash
git add data/test/test.png
```

4. commit your test data to remote branch
```bash
git commit -m "xxx"
```

To pull data from remote repo, just as the same way you pull git files.
```bash
git pull origin branch_name
```


 ## 3. Build pip package
 ```bash
 python setup.py sdist bdist_wheel
 ```
