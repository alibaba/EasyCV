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

 ### 2.2 Test data
 if you add new data, please do the following to commit it to git-lfs before "git commit":
 ```bash
 python git-lfs/git_lfs.py add data/test/new_data
 python git-lfs/git_lfs.py push
 ```

 ## 3. Build pip package
 ```bash
 python setup.py sdist bdist_wheel
 ```
