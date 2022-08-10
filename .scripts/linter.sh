yapf -r -i easycv/ configs/ tests/ tools/ setup.py
isort -rc easycv/ configs/ tests/ tools/ setup.py
flake8 easycv/ configs/ tests/ tools/ setup.py
