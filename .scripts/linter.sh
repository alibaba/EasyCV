yapf -r -i easycv/ configs/ tests/ tools/ setup.py pai_jobs/easycv/resources/*.py
isort -rc easycv/ configs/ tests/ tools/ setup.py pai_jobs/easycv/resources/*.py
flake8 easycv/ configs/ tests/ tools/ setup.py pai_jobs/easycv/resources/*.py
