_base_ = './vitdet_100e.py'

model = dict(backbone=dict(aggregation='bottleneck'))
