# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.utils.registry import Registry, build_from_cfg

EVALUATORS = Registry('hook')


def build_evaluator(evaluator_cfg_list):
    """ build evaluator according to metric name

    Args:
        evaluator_cfg_list: list of evaluator config dict
    Return:
        return a list of evaluator
    """
    if isinstance(evaluator_cfg_list, dict):
        evaluator_cfg_list = [evaluator_cfg_list]
    evaluators = []

    for cfg in evaluator_cfg_list:
        evaluators.append(build_from_cfg(cfg, EVALUATORS))

    return evaluators
