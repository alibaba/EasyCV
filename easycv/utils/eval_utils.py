# Copyright (c) Alibaba, Inc. and its affiliates.
def generate_best_metric_name(evaluate_type, dataset_name, metric_names):
    """
    Generate best metric name for different evaluator / different dataset / different metric_names
    evaluate_type: str
    dataset_name: None or str
    metric_names: None str or list[str] or tuple(str)

    Return:
        list[str]
    """
    base_name = evaluate_type
    if dataset_name is not None:
        base_name = base_name + '_' + dataset_name

    return_name = []
    if metric_names is None:
        return_name = [base_name]
    elif type(metric_names) == str:
        return_name = [base_name + '_' + metric_names]
    elif type(metric_names) == list or type(metric_names) == tuple:
        if len(metric_names) == 0:
            return_name = [base_name]
        for k in metric_names:
            assert (type(k) == str)
            return_name.append(base_name + '_' + k)

    return return_name
