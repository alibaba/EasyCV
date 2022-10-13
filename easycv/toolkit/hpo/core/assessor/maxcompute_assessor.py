# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from hpo_tools.core.assessor.pai_assessor import PAIAssessor
from hpo_tools.core.platform.maxcompute.pyodps_utils import kill_instance
from hpo_tools.core.utils.json_utils import remove_filepath


class MaxComputeAssessor(PAIAssessor):

    def trial_end(self, trial_job_id, success):
        logging.info('trial end')
        # user_cancelled or early_stopped
        if not success:
            # kill mc instance
            kill_instance(trial_id=trial_job_id)
            # remove json file
            remove_filepath(trial_id=trial_job_id)
