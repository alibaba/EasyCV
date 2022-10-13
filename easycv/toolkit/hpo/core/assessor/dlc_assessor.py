# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from hpo_tools.core.assessor.pai_assessor import PAIAssessor
from hpo_tools.core.platform.dlc.dlc_utils import kill_job
from hpo_tools.core.utils.json_utils import remove_filepath


class DLCAssessor(PAIAssessor):

    def trial_end(self, trial_job_id, success):
        logging.info('trial end')
        # user_cancelled or early_stopped
        if not success:
            # kill dlc job
            kill_job(trial_id=trial_job_id)
            # remove json file
            remove_filepath(trial_id=trial_job_id)
