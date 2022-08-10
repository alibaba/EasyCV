# Visualization

EasyCV supports  the visualization of parameters and images of Wandb and tensorboard. Please refer to the following guidance for usage.

## Tensorboard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.

EasyCV show:

![img](https://user-images.githubusercontent.com/30484308/164178438-0c9d5991-cd9f-4744-bd03-cc81777726a9.png)


Please refer to the following usage instructions.

### Giudes

For more usages and details, please refer to : https://www.tensorflow.org/tensorboard/get_started .

1. Prepare environment

   ```shell
   $ pip install tensorboard
   ```

2. Add Tensorboard config

   Add tensorboard config to your config file as follows:

   ```python
   log_config = dict(
       interval=10,
       hooks=[
           dict(type='TensorboardLoggerHookV2'),
       ])
   ```

   **Params:**

   - interval: log intervel to save visualized **parameters**

3. Edit image visualization configuration (Optional)

   The interval of image visualization is equal to the interval of evaluation, and at present we only do image visualization in validation mode. Add or edit image visualization configuration as follows:

   ```python
   # evaluation
   eval_config = dict(
       interval=1,
       visualization_config=dict(
           vis_num=20,
           score_thr=0.5,
       )
   )
   ```

   **Params:**

   - interval: intervel of evaluate and save visualized **images**
   - visualization_config
     - vis_num: number of visualized samples
     - score_thr: the threshold to filter box, boxes with scores greater than score_thr will be kept


## Wandb

[Weights & Biases](https://docs.wandb.ai/) is the machine learning platform for developers to build better models faster. Use W&B's lightweight, interoperable tools to quickly track experiments, version and iterate on datasets, evaluate model performance, reproduce models, visualize results and spot regressions, and share findings with colleagues.

EasyCV show:

![img](https://user-images.githubusercontent.com/30484308/164178749-b1e0b678-d017-4f75-991f-43fb5e730e49.png)

Please refer to the following usage instructions.

### Giudes

For more usages and details, please refer to : https://docs.wandb.ai/guides .

#### register wandb count

Register an account on [wandb website](https://wandb.ai) and obtain the API key of the account.

1. Prepare environment

   ```shell
   $ pip install wandb
   ```

   Run the following command to log in and enter API key.

   ```shell
   $ wandb login
   ```

   If you're running a script in an automated environment, you can control wandb with environment variables set before the script runs or within the script.

   ```python
   # This is secret and shouldn't be checked into version control
   WANDB_API_KEY=$YOUR_API_KEY
   # Name and notes optional
   WANDB_NAME="My first run"
   WANDB_NOTES="Smaller learning rate, more regularization."
   ```

   ```python
   # Only needed if you don't checkin the wandb/settings file
   WANDB_ENTITY=$username
   WANDB_PROJECT=$project
   ```

   ```python
   # If you don't want your script to sync to the cloud
   os.environ['WANDB_MODE'] = 'offline'
   ```

2. add Wandb config

   Add Wandb config to your config file as follows:

   ```python
   log_config = dict(
       interval=10,
       hooks=[
           dict(type='WandbLoggerHookV2'),
       ])
   ```

   **Params:**

   - interval: log intervel to save visualized **parameters**

3.  image visualization configuration (Optional)

   Add or edit image visualization configuration as follows:

   ```python
   # evaluation
   eval_config = dict(
       interval=10,
       visualization_config=dict(
           vis_num=10,
           score_thr=0.5,
       )
   )
   ```

   **Params:**

   - interval: intervel of evaluate and save visualized **images**
   - visualization_config
     - vis_num: number of visualized samples
     - score_thr: the threshold to filter box, boxes with scores greater than score_thr will be kept
