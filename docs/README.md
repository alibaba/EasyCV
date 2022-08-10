## maintain docs
1. install requirements needed to build docs
    ```shell
    # in easycv root dir
    pip install requirements/docs.txt
    ```

2. build docs
    ```shell
    # in easycv/docs dir
    bash build_docs.sh
    ```

3. doc string format

    We adopt the google style docstring format as the standard, please refer to the following documents.
    1. Google Python style guide docstring [link](http://google.github.io/styleguide/pyguide.html#381-docstrings)
    2. Google docstring example [link](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
    3. sample：torch.nn.modules.conv [link](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d)
    4. Transformer as an example：

    ```python
    class Transformer(base.Layer):
        """
            Transformer model from ``Attention Is All You Need``,
            Original paper: https://arxiv.org/abs/1706.03762

            Args:
                num_token (int): vocab size.
                num_layer (int): num of layer.
                num_head (int): num of attention heads.
                embedding_dim (int): embedding dimension.
                attention_head_dim (int): attention head dimension.
                feed_forward_dim (int): feed forward dimension.
                initializer: initializer type.
                activation: activation function.
                dropout (float): dropout rate (0.0 to 1.0).
                attention_dropout (float): dropout rate for attention layer.

            Returns: None
        """
    ```
