from easycv.datasets.registry import PIPELINES


@PIPELINES.register_module()
class TextTokenizer:

    def __init__(
        self,
        tokenizer_type='bert-base-chinese',
        max_length=50,
        padding='max_length',
        truncation=True,
    ):
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_type)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __call__(self, results):
        text = results['text']
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            return_tensors='pt',
            truncation=True)

        results['text_input_ids'] = tokens.input_ids.reshape([-1])
        results['text_input_mask'] = tokens.attention_mask.reshape([-1])
        return results
