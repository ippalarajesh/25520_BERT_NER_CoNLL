from transformers import BertForTokenClassification, BertTokenizerFast

class BERT_NER:
    def __init__(self, model_name="bert-base-cased", num_labels=9):
        self.model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def tokenize(self, texts, labels=None, max_length=128):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            labels=labels
        )

    def predict(self, input_text):
        inputs = self.tokenize(input_text)
        outputs = self.model(**inputs)
        return outputs.logits