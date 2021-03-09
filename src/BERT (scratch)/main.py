import transformers as ppb
import warnings
import torch
from src.utils import *
from src.globals import *

warnings.filterwarnings('ignore')

class BERT_EVENT():

    def __init__(self):

        if not LARGE_BERT:
            if UNCASED:
                """
                12-layer, 768-hidden, 12-heads, 110M parameters.
                Trained on lower-cased Stanford QA dataset.
                """
                model_class, tokenizer_class, pretrained_weights = (
                ppb.BertForQuestionAnswering, ppb.BertTokenizer, 'bert-base-uncased')
            else:
                """ 	
                12-layer, 768-hidden, 12-heads, 110M parameters.
                Trained on cased Stanford QA dataset.
                """
                model_class, tokenizer_class, pretrained_weights = (
                ppb.BertForQuestionAnswering, ppb.BertTokenizer, 'bert-base-cased')
        else:
            if UNCASED:
                """
                24-layer, 1024-hidden, 16-heads, 340M parameters.
                Trained on lower-cased Stanford QA dataset.
                """
                model_class, tokenizer_class, pretrained_weights = (ppb.BertForQuestionAnswering, ppb.BertTokenizer,
                                                                    'bert-large-uncased-whole-word-masking-finetuned-squad')
            else:
                """ 	
                24-layer, 1024-hidden, 16-heads, 340M parameters.
                Trained on cased Stanford QA dataset.
                """
                model_class, tokenizer_class, pretrained_weights = (
                ppb.BertModel, ppb.BertTokenizer, 'bert-large-cased-whole-word-masking-finetuned-squad')

        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)

    def encode_input(self,question,reference):
        return self.tokenizer.encode_plus(question, reference)

    def recreate_answer(self,tokens,start,end):
        answer = tokens[start]
        for i in range(start + 1, end + 1):
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]
            else:
                answer += ' ' + tokens[i]
        return answer

    def ids_to_tokens(self,question,reference):
        encoded = self.encode_input(question, reference)
        return self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])

    def get_start_end(self,question,reference):
        encoded = self.encode_input(question, reference)
        token_type_ids = encoded['token_type_ids']
        return self.model(torch.tensor([encoded['input_ids']]),
                         token_type_ids=torch.tensor([token_type_ids]))

    def ask_question(self,question,reference):
        # To separate our question and answer in the input to the model,
        # we want to create a binary mask for question and answer
        # sep_index = encoded.index(tokenizer_.sep_token_id)
        # question_len = sep_index + 1
        # answer_len = len(encoded) - question_len
        # token_type_ids = [0] * question_len + [1] * answer_len
        [start,end] = self.get_start_end(question,reference)
        answer_start = torch.argmax(start)
        answer_end = torch.argmax(end)
        tokens = self.ids_to_tokens(question,reference)
        answer = self.recreate_answer(tokens,answer_start, answer_end)
        return answer

if __name__  ==  "__main__":
    model = BERT_EVENT()
    test_type = ['Conflict']
    for type in test_type:
        for i,reference in read_samples(type):

            #BERT Trigger part
            trigger_templates = get_trigger_templates()
            for question in trigger_templates:
                trigger = model.ask_question(question,reference)

            #BERT Argument part
            arg_templates = get_argument_templates(type)
            for arg, question in arg_templates.items():
                #arg_templates -> {arg:arg_question}
                arg_templates[arg] = model.ask_question(question, reference)

            with open('src/results/{}/{}_result.txt'.format(type,i), 'w') as f:
                f.write('Trigger >>> ' + trigger + '\n')
                for k, v in arg_templates.items():
                    f.write(str(k) + ' >>> ' + str(v) + '\n')
