from transformers import BertTokenizer, BertForQuestionAnswering
import torch

class QuestionUnderstanding:
    def __init__(self, model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        
    def answer_question(self, question, context):
        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            start_scores, end_scores = outputs.start_logits, outputs.end_logits
        
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        answer = ' '.join(tokens[start_index:end_index])
        
        return answer
    


"""
        if __name__ == "__main__":
            question_understanding = QuestionUnderstanding()
            context = "Transformers are a type of model architecture that has achieved state-of-the-art results on various NLP tasks."
            question = "What are transformers?"
            answer = question_understanding.answer_question(question, context)
            print(f"Question: {question}")
            print(f"Answer: {answer}")
"""