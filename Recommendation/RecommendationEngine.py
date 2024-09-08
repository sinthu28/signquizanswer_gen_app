import numpy as np
from transformers import BertTokenizer, BertModel
# import torch

class RecommendationEngine:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
    
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings
    
    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def recommend_answer(self, question, answers):
        question_vec = self.encode(question)
        answer_vecs = [self.encode(answer) for answer in answers]
        
        similarities = [self.cosine_similarity(question_vec, vec) for vec in answer_vecs]
        best_answer_index = np.argmax(similarities)
        
        return answers[best_answer_index], similarities[best_answer_index]

# Example Usage
# rec_engine = RecommendationEngine()
# recommended_answer, confidence = rec_engine.recommend_answer("What is the capital of France?", ["Paris", "London", "Berlin"])