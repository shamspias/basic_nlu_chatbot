import torch
from transformers import AutoTokenizer, AutoModel

# Load the model and tokenizer
model = AutoModel.from_pretrained("distilbert-base-cased-distilled-squad", from_tf=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad", from_tf=True)

def chatbot_response(question):
    # Encode the input text
    input_ids = torch.tensor([tokenizer.encode(question, add_special_tokens=True)])
    
    # Get the model's prediction
    with torch.no_grad():
        outputs = model(input_ids)
    answer_start_scores, answer_end_scores = outputs[:2]
    
    # Get the best answer
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.decode(input_ids[0][answer_start:answer_end])
    
    return answer

# Test the chatbot
question = input("Ask your question. Ex: What is the return policy for your products? :")
print("Question:", question)
print("Answer:", chatbot_response(question))
