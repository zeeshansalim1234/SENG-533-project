from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the trained model and tokenizer
model_path = './my_bert_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Put model in evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    # Decode JSON request
    data = request.get_json(force=True)
    text = data['text']
    
    # Prepare text for BERT model
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Convert predictions to JSON
    positive_score = predictions[:, 1].item()  # Assuming index 1 is for positive sentiment
    result = {'positive_score': positive_score}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
