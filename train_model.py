# # Step 1: Install necessary libraries
# # !pip install torch torchvision torchaudio
# # !pip install transformers
# # !pip install datasets

# import torch
# from torch.utils.data import DataLoader
# from transformers import BertTokenizer, BertForSequenceClassification, DefaultDataCollator
# from transformers import AdamW
# from datasets import load_dataset

# # Step 2: Load the SST-2 dataset
# dataset = load_dataset("glue", "sst2")

# # Step 3: Preprocess the data
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def tokenize_function(examples):
#     return tokenizer(examples["sentence"], padding="max_length", truncation=True)

# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000)) # Using a subset for speed
# eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(872)) # Adjusted to match the dataset size

# data_collator = DefaultDataCollator(return_tensors="pt")

# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
# eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

# # Step 4: Train the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# model.to(device)

# optimizer = AdamW(model.parameters(), lr=5e-5)

# for epoch in range(3):  # loop over the dataset multiple times
#     model.train()
#     for batch in train_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

# # Step 5: Evaluate the Model
# model.eval()
# # Corrected evaluation loop to exclude unexpected arguments
# total_eval_accuracy = 0
# total_eval_examples = 0

# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items() if k not in ['idx', 'attention_mask']}
#     labels = batch.pop("labels").to(device)
    
#     with torch.no_grad():
#         outputs = model(**batch)
    
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)

#     total_eval_accuracy += (predictions == labels).sum().item()
#     total_eval_examples += labels.size(0)

# avg_eval_accuracy = total_eval_accuracy / total_eval_examples if total_eval_examples > 0 else 0
# print(f"Accuracy: {avg_eval_accuracy:.4f}")


# # After training and evaluation, save the model and tokenizer
# model_path = "my_bert_model"
# tokenizer.save_pretrained(model_path)
# model.save_pretrained(model_path)

from transformers import BertModel, BertTokenizer

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to a directory
model_path = './my_bert_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
