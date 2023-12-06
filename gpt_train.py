import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from rouge import Rouge

# Load the fine-tuned model and tokenizer
model_path = 'gpt2'  # or path to your pre-trained GPT-2 model if you're starting from scratch
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Load and preprocess your chatbot dataset
dataset_path = 'chatbot_dataset.txt'
with open(dataset_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Split dataset into training and testing sets
train_lines, test_lines = train_test_split(lines, test_size=0.2, random_state=42)

# Save training and testing datasets into separate text files
with open('train_dataset.txt', 'w', encoding='utf-8') as train_file:
    train_file.writelines(train_lines)

with open('test_dataset.txt', 'w', encoding='utf-8') as test_file:
    test_file.writelines(test_lines)

# Tokenize training dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='train_dataset.txt',
    block_size=128  # Adjust according to your dataset size
)

# Create data collator for training
train_data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # For causal language modeling tasks like text generation, set mlm to False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Output directory where checkpoints and results will be saved
    overwrite_output_dir=True,
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=4,  # Batch size per GPU/CPU
    save_steps=10_000,  # Save checkpoint every specified number of steps
    save_total_limit=2,  # Limit the total number of saved checkpoints
)

# Create Trainer for training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=train_data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model on the training dataset
trainer.train()

# Save the fine-tuned model
model_output_path = './fine_tuned_chatbot_gpt2'
trainer.save_model(model_output_path)

# Tokenize testing dataset
test_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='test_dataset.txt',
    block_size=128  # Adjust according to your dataset size
)

# Create data collator for testing
test_data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Evaluate the model on the testing dataset
eval_results = trainer.evaluate(test_dataset, data_collator=test_data_collator)

# Print evaluation results
print("Evaluation Results:")
print(eval_results)

# Generate responses for the test set
generator = pipeline('text-generation', model=model_output_path, tokenizer=tokenizer)
generated_responses = []
for line in test_lines:
    user_input = line.strip().replace('User: ', '')
    bot_response = generator(user_input, max_length=50, do_sample=True)[0]['generated_text'].strip()
    generated_responses.append(bot_response)

# Prepare reference and candidate texts for ROUGE calculation
references = [line.strip().replace('Bot: ', '') for line in test_lines]
candidates = generated_responses

# Calculate ROUGE score
rouge = Rouge()
scores = rouge.get_scores(candidates, references, avg=True)

# Print ROUGE scores
print("ROUGE Scores:")
print(f"ROUGE-1: {scores['rouge-1']['f']:.2f}")
print(f"ROUGE-2: {scores['rouge-2']['f']:.2f}")
print(f"ROUGE-L: {scores['rouge-l']['f']:.2f}")
