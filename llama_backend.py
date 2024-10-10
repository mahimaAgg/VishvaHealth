# Filename: train_and_chatbot.py

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from flask import Flask, request, jsonify
import threading

def main():
    # Set up the Flask app
    app = Flask(__name__)

    # Initialize model and tokenizer
    global nlp_pipeline
    nlp_pipeline = None

    # Device setup (use CPU to avoid MPS issues)
    device = torch.device("cpu")

    # Lock to prevent concurrent training sessions
    training_lock = threading.Lock()

    @app.route('/train', methods=['POST'])
    def train():
        global nlp_pipeline
        if training_lock.locked():
            return jsonify({'error': 'Training is already in progress'}), 429

        data = request.json
        training_data = data.get('training_data')
        if not training_data:
            return jsonify({'error': 'No training data provided'}), 400

        def train_thread():
            with training_lock:
                # Save the training data to a file
                training_data_file = 'training_data.txt'
                with open(training_data_file, 'w') as f:
                    f.write(training_data)

                # Load the base model and tokenizer
                model_name = "meta-llama/Llama-3.2-1B-Instruct"
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=False,
                    trust_remote_code=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )

                # Set special tokens
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.bos_token = tokenizer.eos_token
                tokenizer.sep_token = tokenizer.eos_token

                model.resize_token_embeddings(len(tokenizer))
                model.config.pad_token_id = tokenizer.pad_token_id

                # Move model to device
                model.to(device)

                # Prepare the training data
                from datasets import load_dataset

                dataset = load_dataset('text', data_files=training_data_file)

                def tokenize_function(examples):
                    return tokenizer(
                        examples['text'],
                        truncation=True,
                        padding='max_length',
                        max_length=512,
                    )

                tokenized_datasets = dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["text"],
                )

                # Set up data collator
                from transformers import DataCollatorForLanguageModeling

                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                )

                # Define training arguments
                from transformers import TrainingArguments, Trainer

                training_args = TrainingArguments(
                    output_dir='./results',
                    overwrite_output_dir=True,
                    num_train_epochs=1,
                    per_device_train_batch_size=1,
                    save_steps=500,
                    save_total_limit=2,
                    logging_steps=100,
                    evaluation_strategy='no',
                    learning_rate=5e-5,
                    weight_decay=0.01,
                    report_to='none',
                )

                # Initialize Trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=tokenized_datasets['train'],
                )

                # Train the model
                trainer.train()

                # Save the trained model and tokenizer
                trainer.save_model('./trained_model')
                tokenizer.save_pretrained('./trained_model')

                # Set up the pipeline for inference
                global nlp_pipeline
                nlp_pipeline = pipeline(
                    'text-generation',
                    model='./trained_model',
                    tokenizer='./trained_model',
                    device=device,
                    trust_remote_code=True,
                )

                print("Training completed.")

        threading.Thread(target=train_thread).start()

        return jsonify({'message': 'Training has started'}), 202

    @app.route('/chat', methods=['POST'])
    def chat():
        global nlp_pipeline
        if nlp_pipeline is None:
            return jsonify({'error': 'Model is not trained yet'}), 400

        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        # Generate a response using the pipeline
        response = nlp_pipeline(
            user_input,
            max_length=512,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_return_sequences=1,
            eos_token_id=nlp_pipeline.tokenizer.eos_token_id,
            pad_token_id=nlp_pipeline.tokenizer.pad_token_id,
        )

        # Extract and clean up the generated text
        bot_response = response[0]['generated_text'][len(user_input):].strip()

        return jsonify({'response': bot_response})

    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
