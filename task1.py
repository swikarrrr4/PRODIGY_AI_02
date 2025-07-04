

def create_training_data():
    with open("train.txt", "w", encoding="utf-8") as f:
        f.write("Once upon a time, there was a brave knight.\n")
        f.write("The knight fought dragons and saved the kingdom.\n")
        f.write("Peace returned to the land.\n")

def load_model_and_tokenizer():
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def train_model(tokenizer, model):
    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

    dataset = load_dataset("text", data_files={"train": "train.txt"})["train"]

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        logging_steps=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

def save_model(tokenizer, model):
    model.save_pretrained("./gpt2-finetuned")
    tokenizer.save_pretrained("./gpt2-finetuned")

def generate_text(model, tokenizer):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode("Once upon a time", return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=50, temperature=0.7, top_p=0.9, do_sample=True)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

# === Main Flow ===
create_training_data()
tokenizer, model = load_model_and_tokenizer()
train_model(tokenizer, model)
save_model(tokenizer, model)
generate_text(model, tokenizer)

#in terminal run -> python test.py
#wandb -> 3