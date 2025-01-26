from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import Dataset

import torch

#wczytanie modelu
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "speakleash/Bielik-7B-Instruct-v0.1",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
# format danych
# chat =[
#     {
#         "role": "system",
#     "content":"Odpowiadaj w stylu Bartosza Walaszka"
# },
# {
#     "role": "user",
#     "content": "Jestem Cezary Baryka, od jakichś 20 minut"
# },
# {
#     "role": "assistant",
#     "content": "I już trochę żałuję, że go kupiłem. W nocy zimno jak cholera, a w dzień upał nie do wytrzymania. Zero przewiewu i brak kanalizacji"
# },
# ]


# przypisanie chat template do tokenizera
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "mistral",
)
#funckja formatująca dane
def formatting_prompts_func(examples):
    convos = examples["conversation"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False) for convo in convos]
    return { "text" : texts }

# wczytanie danych
train_data =

data_in_dict = {"conversation": train_data}

# Utwórz obiekt Dataset
dataset = Dataset.from_dict(data_in_dict)

# Podział na zbiór treningowy i testowy
train_test_split = dataset.train_test_split(test_size=0.1, seed=420)

train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

#formatowanie danych
train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
test_dataset = test_dataset.map(formatting_prompts_func, batched = True,)

#Ustawienie modelu
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0.1,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 1337,
    use_rslora = False,
    loftq_config = None,
)

#inicjalizacja trenera

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 2000,
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 2137,
        output_dir = "outputs",
        report_to = "none",
        evaluation_strategy = "steps",
        eval_steps = 200,
        save_steps = 200,
        load_best_model_at_end = True,
    ),
)
#trenowanie modelu
trainer_stats = trainer.train()
