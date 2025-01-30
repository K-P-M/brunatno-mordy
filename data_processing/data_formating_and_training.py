from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import Dataset
from transformers import EvalPrediction
import evaluate
import numpy as np

import torch
import json
from nltk.translate.meteor_score import meteor_score



def load_and_format_training_data(file_path: str) -> list[list[dict]]:
    system_context = {
        "role": "system",
        "content": "Odpowiadaj w stylu Bartosza Walaszka."
    }

    formatted_data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            formatted_data.append([
                system_context,
                {"role": "user", "content": data["user"]},
                {"role": "assistant", "content": data["assistant"]}
            ])

    return formatted_data


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "speakleash/Bielik-7B-Instruct-v0.1",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "mistral",
)

def formatting_prompts_func(examples):
    convos = examples["conversation"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False) for convo in convos]
    return { "text" : texts }


train_data = load_and_format_training_data("descriptions_gpt.jsonl")

data_in_dict = {"conversation": train_data}

dataset = Dataset.from_dict(data_in_dict)

train_test_split = dataset.train_test_split(test_size=0.1, seed=420)

train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
test_dataset = test_dataset.map(formatting_prompts_func, batched = True,)


model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
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
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 2000,
        learning_rate = 5e-5,
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

trainer_stats = trainer.train()
model.save_pretrained_merged("brunatno-mordy-7B-Instruct-Torpeda", tokenizer, save_method = "lora")