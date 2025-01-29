import time
import streamlit as st
from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
import torch

st.title("💬 Brunatno - modry")


@st.cache_resource
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
    )

    return model, tokenizer


model, tokenizer = load_model()

prompt = st.chat_input("Say something")

if prompt:
    messages = [
        {
            "role": "system",
            "content": "Odpowiadaj w stylu Bartosza Walaszka."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]



    with st.chat_message("user"):
        st.write(prompt)
        time.sleep(1)

    with st.chat_message("assistant"):
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            outputs = model.generate(input_ids=inputs, max_new_tokens=400, use_cache=True)
            response = tokenizer.batch_decode(outputs)
            st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
