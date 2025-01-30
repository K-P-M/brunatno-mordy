import streamlit as st
from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
import torch

st.title("💬 Brunatno - mordy")


@st.cache_resource
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="brunatno-mordy-7B-Instruct-translate",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )

    model.generation_config.repetition_penalty = 2.0

    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
    )

    return model, tokenizer


model, tokenizer = load_model()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Say something")

if prompt:
    messages = [
        {
            "role": "system",
            "content": "Odpowiadaj w stylu Bartosza Walaszka. Ogranicz się do dwóch zdań."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    with st.chat_message("user"):
        st.write(prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            outputs = model.generate(input_ids=inputs, max_new_tokens=200, use_cache=True)
            response = tokenizer.batch_decode(outputs)

            start = "[/INST]"
            end = "</s>"

            part1, part2 = response[0].split("[/INST]", 1)
            result = part2.split("</s>", 1)[0]
            # result = response[0]

            st.write(result)
            st.session_state["messages"].append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"An error occurred: {e}")
