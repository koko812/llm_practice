import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# dataset prepairation
jpn_dev_path = "jp-zho/jpn_Jpan.dev"
zho_dev_path = "jp-zho/zho_Hans.dev"

jpn_test_path = "jp-zho/jpn_Jpan.devtest"
zho_test_path = "jp-zho/zho_Hans.devtest"


with open(jpn_dev_path, 'r') as f:
    jpn_dev = [line for line in f]

print(len(jpn_dev))

#base_model_name = "llm-jp/llm-jp-3-3.7B"
qwen_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #device_map="auto",
        torch_dtype="auto",
    )
    return model


#base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)

# use_qwen
qwen_model = load_model(qwen_model_name)
responses = []

for i in range(len(jpn_dev)):
    prompt = f"次の文を中国語に翻訳してください.\n{jpn_dev[i]}"

    qwen_messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]

    qwen_text = qwen_tokenizer.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True
    )

    qwen_model_inputs = qwen_tokenizer([qwen_text], return_tensors="pt").to(qwen_model.device)
    qwen_eos_token_id = qwen_tokenizer.eos_token_id

    generated_ids = qwen_model.generate(
        **qwen_model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(qwen_model_inputs.input_ids, generated_ids)
    ]

    response = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].replace('\n' , '')
    print(response)
    responses.append(response)

output_file_path = "qwen_generated_jp2zho_trans.txt"
with open(output_file_path, 'w') as f:
    f.write('\n'.join(responses))