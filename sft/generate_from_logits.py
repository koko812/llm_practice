from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# device_map をサンプルの時点から実装してるのはとても偉いと思う（当たり前か）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#元々は英語だが，何言ってるかわからんので日本語に変更した
prompt = "LLMとはなんですか．100文字以内で答えてください．"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
max_length = 100

with torch.no_grad():
    output = model(**model_inputs)
    print('--- inputs ---')
    print(model_inputs)
    for i in range(max_length):
        generated_ids = torch.max( output.logits , axis = 2).indices
        output = model(generated_ids, attention_mask =torch.ones(generated_ids.shape[-1]))
        print(generated_ids)

    generated_ids = torch.max( output.logits , axis = 2).indices

print('--- outputs ---')
print(output)
print('--- ids ---')
print(generated_ids)


response = tokenizer.batch_decode(torch.squeeze( generated_ids ), skip_special_tokens=True)
print(response)

'''
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
output = model(**model_inputs)

print('--- inputs ---')
print(model_inputs)
print('--- outputs ---')
print(output)
generated_ids = torch.max( output.logits , axis = 2).indices
print('--- ids ---')
print(generated_ids)

'''