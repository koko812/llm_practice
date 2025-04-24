from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# device_map をサンプルの時点から実装してるのはとても偉いと思う（当たり前か）
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 元々は英語だが，何言ってるかわからんので日本語に変更した
prompt = "LLMとはなんですか．100文字以内で答えてください．"
messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# tokenizer の出力は，input_ids と attention_mask の辞書（それぞれは tensor）
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print(model_inputs)
output = model(**model_inputs)

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("--- output ---")
print(output)
print("--- logits ---")
print(output.logits, output.logits.shape)
# axis は，でかい方から数えていく感じで
# 一番内側で max を取りたい時は基本的に -1 でOKっぽい
print(torch.max(output.logits, axis=2))
generated_ids = torch.max(output.logits, axis=2).indices
# バッチデコードのはずが，普通に squeeze させられるのが謎
# id を一つずつ見ていないとか，そういう種類でのバッチ化ってっことですか
response = tokenizer.batch_decode(
    torch.squeeze(generated_ids), skip_special_tokens=True
)
print(response)
