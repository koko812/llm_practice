from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name1 = "Qwen/Qwen2.5-0.5B-Instruct"
model_name2 = "llm-jp/llm-jp-3-440m-instruct3"
# device_map をサンプルの時点から実装してるのはとても偉いと思う（当たり前か）
qwen_model = AutoModelForCausalLM.from_pretrained(
    model_name1,
    torch_dtype="auto",
    device_map="auto"
)
qwen_tokenizer = AutoTokenizer.from_pretrained(model_name1)

jp_model = AutoModelForCausalLM.from_pretrained(
    model_name2,
    torch_dtype="auto",
    device_map="auto"
)
jp_tokenizer = AutoTokenizer.from_pretrained(model_name2)


#元々は英語だが，何言ってるかわからんので日本語に変更した
prompt = "请100个字以内说明LLM。"
qwen_messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
jp_messages = [
    {"role": "system", "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"},
    {"role": "user", "content": prompt},
]

qwen_text = qwen_tokenizer.apply_chat_template(
    qwen_messages,
    tokenize=False,
    add_generation_prompt=True
)

jp_text = jp_tokenizer.apply_chat_template(
    jp_messages,
    tokenize=False,
    add_generation_prompt=True
)

# 翻訳の場合，翻訳した箇所が同じならばマージ
# 生成させたものを
# ( ならば，両方ともトークンを持ってるだろうので，計算できる
# 大規模　という文字列は qwen にはないかもしれない
# だしてモデルが進んだ時に生成できたら，その確率を採用する
# 出さなければ，別の方のモデルで推論を続ける
# 確率のダイナミクスが違うかも（レンジが違う)
# 平均的な値と振れ幅が変わってこないとわからない

max_length = 50

qwen_model_inputs = qwen_tokenizer([qwen_text], return_tensors="pt").to(qwen_model.device)
qwen_input_ids = qwen_model_inputs['input_ids']
qwen_eos_token_id = qwen_tokenizer.eos_token_id

jp_model_inputs = jp_tokenizer([jp_text], return_tensors="pt").to(jp_model.device)
jp_input_ids = jp_model_inputs['input_ids']
jp_eos_token_id = jp_tokenizer.eos_token_id

qwen_tokens = []
jp_tokens = []

with torch.no_grad():
    qwen_generating = True
    jp_generating  = True
    for i in range(max_length):
        if qwen_generating:
            qwen_output = qwen_model(qwen_input_ids)
            qwen_next_token_logits = qwen_output.logits[0, -1, :]
            qwen_sorted_ids = torch.argsort(qwen_next_token_logits, descending=True)
            qwen_next_token_id = qwen_sorted_ids[0]
            qwen_next_token = qwen_tokenizer.decode(qwen_next_token_id)
            qwen_tokens.append(qwen_next_token)
            if qwen_next_token_id.item() == qwen_eos_token_id:
                qwen_generating = False
            qwen_input_ids = torch.cat([qwen_input_ids, qwen_next_token_id.unsqueeze(0).unsqueeze(0)], dim=-1)

        if jp_generating:
            jp_output = jp_model(jp_input_ids)
            jp_next_token_logits = jp_output.logits[0, -1, :]
            jp_sorted_ids = torch.argsort(jp_next_token_logits, descending=True)
            jp_next_token_id = jp_sorted_ids[0]
            jp_next_token = jp_tokenizer.decode(jp_next_token_id)
            jp_tokens.append(jp_next_token)
            if jp_next_token_id.item() == jp_eos_token_id:
                jp_generating = False
            jp_input_ids = torch.cat([jp_input_ids, jp_next_token_id.unsqueeze(0).unsqueeze(0)], dim=-1)


#print(qwen_tokenizer.batch_decode(qwen_input_ids))
#print(jp_tokenizer.batch_decode(jp_input_ids))
print (' '.join(qwen_tokens))
print(' '.join(jp_tokens))