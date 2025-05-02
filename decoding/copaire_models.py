import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデル名を変更するときの拡張性が激しく物足りないので，ここは工夫が欲しい
#とはいえ，随分と，当初に比べる見やすいコードになったのではないだろうか

base_model_name = "llm-jp/llm-jp-3-3.7B"
sft_model_name = "llm-jp/llm-jp-3-1.8B-instruct3"
#sft_model_name = "google/gemma-2-2b-jpnhit"
untune_model_name = "llm-jp/llm-jp-3-980m"
qwen_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# device_map をサンプルの時点から実装してるのはとても偉いと思う（当たり前か）

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #device_map="auto",
        torch_dtype="auto",
    )
    return model


# base_model = load_model(base_model_name)
# sft_model = load_model(sft_model_name).to(device)
#untune_model = load_model(untune_model_name)
qwen_model = load_model(qwen_model_name).to(device)

# base と sft_tokenizer で ChatTemplate の有無があるので分離
sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_name)
#base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)


prompt = "次の文に続く文を生成してください．吾輩は猫である．"

sft_messages = [
    {
        "role": "system",
        "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。",
    },
    {"role": "user", "content": prompt},
]

#sft_messages = [
#    {"role": "user", "content": prompt},
#]


qwen_messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": prompt},
]

sft_text = sft_tokenizer.apply_chat_template(
    sft_messages, tokenize=False, add_generation_prompt=True
)
qwer_text = qwen_tokenizer.apply_chat_template(
    qwen_messages, tokenize=False, add_generation_prompt=True
)
