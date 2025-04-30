import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデル名を変更するときの拡張性が激しく物足りないので，ここは工夫が欲しい

base_model_name = "llm-jp/llm-jp-3-3.7B"
sft_model_name = "llm-jp/llm-jp-3-980m-instruct3"
untune_model_name = "llm-jp/llm-jp-3-980m"
# device_map をサンプルの時点から実装してるのはとても偉いと思う（当たり前か）


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    return model


# base_model = load_model(base_model_name)
sft_model = load_model(sft_model_name)
untune_model = load_model(untune_model_name)

# base と sft_tokenizer で ChatTemplate の有無があるので分離
sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_name)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)


prompt = "次の文に続く文を生成してください．\n吾輩は猫である．"
sft_messages = [
    {
        "role": "system",
        "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。",
    },
    {"role": "user", "content": prompt},
]

sft_text = sft_tokenizer.apply_chat_template(
    sft_messages, tokenize=False, add_generation_prompt=True
)

max_length = 20

base_model_inputs = base_tokenizer([prompt], return_tensors="pt").to(
    untune_model.device
)
sft_model_inputs = sft_tokenizer([sft_text], return_tensors="pt").to(sft_model.device)
base_input_ids = base_model_inputs["input_ids"]
sft_input_ids = sft_model_inputs["input_ids"]
base_eos_token_id = base_tokenizer.eos_token_id
sft_eos_token_id = sft_tokenizer.eos_token_id

# ChatTemplate の扱い方に真剣に悩んでいる，どうしようか
# とりあえず，qwen と一緒に出力させてみるか?
# これって，ChatTemplate 分の出力って含まれてるんだろうか，いつも出力させてる時は見えてないけど

model_index = ["ft", "untune"]

print("base_inputs", base_input_ids)
print("sft_inputs", sft_input_ids)


def simultaneous_generation(tokenizers, models, input_ids, max_length):
    sft_generating = True
    # 一周目の実行がちょっと困る感じがあって，そこがmh微妙にめんどくさくて困る

    sequence = []
    for i in range(max_length):
        if sft_generating:
            next_token_ids = {}
            tmp_logits = {}

            print(f"\n---- turn{i} ----")

            # トークン生成
            print("<generate_token>")
            for j in range(len(models)):
                output = models[j](input_ids[j])

                next_tokens_logits = output.logits[0, -1, :]
                tmp_logits[model_index[j]] = next_tokens_logits

                sorted_ids = torch.argsort(next_tokens_logits, descending=True)
                next_token_id = sorted_ids[0]
                next_token_ids[model_index[j]] = next_token_id

                # EOS ならば，生成終了
                if next_token_id.item() == sft_eos_token_id:
                    sft_generating = False

                print("  ", tokenizers[j].decode(next_token_id.item()), next_tokens_logits[next_token_id.item()])
                

            # 確率取得・選択
            print("\n<choice_next_token>")
            max_logit = -10000
            max_token_id = None
            max_token = None
            for j in range(len(models)):
                print(model_index[j])
                for _, nti in next_token_ids.items():
                    logit = tmp_logits[model_index[j]][nti.item()].item()
                    print("  ", tokenizers[j].decode(nti.item()), logit)
                    if logit > max_logit:
                        max_logit = logit
                        max_token_id = nti
                        max_token = tokenizers[j].decode(nti)

            print("\nmax_token", max_token, max_logit)
            sequence.append(max_token)


            # logit 最大のトークンを末尾に追加
            print("<add next_tokens>")
            for j in range(len(model_index)):
                input_ids[j] = torch.cat(
                    [input_ids[j], max_token_id.unsqueeze(0).unsqueeze(0)], dim=-1
                )
            print(", ".join(sequence))


tokenizers = [sft_tokenizer, base_tokenizer]
models = [sft_model, untune_model]
input_ids = [sft_input_ids, base_input_ids]
tokens = simultaneous_generation(tokenizers, models, input_ids, max_length)
