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
sft_model = load_model(sft_model_name).to(device)
#untune_model = load_model(untune_model_name)
qwen_model = load_model(qwen_model_name).to(device)

# base と sft_tokenizer で ChatTemplate の有無があるので分離
sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_name)
#base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)


#prompt = "次の文を中国語に翻訳してください．\nあなたの名前はなんですか．"
#prompt = "大規模言語モデルについて，100文字以内で説明してください．"
prompt = "次の文に続く文を生成してください．吾輩は猫である．"
#prompt = "次の文に続く文を生成してください．春はあけぼの，"
#prompt = "次の文章を翻訳してください．"
#prompt = "List the American presidents in order of age."
#prompt = "Can you explain large-scale language models?"

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


max_length = 30

print(sft_model.device)
#base_model_inputs = base_tokenizer([prompt], return_tensors="pt").to(untune_model.device)
sft_model_inputs = sft_tokenizer([sft_text], return_tensors="pt").to(sft_model.device)
qwen_model_inputs = qwen_tokenizer([sft_text], return_tensors="pt").to(qwen_model.device)

#base_input_ids = base_model_inputs["input_ids"]
sft_input_ids = sft_model_inputs["input_ids"]
qwen_input_ids = qwen_model_inputs["input_ids"]

#base_eos_token_id = base_tokenizer.eos_token_id
sft_eos_token_id = sft_tokenizer.eos_token_id
qwen_eos_token_id = qwen_tokenizer.eos_token_id

# ChatTemplate の扱い方に真剣に悩んでいる，どうしようか
# とりあえず，qwen と一緒に出力させてみるか?
# これって，ChatTemplate 分の出力って含まれてるんだろうか，いつも出力させてる時は見えてないけど

model_index = ["jp", "qwen"]

#print("base_inputs", base_input_ids)
print("sft_inputs", sft_input_ids)
print("qwen_inputs", qwen_input_ids)


def simultaneous_generation(tokenizers, models, input_ids, max_length):
    sft_generating = True
    # 一周目の実行がちょっと困る感じがあって，そこがmh微妙にめんどくさくて困る

    sequence = []
    for i in range(max_length):
        if sft_generating:
            next_token_ids = {}
            next_tokens = {}
            next_logits = {}
            tmp_logits = {}

            print(f"\n---- turn{i} ----")

            # トークン生成
            print("<generate_token>")
            for j in range(len(models)):
                output = models[j](input_ids[j])

                next_tokens_logits = output.logits[0, -1, :]
                #if(model_index[j]=='qwen'):
                #    next_tokens_logits *= 0.8
                tmp_logits[model_index[j]] = next_tokens_logits

                sorted_ids = torch.argsort(next_tokens_logits, descending=True)
                next_token_id = sorted_ids[0]
                next_token_ids[model_index[j]] = next_token_id

                next_token = tokenizers[j].decode(next_token_id)
                next_tokens[model_index[j]] = next_token

                next_logits[model_index[j]] = next_tokens_logits[next_token_id.item()]

                # EOS ならば，生成終了
                if next_token_id.item() == sft_eos_token_id:
                    sft_generating = False

                print(model_index[j], "  ", repr(tokenizers[j].decode(next_token_id.item())), next_tokens_logits[next_token_id.item()].item())
                

            # 確率取得・選択
            print("\n<choice_next_token>")
            max_logit = -10000
            max_token_id = None
            max_token = None
            max_model = 0
            for j in range(len(models)):
                print(model_index[j])
                cnt = 0
                for _, nt in next_tokens.items():
                    nti = tokenizers[j].encode(nt)[-1]
                    #print(nti)
                    #print(nti, next_token_ids[model_index[j]])

                    if cnt==j:
                        logit = next_logits[model_index[j]].item()
                    else:
                        logit = tmp_logits[model_index[j]][nti].item()

                    nti = torch.tensor([nti])
                    print("  ", repr(tokenizers[j].decode(nti.item())), logit)
                    if logit > max_logit:
                        max_logit = logit
                        max_token = tokenizers[j].decode(nti)
                        max_model = j

                    cnt+=1

            print("\nmax_token", max_token, max_logit)
            RED = '\033[31m'
            END = '\033[0m'
            if max_model==1:
                sequence.append(RED+max_token+END)
            else:
                sequence.append(max_token)

            # logit 最大のトークンを末尾に追加
            print("\n<add next_tokens>")
            for j in range(len(model_index)):
                max_token_id = torch.tensor(tokenizers[j].encode(max_token)).to(device)
                input_ids[j] = torch.cat(
                    [input_ids[j], max_token_id.unsqueeze(0)], dim=-1
                ).to(models[j].device)
            print(", ".join(sequence))


tokenizers = [sft_tokenizer, qwen_tokenizer]
models = [sft_model, qwen_model]
input_ids = [sft_input_ids, qwen_input_ids]
tokens = simultaneous_generation(tokenizers, models, input_ids, max_length)
