import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import urllib.request
import time

# ----------------------------
# 1. 下载并加载评论数据（约1000条）
# ----------------------------
def load_comments():
    # url = "https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"
    # local_file = "jd_binary.csv"
    # if not os.path.exists(local_file):
    #     print("正在下载京东评论数据集（约1.5MB）...")
    #     urllib.request.urlretrieve(url, local_file)
    df = pd.read_csv("./data/ChnSentiCorp_htl_all.csv")
    # 只取前1000条，label=1为正面，0为负面
    df = df.head(100).copy()
    df['comment'] = df['review']
    df['true_sentiment'] = df['label'].map({1: '正面', 0: '负面'})
    return df[['comment','true_sentiment']].values.tolist()

# ----------------------------
# 2. 三种 Prompt 策略
# ----------------------------
def zero_shot_prompt(comment):
    return f"""你是一个酒店评论分析助手。请对以下用户评论进行分析：
1. 判断情感倾向：正面 / 负面
2. 提取1-2个关键词标签（如“卫生差”“服务好”“隔音差”）

评论：{comment}

请严格按照以下格式输出，不要任何解释：
情感：正面/负面
标签：xxx, xxx
"""

def few_shot_prompt(comment):
    return f"""你是一个酒店评论分析助手。请参考以下示例，对新评论进行分析：

示例1：
评论：房间干净整洁，前台服务热情，地理位置也很方便。
情感：正面
标签：卫生好, 服务好

示例2：
评论：半夜隔壁吵得睡不着，空调也不制冷，非常失望。
情感：负面
标签：隔音差, 空调故障

现在请分析以下评论：
评论：{comment}

请严格按照以下格式输出，不要任何解释：
情感：正面/负面
标签：xxx, xxx
"""

def cot_prompt(comment):
    return f"""你是一个酒店评论分析助手。请按以下步骤分析评论：
步骤1：判断用户对酒店的整体体验是满意（正面）还是不满意（负面）。
步骤2：找出提到的具体方面，如“卫生”“服务”“隔音”“设施”“位置”等，并总结1-2个关键词。
步骤3：按格式输出结果。

评论：{comment}

请严格按照以下格式输出，不要任何解释：
情感：正面/负面
标签：xxx, xxx
"""

# ----------------------------
# 3. 模型加载（使用 Qwen2-1.5B-Instruct）
# ----------------------------
print("正在加载 Qwen2-1.5B-Instruct 模型（首次运行会自动下载）...")
model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.eval()

# ----------------------------
# 4. 推理函数
# ----------------------------
def generate_response(prompt, max_new_tokens=70):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 确保结果稳定
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只保留模型生成部分（去掉 Prompt）
    if "请严格按照以下格式输出,不要任何解释：" in response:
        return response.split("请严格按照以下格式输出,不要任何解释：")[-1].strip()
    else:
        return response.strip()

# ----------------------------
# 5. 主实验流程
# ----------------------------
# 加载和准备待分析的酒店评论数据。通过 load_comments() 函数读取所有评论，comments 变量中存储着每条评论及其真实情感标签。
# 初始化一个空列表 results 用于存储处理结果。接着输出要处理的评论总数及将使用三种 Prompt 策略（零样本、少样本、思维链/COT）。
# 最后，使用 tqdm 让进度可视化，遍历每条评论及其真实情感，为后续处理做准备。
comments = load_comments()
# print(f"原始评论：{comments}")
results = []

print(f"开始处理 {len(comments)} 条评论，三种 Prompt 策略...")

for i, (comment, true_sent) in enumerate(tqdm(comments)):
    # 跳过空评论
    if not isinstance(comment, str) or len(comment.strip()) < 5:
        continue

    row = {"comment": comment, "true_sentiment": true_sent}

    # 零样本
    try:
        resp0 = generate_response(zero_shot_prompt(comment))
        row["zero_shot"] = resp0
    except Exception as e:
        row["zero_shot"] = f"ERROR: {str(e)}"

    # 少样本
    try:
        resp1 = generate_response(few_shot_prompt(comment))
        row["few_shot"] = resp1
    except Exception as e:
        row["few_shot"] = f"ERROR: {str(e)}"

    # 思维链
    try:
        resp2 = generate_response(cot_prompt(comment))
        row["cot"] = resp2
    except Exception as e:
        row["cot"] = f"ERROR: {str(e)}"

    results.append(row)

    # 避免显存溢出，每50条清一次缓存
    if i % 50 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

# ----------------------------
# 6. 保存为 CSV
# ----------------------------
df_out = pd.DataFrame(results)
df_out.to_csv("./output/prompt_experiment_results.csv", index=False, encoding="utf-8-sig")
print("\n实验完成！结果已保存至：prompt_experiment_results.csv")