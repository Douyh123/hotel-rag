from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
# 加载开源中文大模型
model_name = "Qwen/Qwen2-1.5B-Instruct"
# model_name = "Qwen/Qwen1.5-0.5B-Chat"
# model_name = "THUDM/glm-4-9b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float16,
    # local_files_only=True  # 只用本地，不联网
)
model.eval()

# 读入评论
def readtxt():
    with open("./data/comments.txt", "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]

def load_comments():
    # url = "https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"
    # local_file = "jd_binary.csv"
    # if not os.path.exists(local_file):
    #     print("正在下载京东评论数据集（约1.5MB）...")
    #     urllib.request.urlretrieve(url, local_file)
    df = pd.read_csv("./data/ChnSentiCorp_htl_all.csv")
    # 只取前1000条，label=1为正面，0为负面
    df = df.head(10).copy()
    df['comment'] = df['review']
    # df['true_sentiment'] = df['label'].map({1: '正面', 0: '负面'})
    return df['comment'].values.tolist()

# 定义 Prompt 模板
def zero_shot_prompt(comment):
    return f""" 
    评论：{comment},这是一个酒店的评论数据。你是一个酒店评论分析专家。请对用户的评论严格按以下要求分析：
      1. 判断情感倾向：正面 / 负面  
      2. 提取1-2个关键词标签（如“环境差”“服务差”）。
    分析完过后严格按照以下示例进行输出，一定不要输出其他多余内容。
    示例：
      情感倾向：正面。
      关键词标签：环境优美，服务态度好。
    """
def few_shot_prompt(comment):
    return f""" 
    电商评论：{comment},这是一个电商的评论数据，你是一个电商评论分析助手。然后对上述的电商评论进行分析，不需要说明分析过程以及需要什么工具，只需输出该条电商评论的情感类型（例如正面或负面）以及标签。
 """
def cot_prompt(comment):
    return f""" 
   你是一个电商评论分析助手。请严格按以下规则处理：
    规则1：只输出两行，格式如下：
    情感：正面/负面
    标签：关键词1, 关键词2
    规则2：关键词必须来自评论原文的核心优点或缺点，最多2个，用中文逗号分隔。
    规则3：不要输出“提取词”、不要解释、不要总结、不要任何额外文字。
    示例：
    评论：房间干净，服务热情，就是空调有点吵。
    情感：正面
    标签：房间干净, 服务热情

    现在处理以下评论：
    评论：{comment}

    输出：
    """

def f(comments):
# 逐条分析
    for comment in comments:
        prompt = zero_shot_prompt(comment)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100,do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n" + "="*50)
        # print("原始 response：", response)
        print("原始评论：", comment)
        print("模型输出：", response.split("示例：")[-1].strip())
        

if __name__ == "__main__":
    comments = load_comments()
    # print(comments)
    f(comments)


