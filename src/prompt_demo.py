from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载开源中文大模型（Qwen-1.8B，约 3.5GB，第一次运行会自动下载）
model_name = "Qwen/Qwen2-1.5B-Instruct"
# model_name = "Qwen/Qwen1.5-0.5B-Chat"
# model_name = "THUDM/glm-4-9b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float16,
    # local_files_only=True  # ✅ 只用本地，不联网
)
model.eval()

# 读入评论
with open("./data/comments.txt", "r", encoding="utf-8") as f:
    comments = [line.strip() for line in f if line.strip()]

# 定义 Prompt 模板
def zero_shot_prompt(comment):
    return f""" 
    评论：{comment},这是一个电商的评论数据，你是一个电商评论分析助手。请对以用户的评论进行分析：
      1. 判断情感倾向：正面 / 负面  
      2. 提取1-2个关键词标签（如“物流快”“包装差”）。
      最后严格按照以下格式输出：
        情感：正面 或 负面 
        关键词：在原始评论中提取一两个关键词
    """
def few_shot_prompt(comment):
    return f""" 
    电商评论：{comment},这是一个电商的评论数据，你是一个电商评论分析助手。然后对上述的电商评论进行分析，不需要说明分析过程以及需要什么工具，只需输出该条电商评论的情感类型（例如正面或负面）以及标签。
 """
def cot_prompt(comment):
    return f""" 
    你是一个电商评论分析助手。请按以下步骤分析评论：
    步骤1：仔细阅读评论，判断用户整体情绪是满意（正面）还是不满意（负面）。  
    步骤2：找出评论中提到的具体方面，如“物流”“包装”“客服”“质量”等，并总结关键词。  
    步骤3：综合以上，输出最终结果。
    评论：{comment}
    """

# 逐条分析
for comment in comments:
    prompt = few_shot_prompt(comment)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100,do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print("原始 response：", response)
    # print("原始评论：", comment)
    print("模型输出：", response.split("请严格按照以下格式输出：")[-1].strip())
