import openai
import json

client = openai.OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key="sk-c8cd17b77764463da9bd39bcc6ef552f" 
)

def get_llm_judgment(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an expert investment advisor. Please strictly follow JSON format and ensure the 'score' is between -1 and 1."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    content = response.choices[0].message.content.strip()

    # 清理掉 Markdown 代码块
    if content.startswith('```json'):
        content = content.lstrip('```json').rstrip('```').strip()
    elif content.startswith('```'):
        content = content.lstrip('```').rstrip('```').strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        print("LLM返回内容无法解析为JSON：", content)
        raise e

    # 确保包含必要字段
    if "score" not in result:
        result["score"] = 0.0  # 默认值
    if "reasoning" not in result:
        result["reasoning"] = "No reasoning provided by LLM."

    # 将 score 限制在 [-1,1]
    try:
        score = float(result["score"])
        # 这里可以选择 Clamp 或放缩：
        # Clamp
        score = max(min(score, 1.0), -1.0)
        # 或者 Rescale（假设 LLM score 原范围 [-5,5]，你可以根据经验调整）
        # score = max(min(score, 5.0), -5.0)
        # score = score / 5.0
        result["score"] = score
    except (ValueError, TypeError):
        result["score"] = 0.0  # 如果解析失败，默认 0.0

    return result
