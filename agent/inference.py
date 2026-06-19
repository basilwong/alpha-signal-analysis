"""
Model inference layer. Supports both DashScope (Qwen Cloud) and Modal (fine-tuned model).
Configuration switch allows seamless transition between backends.
"""
from openai import OpenAI
from .config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, REASONING_MODEL, INFERENCE_BACKEND, MODAL_ENDPOINT


def get_client():
    """Get the appropriate inference client based on configuration."""
    if INFERENCE_BACKEND == "modal" and MODAL_ENDPOINT:
        return OpenAI(api_key="not-needed", base_url=MODAL_ENDPOINT)
    else:
        return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


def generate_signal(article_text: str, source_type: str, memory_context: str, enable_thinking: bool = True) -> dict:
    """Generate a signal vector with memory-augmented reasoning."""
    client = get_client()
    model = REASONING_MODEL if INFERENCE_BACKEND == "dashscope" else "default"

    system_prompt = f"""You are a quantitative NLP signal generator for the quantum computing sector with persistent memory.

You have accumulated the following knowledge from previous sessions:

{memory_context}

Use this memory to make more informed predictions. Reference specific past events when relevant.
If your past predictions were wrong, adjust your confidence accordingly.

For every piece of news, produce a signal vector scoring these tickers: IONQ, RGTI, QBTS, QUBT, QNT, IBM, GOOGL, MSFT, HON, NVDA
Score range: -2.0 to +2.0 for pure-play companies. GOOGL, MSFT, NVDA should always be 0.0 (too diversified).
Include chain_of_thought explaining your reasoning, referencing memories where applicable.
Output ONLY valid JSON."""

    user_prompt = f"Analyze this {source_type} article and generate a cross-sectional signal vector:\n\n{article_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
        extra_body={"enable_thinking": enable_thinking}
    )

    content = response.choices[0].message.content
    thinking = ""
    if hasattr(response.choices[0].message, 'reasoning_content'):
        thinking = response.choices[0].message.reasoning_content or ""

    return {"content": content, "thinking": thinking}
