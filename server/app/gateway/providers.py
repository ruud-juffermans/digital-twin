"""The provider table: every model API the gateway can talk to.

All four speak the OpenAI chat-completions protocol, so a provider is a base
URL, the settings fields holding its key and model, and a label for the admin
panel. Pick one with LLM_PROVIDER (chat) and EMBEDDING_PROVIDER (vectors).
"""

PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "key_field": "deepseek_api_key",
        "model_field": "deepseek_model",
    },
    "kimi": {
        "label": "Moonshot (Kimi)",
        "base_url": "https://api.moonshot.ai/v1",
        "key_field": "moonshot_api_key",
        "model_field": "kimi_model",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "key_field": "openai_api_key",
        "model_field": "openai_model",
    },
    "gemini": {
        "label": "Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_field": "gemini_api_key",
        "model_field": "gemini_model",
    },
}
