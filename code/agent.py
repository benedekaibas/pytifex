from typing import Any, Optional
from pydantic import BaseModel, Field, HttpUrl
import os
import httpx
import argparse

import generate_json
from prompts import build_expert_prompt


class GetAccessToGemini(BaseModel):
    """LLM agent to send requests to Google Gemini API."""

    url: str = "https://generativelanguage.googleapis.com/v1beta"
    model: str = Field(..., description="Model id, e.g. 'gemini-2.5-flash'")
    api_base: HttpUrl = Field(HttpUrl(url), description="Google Gemini API base")
    timeout: float = Field(120.0, gt=0, description="Timeout (seconds)")
    token: str = Field(..., description="Google API Key")

    AVAILABLE_MODELS: list[str] = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]

    def setup(
        self,
        model: Optional[str] = None,
        api_base: Optional[HttpUrl] = None,
        timeout: Optional[float] = None,
        token: Optional[str] = None,
    ) -> None:
        """Validated updates (optional)."""
        updates: dict[str, Any] = {}
        if model is not None:
            updates["model"] = model
        if api_base is not None:
            updates["api_base"] = api_base
        if timeout is not None:
            updates["timeout"] = timeout
        if token is not None:
            updates["token"] = token
        if updates:
            new_self = self.model_copy(update=updates)
            self.model, self.api_base, self.timeout, self.token = (
                new_self.model,
                new_self.api_base,
                new_self.timeout,
                new_self.token,
            )

    def communicate(self, prompt: str) -> str:
        """Send a prompt to Google Gemini and return the text reply."""
        base = str(self.api_base).rstrip("/")
        url = f"{base}/models/{self.model}:generateContent"

        headers = {"Content-Type": "application/json", "x-goog-api-key": self.token}

        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            try:
                candidate = data.get("candidates", [{}])[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [{}])
                msg = parts[0].get("text")
            except (IndexError, AttributeError):
                msg = None

            if not msg:
                raise ValueError(f"Invalid Gemini response: {data}")
            return str(msg)

        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"HTTP {e.response.status_code} from {e.request.method} {e.request.url}: {e.response.text}"
            ) from e
        except httpx.HTTPError as e:
            raise ValueError(f"Network error contacting Google Gemini: {e}") from e

    def predict(self, prompt: str) -> str:
        return self.communicate(prompt)

    def print_models(self):
        """Display available models."""
        print("Available Gemini models:")
        for i, model in enumerate(self.AVAILABLE_MODELS, 1):
            print(f"{i}. {model}")

    def cli_parser(self):
        """Creating a CLI to select different LLM models."""
        parser = argparse.ArgumentParser(description="Select the Gemini model to use")
        parser.add_argument(
            "--model",
            choices=self.AVAILABLE_MODELS,
            default=self.model,
            help=f"Choose model from available options (default: {self.model})",
        )
        parser.add_argument(
            "--list-models",
            action="store_true",
            help="List all available models and exit",
        )
        return parser


if __name__ == "__main__":
    token = os.environ.get("GEMINI_API_KEY")
    if not token:
        raise ValueError("Please set GEMINI_API_KEY environment variable")

    agent = GetAccessToGemini(
        model="gemini-2.5-flash",
        token=token,
        api_base=HttpUrl("https://generativelanguage.googleapis.com/v1beta"),
        timeout=320.0,
    )

    parser = agent.cli_parser()
    args = parser.parse_args()

    if args.list_models:
        agent.print_models()
        exit(0)

    if args.model:
        agent.setup(model=args.model)

    print(f"Using model: {agent.model}")
    print("Generating type checker divergence examples...")

    prompt = build_expert_prompt()
    response = agent.predict(prompt)
    print("\n" + "=" * 60)
    print("GENERATED CODE EXAMPLES:")
    print("=" * 60)
    print(response)

    print("\n[INFO] Processing and saving output...")

    examples = generate_json.parse_generated_content(response)

    if examples:
        generate_json.save_output(examples, response, agent.model)
    else:
        print("[WARNING] No code examples found to save.")

