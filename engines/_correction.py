"""Shared Claude correction logic used by claude-vision and tesseract+claude engines."""

import getpass
import os
import sys
from pathlib import Path

from engines._colors import yellow, red, Spinner

PROJECT_DIR = Path(__file__).resolve().parent.parent

CORRECTION_SYSTEM_PROMPT = """\
You are an expert proofreader specializing in historical Norwegian text. \
You are given raw OCR output from a 1950s Norwegian newspaper scan. The OCR \
contains errors from misread characters, especially in fraktur/antiqua typefaces.

Your task is to correct obvious OCR errors while preserving the original text \
as faithfully as possible.

Rules:
- Fix clear character-level OCR errors (e.g. rn\u2192m, li\u2192h, cl\u2192d, \
\u00f8\u2192o, \u00e6\u2192ae confusions, doubled/missing letters).
- Fix garbled words where the correct Norwegian word is obvious from context.
- Preserve the original paragraph structure, line breaks, and formatting exactly.
- Preserve \u00ab\u00bb quotes, headings, and verse formatting.
- Do NOT rewrite, modernize spelling, or rephrase. Keep the 1950s Norwegian \
orthography (e.g. \u00abbleven\u00bb not \u00abblitt\u00bb, \u00abhvad\u00bb not \u00abhva\u00bb).
- If a word is ambiguous and you cannot determine the correct reading, leave it \
as-is with [?] after it.
- NEVER delete or remove words. Every word in the input must appear in the output. \
If a word looks wrong but you are unsure of the correction, leave it exactly as-is.
- Do NOT add commentary or notes. Output only the corrected text.\
"""

CORRECTION_USER_PROMPT = """\
Correct OCR errors in the following text from a 1950s Norwegian newspaper. \
Fix only clear misreadings. Preserve original spelling and structure.\n\n{text}\
"""

BEDROCK_MODEL_MAP = {
    "claude-sonnet-4-20250514": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4-20250514": "us.anthropic.claude-opus-4-20250514-v1:0",
}


def _load_dotenv():
    from dotenv import load_dotenv
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def _has_anthropic_api_key():
    """Return True if ANTHROPIC_API_KEY is available in env or .env file."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    _load_dotenv()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _has_aws_credentials():
    """Return True if AWS credentials are available (profile, env vars, or default session)."""
    if os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID"):
        return True
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None
    except Exception:
        return False


def _prompt_api_key():
    """Prompt for Anthropic API key and save to .env."""
    print(yellow("No ANTHROPIC_API_KEY found in environment or .env file."))
    print(yellow("No AWS credentials detected for Bedrock."))
    key = getpass.getpass("Enter your Anthropic API key: ").strip()
    if not key:
        print(red("Error: API key is required."), file=sys.stderr)
        sys.exit(1)

    env_file = PROJECT_DIR / ".env"
    env_file.write_text(f"ANTHROPIC_API_KEY={key}\n", encoding="utf-8")
    os.environ["ANTHROPIC_API_KEY"] = key
    print("Saved to .env")


def has_claude_auth():
    """Return True if any Claude authentication method is available."""
    return _has_anthropic_api_key() or _has_aws_credentials()


def get_claude_client(region="eu-north-1"):
    """Create an Anthropic client using the best available auth method.

    Priority:
    1. ANTHROPIC_API_KEY -> direct Anthropic API
    2. AWS credentials -> Bedrock
    3. Prompt for API key -> direct Anthropic API
    """
    import anthropic

    if _has_anthropic_api_key():
        print(yellow("  Auth: using Anthropic API key"))
        return anthropic.Anthropic()

    if _has_aws_credentials():
        print(yellow(f"  Auth: using AWS Bedrock (region={region})"))
        return anthropic.AnthropicBedrock(aws_region=region)

    _prompt_api_key()
    print(yellow("  Auth: using Anthropic API key"))
    return anthropic.Anthropic()


def resolve_model(client, model):
    """Return the model ID appropriate for the client type."""
    import anthropic

    if isinstance(client, anthropic.AnthropicBedrock):
        return BEDROCK_MODEL_MAP.get(model, model)
    return model


def correct_ocr(client, model, max_tokens, text):
    """Run a text-only correction pass on raw OCR output.

    Returns corrected text on success, or None on failure.
    """
    try:
        with Spinner("Correcting OCR errors") as spinner, \
             client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                system=CORRECTION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": CORRECTION_USER_PROMPT.format(text=text),
                    }
                ],
             ) as stream:
            token_count = 0
            for chunk in stream.text_stream:
                token_count += 1
                if token_count % 20 == 0:
                    spinner.update(f"~{token_count} tokens")
            message = stream.get_final_message()
        elapsed = spinner.elapsed

        usage = message.usage
        stop = message.stop_reason
        status = "complete" if stop == "end_turn" else "truncated"
        print(yellow(
            f"  {status.capitalize()} in {elapsed:.1f}s"
            f"  ({usage.input_tokens} in / {usage.output_tokens} out)"
        ))
        if stop == "max_tokens":
            print(yellow(
                f"  Warning: correction was truncated at {max_tokens} tokens."
                " Re-run with a higher --max-tokens value."
            ))
        return message.content[0].text
    except Exception as exc:
        print(red(f"  Correction pass failed: {exc}"), file=sys.stderr)
        return None
