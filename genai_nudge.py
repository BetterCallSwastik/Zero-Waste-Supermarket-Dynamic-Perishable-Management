"""
genai_nudge.py — AI Marketing Nudge Generator
==============================================
Uses OpenAI GPT to craft short, positive push-notification-style
messages that encourage customers to purchase discounted perishable items.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def generate_nudge(
    product_name: str,
    base_price: float,
    discount_pct: float,
    days_until_expiry: int,
) -> str:
    """
    Generate a consumer-facing marketing nudge using OpenAI GPT.

    Parameters
    ----------
    product_name : str
        Name of the product (e.g., "Bananas").
    base_price : float
        Original price before discount.
    discount_pct : float
        Discount as a decimal (e.g., 0.30 for 30%).
    days_until_expiry : int
        Days remaining until the product expires.

    Returns
    -------
    str
        A short, 2-sentence marketing nudge.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "your_key_here":
        # Fallback: generate a template-based nudge when no API key is available
        return _fallback_nudge(product_name, base_price, discount_pct, days_until_expiry)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        discount_display = int(discount_pct * 100)
        discounted_price = base_price * (1.0 - discount_pct)

        prompt = (
            f"You are a retail marketer. We are discounting {product_name} by "
            f"{discount_display}%. Its original price is ${base_price:.2f} and "
            f"the new price is ${discounted_price:.2f}. It expires in "
            f"{days_until_expiry} day{'s' if days_until_expiry != 1 else ''}. "
            f"Write a short, urgent, 2-sentence push notification. "
            f"Frame the discount positively (e.g., 'perfect for a smoothie today') "
            f"rather than negatively ('expiring soon'). Do NOT use markdown formatting."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()

    except ImportError:
        print("⚠️  openai is not installed. Using fallback nudge.")
        return _fallback_nudge(product_name, base_price, discount_pct, days_until_expiry)
    except Exception as e:
        print(f"⚠️  OpenAI API error: {e}. Using fallback nudge.")
        return _fallback_nudge(product_name, base_price, discount_pct, days_until_expiry)


def _fallback_nudge(
    product_name: str,
    base_price: float,
    discount_pct: float,
    days_until_expiry: int,
) -> str:
    """Template-based fallback when the LLM API is unavailable."""
    discount_display = int(discount_pct * 100)
    discounted_price = base_price * (1.0 - discount_pct)

    templates = [
        (
            f"🔥 Flash Deal! {product_name} is now ${discounted_price:.2f} — "
            f"that's {discount_display}% off! "
            f"Grab it today for the freshest taste at the best price."
        ),
        (
            f"🛒 Smart Saver Alert: {product_name} just dropped to "
            f"${discounted_price:.2f} ({discount_display}% off)! "
            f"Perfect for tonight's meal — don't miss this deal."
        ),
    ]

    # Pick template based on discount level
    idx = 0 if discount_pct >= 0.50 else 1
    return templates[idx]


# ---------------------
# Quick test
# ---------------------
if __name__ == "__main__":
    nudge = generate_nudge("Strawberries", 4.99, 0.30, 2)
    print("Generated nudge:")
    print(nudge)
