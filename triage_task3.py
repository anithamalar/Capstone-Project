import json

# 1. Zero-Shot System Prompt
ZERO_SHOT_PROMPT = """You are an AI support ticket classifier. Your task is to CLASSIFY incoming customer support tickets into exactly one of these four allowed categories: billing, delivery, technical, or fraud. 

You must reply with a valid JSON-only response using this exact schema:
{"category": "...", "reason": "..."}
"""

# 2. Few-Shot System Prompt
FEW_SHOT_PROMPT = """You are an AI support ticket classifier. Your task is to CLASSIFY incoming customer support tickets into exactly one of these four allowed categories: billing, delivery, technical, or fraud. 

You must reply with a valid JSON-only response using this exact schema:
{"category": "...", "reason": "..."}

EXAMPLES:
Input: "I was charged twice for my order on my credit card statement."
Output: {"category": "billing", "reason": "The customer reported a duplicate charge on their credit card."}

Input: "My delivery driver dropped my grocery bags and ran away without handing them over."
Output: {"category": "delivery", "reason": "The issue relates to courier handling and delivery service failure."}

Input: "Someone used my saved payment method to place an order in another city without my knowledge!"
Output: {"category": "fraud", "reason": "Unauthorized account usage and fraudulent transaction reported."}
"""

# Verification and Testing
print("--- Task 3 Prompt Verification ---")
print("1. Zero-Shot Prompt Defined Successfully.")
print(f"   - Contains 'CLASSIFY': {'CLASSIFY' in ZERO_SHOT_PROMPT}")
print(f"   - Contains 'EXAMPLES:': {'EXAMPLES:' in ZERO_SHOT_PROMPT}")

print("\n2. Few-Shot Prompt Defined Successfully.")
print(f"   - Contains 'CLASSIFY': {'CLASSIFY' in FEW_SHOT_PROMPT}")
print(f"   - Contains 'EXAMPLES:': {'EXAMPLES:' in FEW_SHOT_PROMPT}")

# Save prompts for use in subsequent pipeline tasks
prompts_dict = {
    "zero_shot": ZERO_SHOT_PROMPT,
    "few_shot": FEW_SHOT_PROMPT
}
import os
os.makedirs("outputs", exist_ok=True)
with open("outputs/prompts.json", "w", encoding="utf-8") as f:
    json.dump(prompts_dict, f, indent=4)

print("\nPrompts saved to outputs/prompts.json successfully!")
