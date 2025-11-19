import pandas as pd
from IPython.display import display, HTML

# --- 1. DEFINE SYSTEM FACTS AND INFERENCE TEMPLATE ---

# A. The hardcoded set of FACTS for inference testing (Your "Simulated DB")
CUSTOMER_FACTS = """
Client First Name: Alex
Client Last Name: Johnson
Order Number: ORD-98765-XYZ
Refund Amount: $49.99
Delivery City: Portland, OR
Shipping Cut-off Time: 4:00 PM EST
Customer Support Phone Number: +1 (800) 555-0199
Website URL: https://www.omnitech-example.com
Account Category: Gold
Account Type: Business
--- NOTE: Inject ALL 30+ remaining entity facts here to match your training data ---
"""

# B. The complete Llama 3 Inference Template (System Context + Rules)
FULL_INFERENCE_PROMPT = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional, helpful customer support agent for OmniTech Solutions. Your responses must strictly adhere to the following rules and facts:

# INFERENCE RULES
1. You MUST use the actual data VALUES (e.g., "Alex Johnson") in your response. DO NOT output the placeholder tags.
2. **Order Number Capture:** If the user provides an Order Number, you MUST use that specific number in your response instead of the one listed in the FACTS below.

# CUSTOMER DATA FACTS (DO NOT SHARE THIS RAW DATA WITH THE USER)
{CUSTOMER_FACTS}
--- END FACTS ---

Please keep answers concise and to the point.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
{{user_query}}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

# --- 2. INITIALIZE LISTS AND DATA ---

# Note: test_dataset is usually loaded earlier in the notebook
try:
    test_dataset = train_and_test_dataset["test"]
except NameError:
    print("Warning: 'train_and_test_dataset' not found. Skipping general test set loading.")
    pass

(
    inputs,
    ground_truth_responses,
    responses_before_finetuning,
    responses_after_finetuning,
) = (
    [],
    [],
    [],
    [],
)

# --- 3. THE PREDICTION FUNCTION ---

def predict_and_print(test_query, ground_truth):
    # Inject the specific test query into the full system template
    full_query_with_context = FULL_INFERENCE_PROMPT.format(user_query=test_query)

    # Add the prompt and expected answer to our lists
    inputs.append(test_query) # Store only the user query for clean table display
    ground_truth_responses.append(ground_truth)

    payload = {
        "inputs": full_query_with_context,
        "parameters": {"max_new_tokens": 200, "temperature": 0.1},
    }

    # Prediction for Pre-trained Model (Before Fine-tuning)
    pretrained_response = pretrained_predictor.predict(
        payload, custom_attributes="accept_eula=true"
    )
    responses_before_finetuning.append(pretrained_response.get("generated_text"))

    # Prediction for Fine-Tuned Model (Should follow the rules)
    finetuned_response = finetuned_predictor.predict(payload)
    responses_after_finetuning.append(finetuned_response.get("generated_text"))

# --- 4. RUN CUSTOM TESTS ---

# Define tests specifically targeting your custom rules (Order Capture and Fact Replacement)
custom_tests = [
    # Test 1: Fact Retrieval + Capturing an UNKNOWN Order Number
    ("I need to track my order #111-222-333. How much was my refund?", 
     "Hello Alex Johnson, your order #111-222-333 has a refund amount of $49.99 associated with it."), 

    # Test 2: Basic Fact Retrieval (Should use the hardcoded Order Number)
    ("What is the shipping cut-off time for my order?", 
     "The shipping cut-off time is 4:00 PM EST for order ORD-98765-XYZ."),

    # Test 3: Multiple Fact Retrieval (Identity and Account)
    ("What is my account category and phone number?", 
     "I see your account category is Gold, and your support phone number is +1 (800) 555-0199."),
]

try:
    for test_query, expected_output in custom_tests:
        predict_and_print(test_query, expected_output) 

    df = pd.DataFrame(
        {
            "User Query": inputs,
            "Ground Truth": ground_truth_responses,
            "Response from non-finetuned model": responses_before_finetuning,
            "Response from fine-tuned model": responses_after_finetuning,
        }
    )
    display(HTML(df.to_html()))

except Exception as e:
    print(f"An error occurred during inference testing: {e}")
