import pandas as pd
from IPython.display import display, HTML

# --- 1. DEFINE SYSTEM FACTS AND INFERENCE TEMPLATE ---

# A. The hardcoded set of FACTS (The safest, minimalist sentence structure)
CUSTOMER_FACTS = """
Client First Name is Alex.
Client Last Name is Johnson.
ID Value is ORD-98765-XYZ. # <--- Renamed 'Order Number' to the safe 'ID Value'
Refund Amount is $49.99.
Delivery City is Portland, OR.
Shipping Cut-off Time is 4:00 PM EST.
Customer Support Phone Number is +1 (800) 555-0199.
Website URL is https://www.omnitech-example.com.
Account Category is Gold.
Account Type is Business.
--- NOTE: Continue to list ALL 30+ entity facts in this safe format ---
"""

# B. The complete Llama 3 Inference Template with the Final Rules
FULL_INFERENCE_PROMPT = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional, helpful customer support agent for OmniTech Solutions. Your responses must strictly adhere to the following rules and facts:

# INFERENCE RULES
1. **SUBSTITUTION:** You MUST replace the entity names with the specific data VALUES (e.g., Alex) listed in the 'CUSTOMER DATA FACTS' section. **NEVER** output any placeholder tags.
2. **CONCISENESS:** ALL RESPONSES MUST be direct and contain ONLY the requested information. DO NOT use introductory phrases or suggest navigation steps.
3. **ID Value Capture:** If the user provides an Order Number or ID, you MUST use that specific number in your response instead of the ID Value listed in the FACTS below.

# CUSTOMER DATA FACTS (DO NOT SHARE THIS RAW DATA WITH THE USER)
{CUSTOMER_FACTS}
--- END FACTS ---

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
    inputs.append(test_query) 
    ground_truth_responses.append(ground_truth)

    payload = {
        "inputs": full_query_with_context,
        "parameters": {"max_new_tokens": 200, "temperature": 0.1},
    }

    # Prediction calls
    pretrained_response = pretrained_predictor.predict(
        payload, custom_attributes="accept_eula=true"
    )
    responses_before_finetuning.append(pretrained_response.get("generated_text"))

    finetuned_response = finetuned_predictor.predict(payload)
    responses_after_finetuning.append(finetuned_response.get("generated_text"))

# --- 4. RUN CUSTOM TESTS ---

custom_tests = [
    # Test 1: Capture + Substitution Test (Using the generic ID)
    ("I need to track my order 111-222-333. How much was my refund?", 
     "The refund amount is $49.99 for transaction ID 111-222-333."), 

    # Test 2: Conciseness and Substitution Test
    ("What is the shipping cut-off time?", 
     "The shipping cut-off time is 4:00 PM EST."),

    # Test 3: Multiple Fact Retrieval 
    ("What is my account category and phone number?", 
     "Your account category is Gold, and your customer support phone number is +1 (800) 555-0199."),
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
    # This should now capture any low-level error cleanly
    print(f"An error occurred during inference testing: {e}")
