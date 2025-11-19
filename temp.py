# 1. Define the complete, hardcoded set of FACTS for inference testing
#    NOTE: This must include ALL 30+ entities. We show the primary ones here.
CUSTOMER_FACTS = """
# IDENTITY & ACCOUNT INFO
Client First Name: Alex
Client Last Name: Johnson
Salutation: Mr. Johnson
Customer Support Phone Number: +1 (800) 555-0199
Customer Support Email: support@omnitech.com
Account Type: Business
Account Category: Gold
Profile Type: Premium
Program: Loyalty Rewards Program
Account Change: Change to Monthly Billing
Upgrade Account: Annual Subscription
Profile: Customer Account Summary
Settings: Notification Settings

# ORDER & FINANCIAL INFO
Order Number: ORD-98765-XYZ
Invoice Number: INV-2025-0312
Refund Amount: $49.99
Money Amount: $129.99

# LOCATION & TIME
Delivery City: Portland, OR
Delivery Country: USA
Shipping Cut-off Time: 4:00 PM EST
Store Location: Miami Flagship Store
Date: 2025-11-18
Date Range: 7-10 days

# INTERACTION & SUPPORT INFO
Online Order Interaction: "Track Order Status"
Online Payment Interaction: "Manage Payment Details"
Online Navigation Step: "Click 'My Account' then 'Subscriptions'"
Online Customer Support Channel: "Knowledge Base"
Online Company Portal Info: Customer Dashboard
Live Chat Support: "Chat with a Specialist"
Website URL: https://www.omnitech-example.com

"""

# 2. Define the complete Llama 3 Inference Template (System Context + User Query)
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




# This entire section REPLACES your existing def predict_and_print(datapoint):
def predict_and_print(test_query, ground_truth):
    # The prompt building is simplified to inject the user's query into the full system template
    full_query_with_context = FULL_INFERENCE_PROMPT.format(user_query=test_query)

    # Note: We are no longer using the input_output_demarkation_key as it's built into the Llama format

    # Add the prompt and expected answer to our lists
    inputs.append(full_query_with_context)
    ground_truth_responses.append(ground_truth)

    payload = {
        "inputs": full_query_with_context,
        "parameters": {"max_new_tokens": 200},
    }

    # --- Prediction Calls Remain The Same ---
    # Pre-trained Model (will likely fail the test)
    pretrained_response = pretrained_predictor.predict(
        payload, custom_attributes="accept_eula=true"
    )
    responses_before_finetuning.append(pretrained_response.get("generated_text"))

    # Fine-Tuned Model (should pass the test)
    finetuned_response = finetuned_predictor.predict(payload)
    responses_after_finetuning.append(finetuned_response.get("generated_text"))




# Create a custom set of tests specifically for the entity logic
custom_tests = [
    # Test 1: Fact Retrieval + Capturing an UNKNOWN Order Number
    ("I need to track my order #111-222-333. How much was my refund?", 
     "Order #111-222-333 has a refund amount of $49.99 associated with it."), 

    # Test 2: Basic Fact Retrieval (Should use the hardcoded Order Number)
    ("What is the shipping cut-off time for my order?", 
     "The shipping cut-off time is 4:00 PM EST for order ORD-98765-XYZ."),

    # Test 3: Identity Retrieval
    ("Can I speak with Alex Johnson?", 
     "Certainly, I see your name is Alex Johnson. How can I help you?"),
]

try:
    for test_query, expected_output in custom_tests:
        # Note: We pass the query and the expected ground truth directly
        predict_and_print(test_query, expected_output) 

    df = pd.DataFrame(
        {
            "Inputs": inputs,
            "Ground Truth": ground_truth_responses,
            "Response from non-finetuned model": responses_before_finetuning,
            "Response from fine-tuned model": responses_after_finetuning,
        }
    )
    display(HTML(df.to_html()))

except Exception as e:
    print(e)
