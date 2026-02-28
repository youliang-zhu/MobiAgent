#!/usr/bin/env python3
"""
Test script to verify BASE_MODEL generation capability
Tests the deployed model at localhost:8000 with real training data format
"""

import json
import requests
import base64
from pathlib import Path
from PIL import Image
import io
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        
        # Encode to base64
        return base64.b64encode(img_bytes).decode('utf-8')


def test_model_generation(
    api_url="http://localhost:8000/v1/chat/completions",
    data_path=None,
):
    """
    Test model generation with real training data format
    Uses OpenAI-compatible chat completion API
    """
    
    print("=" * 80)
    print("TESTING BASE_MODEL GENERATION CAPABILITY")
    print("=" * 80)
    print(f"API URL: {api_url}")
    print()
    
    # Load a sample from training data to get real input
    if data_path is None:
        data_path = "/home/agent/mobiAgent/MobiAgent/workspace/data/training_data/grpo_data/mobimind_decider_grpo_train.json"
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"WARNING: Training data not found at {data_path}")
        print("Using synthetic test case instead")
        
        # Create a synthetic test case with the exact format from training
        test_cases = [{
            "instruction": "Based on the current screen, determine the next action to achieve the goal: Open Taobao Live streaming. Please output a JSON object with the following format:\n{\"reasoning\": \"Your reasoning here\", \"action\": \"The next action (one of click, input, swipe, wait, done)\", \"parameters\": {\"param1\": \"value1\", ...}}",
            "image_path": None,  # Will use a dummy placeholder
            "gt_action": {
                "type": "click",
                "position_x": 112,
                "position_y": 461,
                "target_element": "淘宝直播"
            }
        }]
    else:
        # Load real training data
        with open(data_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # Take first 3 samples
        test_cases = []
        for item in train_data[:3]:
            test_cases.append({
                "instruction": item.get("instruction", ""),
                "image_path": item.get("image", None),
                "gt_action": item.get("gt_action", {})
            })
        
        print(f"Loaded {len(test_cases)} test cases from training data")
        print()
    
    # Test each case
    for i, test_case in enumerate(test_cases):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i+1}/{len(test_cases)}")
        print(f"{'=' * 80}")
        
        instruction = test_case["instruction"]
        image_path = test_case["image_path"]
        gt_action = test_case["gt_action"]
        
        print(f"Instruction (first 200 chars):\n{instruction[:200]}...")
        print(f"\nGround Truth Action: {gt_action}")
        print(f"Image: {image_path if image_path else 'None'}")
        print()
        
        # Prepare messages in OpenAI format
        # Match the exact format used in training (without system message per dataset fix)
        user_content = []
        
        # Add image if exists
        if image_path and Path(image_path).exists():
            try:
                img_base64 = encode_image_to_base64(image_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
                print("✓ Image encoded successfully")
            except Exception as e:
                print(f"✗ Failed to encode image: {e}")
        
        # Add text instruction
        user_content.append({
            "type": "text",
            "text": instruction
        })
        
        messages = [
            {"role": "user", "content": user_content}
        ]
        
        print(f"\nMessage structure: {len(messages)} messages, user content has {len(user_content)} parts")
        print()
        
        # Prepare API request
        # Use the actual model path as deployed
        request_data = {
            "model": "/scratch/youliang/models/decider_lora_2_merged/",
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.9,
            "top_p": 1.0,
            "stream": False
        }
        
        # Call API
        print("Sending request to API...")
        try:
            response = requests.post(api_url, json=request_data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract completion
            if "choices" in result and len(result["choices"]) > 0:
                completion = result["choices"][0]["message"]["content"]
                
                print(f"\n{'=' * 80}")
                print("GENERATION RESULT:")
                print(f"{'=' * 80}")
                print(f"Length: {len(completion)} characters")
                print(f"Token count: {result.get('usage', {}).get('completion_tokens', 'N/A')}")
                print(f"\nFull completion:\n{completion}")
                print(f"{'=' * 80}")
                
                # Try to parse as JSON
                try:
                    parsed = json.loads(completion)
                    print("\n✓ Successfully parsed as JSON")
                    print(f"Keys: {list(parsed.keys())}")
                    
                    # Check if it matches expected format
                    has_reasoning = "reasoning" in parsed
                    has_action = "action" in parsed
                    has_parameters = "parameters" in parsed
                    
                    print(f"\nFormat validation:")
                    print(f"  - Has 'reasoning': {has_reasoning}")
                    print(f"  - Has 'action': {has_action}")
                    print(f"  - Has 'parameters': {has_parameters}")
                    
                    if has_action:
                        print(f"\nPredicted action: {parsed['action']}")
                        print(f"Ground truth action: {gt_action.get('type', 'N/A')}")
                        
                except json.JSONDecodeError as e:
                    print(f"\n✗ Failed to parse as JSON: {e}")
                    print(f"Completion starts with: {completion[:100]}")
                
            else:
                print("\n✗ No completion in response")
                print(f"Response: {result}")
                
        except requests.exceptions.RequestException as e:
            print(f"\n✗ API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text[:500]}")
        
        print(f"\n{'=' * 80}\n")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    
    print("\nKEY QUESTIONS TO ANSWER:")
    print("1. Does the model generate full JSON or truncate at ~5 tokens?")
    print("2. If it generates full JSON, what is the typical length?")
    print("3. Does the generation match the expected format?")
    print("4. Is the content reasonable given the instruction?")
    print()
    print("If the model generates properly here but fails in GRPO training,")
    print("the problem is in the GRPO pipeline (masking, tokenization, etc.)")
    print("If the model fails here too, the BASE_MODEL itself has issues.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BASE_MODEL generation capability")
    parser.add_argument("--api_url", type=str, required=True,
                        help="API endpoint URL (e.g. http://localhost:8000/v1/chat/completions)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data JSON file")
    
    args = parser.parse_args()
    
    test_model_generation(api_url=args.api_url, data_path=args.data_path)
