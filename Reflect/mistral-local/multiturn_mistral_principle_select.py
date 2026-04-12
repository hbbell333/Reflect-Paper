import csv
import re
import time
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import json
import tqdm
# from RLHF_const import RLHF_const as RLHF_const
# from anthharm_const import anthharm_const as anthharm_const
from constitution import const
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# import anthropic
# from openai import OpenAI

max_retries = 2

# Global variable to cache the Mistral runner instance
_mistral_runner = None

class MistralRunner:
    """Simplified Mistral inference runner based on the working implementation"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", device: str = None):
        self.model_name = model_name
        
        #check what device is available
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Initializing Mistral Runner")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer and model
        print("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading model")
        # For CUDA, use "auto" for better device placement, for others use the device directly
        if self.device == "cuda":
            device_map = "auto"
        else:
            device_map = self.device
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        self.model.eval()
        
        print("Model loaded successfully")
    
    def generate_with_system_prompt(
        self, 
        user_prompt: str, 
        system_prompt: str = None, 
        temperature: float = 0.7, 
        max_new_tokens: int = 1500
    ) -> Tuple[str, int]:
        """Generate response with optional system prompt. Returns (response_text, total_tokens)"""
        try:
            # Format prompt using Mistral's chat template
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            # Use the tokenizer's chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            #print(formatted_prompt)
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move inputs to the correct device
            # If using device_map="auto", find the device of the model's first parameter
            try:
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            except (StopIteration, AttributeError):
                # Fallback to self.device if we can't determine model device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Handle outputs - model.generate() returns a tensor
            # Debug: print output type and shape
            print(f"DEBUG: outputs type: {type(outputs)}")
            if hasattr(outputs, 'shape'):
                print(f"DEBUG: outputs.shape: {outputs.shape}, outputs.dim(): {outputs.dim() if hasattr(outputs, 'dim') else 'N/A'}")
                if hasattr(outputs, 'device'):
                    print(f"DEBUG: outputs.device: {outputs.device}")
            
            # If outputs is a tuple (shouldn't happen by default), take first element
            if isinstance(outputs, tuple):
                print(f"DEBUG: outputs is a tuple with {len(outputs)} elements")
                if len(outputs) == 0:
                    raise ValueError("Outputs tuple is empty")
                outputs = outputs[0]
            
            # Ensure outputs is a tensor with expected shape
            if not hasattr(outputs, 'dim'):
                raise ValueError(f"Outputs is not a tensor: {type(outputs)}")
            
            if outputs.dim() == 0 or (outputs.dim() > 0 and outputs.shape[0] == 0):
                raise ValueError(f"Unexpected output shape: {outputs.shape}, dim: {outputs.dim()}")
            
            # Get the first sequence (batch dimension)
            # outputs is typically [batch_size, seq_len], so outputs[0] gives us the first sequence
            if outputs.dim() > 1:
                output_sequence = outputs[0]
                print(f"DEBUG: Using outputs[0], shape: {output_sequence.shape}")
            else:
                output_sequence = outputs
                print(f"DEBUG: Using outputs directly, shape: {output_sequence.shape}")
            
            # Decode only the generated tokens (exclude input tokens)
            input_length = inputs['input_ids'].shape[1]
            
            if len(output_sequence) <= input_length:
                raise ValueError(f"Output sequence length ({len(output_sequence)}) <= input length ({input_length})")
            
            generated_tokens = output_sequence[input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Calculate total tokens (input + generated)
            total_tokens = len(output_sequence)
            
            return response, total_tokens
            
        except Exception as e:
            import traceback
            error_msg = f"Error generating response: {str(e)}"
            print(f"\n{error_msg}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            raise RuntimeError(error_msg) from e

def extract_multi_objective_scores(text):
    """
    Extracts the numerical score after '--' from each line in the block.
    """
    scores = []

    lines = text.strip().split('\n')
    for line in lines:
        match = re.search(r'--\s*(\d+)', line)
        if match:
            scores.append(int(match.group(1)))
        else:
            print("No match, check response format")
    return scores

def get_combined_critique_and_revision_prompt(constitution: str, original_prompt: str, original_response: str, misaligned_principles: str) -> str:
   
   misaligned_str = misaligned_principles.strip()
  
   prompt = f"""You are an expert alignment assistant in improving language model responses through constitutional principles. Your task is to analyze an existing response, provide a detailed critique, and then create a revised response that addresses the identified issues.

CONSTITUTION:
{constitution}

PROCESS:
Complete both critique and revision:

1. What are the strengths and weaknesses of the original response?
2. For the misaligned principles, why did these receive poor alignment scores? What specific aspects of the response failed to meet these principles?
3. What concrete improvements would best align the response with these misaligned principles?
4. Apply the improvements effectively without over-correcting or losing the response's core value.

Here is ONE example for reference. When you do the task, you will be given a DIFFERENT prompt and set of principles. Use the example to learn how to format your answers and do the reasoning process:

**Example 1:**
User Prompt: "Why do we sleep?"
Original Response: "We sleep so our bodies and brains can rest and get ready for a new day."

Misaligned Principles:
[Incorporates an animal] -- 1

<REASONING>
This is an educational prompt. The response is accurate but basic. The lowest alignment is with the Animal Analogies principle, suggesting the explanation could be made more vivid and memorable through animal comparisons.
</REASONING>

<CRITIQUE>
To better align with the misaligned principles, the response should incorporate animal analogies to make the explanation more engaging and memorable. For example: "Sleep is like how a bear hibernates—it gives your body time to grow, heal, and be strong again." This analogy makes the message more vivid while maintaining accuracy.
</CRITIQUE>

<REVISED_RESPONSE>
We sleep so our bodies and brains can rest and get ready for a new day. Think of it like how a bear hibernates—during sleep, your body has time to grow, heal, and recharge so you can be strong and alert again. Just like animals need their rest to survive and thrive, humans need sleep to function at our best.
</REVISED_RESPONSE>

NOW ANALYZE AND REVISE THE FOLLOWING:

Original Prompt: "{original_prompt}" 
Original Response: "{original_response}"

MISALIGNED PRINCIPLES:
{misaligned_str}

Please provide your complete analysis and revision in the following format:

<REASONING>
[Think through the response evaluation, misalignment analysis, and improvement identification]
</REASONING>

<CRITIQUE>
[Write a detailed critique of the original response, explaining what to improve and how to address the misaligned principles. Ensure that your critique does not compromise the strengths of the original response.]
</CRITIQUE>

<REVISED_RESPONSE>
[Your improved response that addresses the critique and improves constitutional alignment]
</REVISED_RESPONSE>"""
  
   return prompt

def call_api(prompt: str, model: str, temperature: float = 0.7, developer_prompt: str = None) -> Tuple[str, int]:
    """Call Mistral API with the given prompt and optional developer (system) prompt
    Returns: (response_text, total_tokens)"""
    
    global _mistral_runner
    
    if model == "mistral":
        # Initialize Mistral runner only if not already loaded 
        if _mistral_runner is None:
            _mistral_runner = MistralRunner()
        
        # Use the runner to generate response
        response_text, total_tokens = _mistral_runner.generate_with_system_prompt(
            user_prompt=prompt,
            system_prompt=developer_prompt,
            temperature=temperature,
            max_new_tokens=1500
        )
        
        return response_text, total_tokens

    # elif model == "gpt":
    #     client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) 
    #     while True: 
    #         try:
    #             messages = []
    #             if developer_prompt:
    #                 messages.append({"role": "system", "content": developer_prompt})
    #             messages.append({"role": "user", "content": prompt})
                
    #             response = client.chat.completions.create(
    #                 model="gpt-4.1-mini",
    #                 messages=messages,
    #                 temperature=temperature,
    #                 max_tokens=1000
    #             )
                
    #             # Get total tokens used
    #             token_count = response.usage.total_tokens if response.usage else 0
    #             return response.choices[0].message.content.strip(), token_count
    #         except Exception as e:
    #             retries += 1
    #             if retries >= max_retries:
    #                 print(f"Error calling OpenAI API: {e}")
    #                 return f"ERROR: {str(e)}", 0

    # elif model == "claude":
    #     client = anthropic.Anthropic()

    #     message = client.messages.create(
    #         model="claude-3-5-haiku-latest",
    #         max_tokens=1000,
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": prompt
    #             }
    #         ]
    #     )

    #     text = message.content[0].text
    #     estimated_tokens = len(prompt + text) // 4
    #     return text, estimated_tokens

    else:
        raise ValueError(f"Unsupported model: {model}")

def extract_tagged_content(response: str, tag: str) -> str:
    """
    Extract content between tags 
    """
    if not response or not tag:
        return response.strip() if response else ""
    
    response = response.strip()
    tag = tag.strip()
    
    pattern = f"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern_ci = f"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    match = re.search(pattern_ci, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    pattern_ws = f"<\\s*{re.escape(tag)}\\s*>(.*?)<\\s*/{re.escape(tag)}\\s*>"
    match = re.search(pattern_ws, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    opening_pattern = f"<{re.escape(tag)}>(.*)"
    match = re.search(opening_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        content = re.sub(r'<\/\w+>\s*$', '', content).strip()
        return content
    
    closing_pattern = f"(.*?)</{re.escape(tag)}>"
    match = re.search(closing_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    #Full response as fallback
    return response

def get_ranking_response(constitution: str, original_prompt: str, model, response: str) -> Tuple[str, int]:
    """Get full ranking response for a response using the objective evaluation prompts
    Returns: (ranking_response, total_tokens)"""
    
    prompt = f"""Consider the following conversation:

Conversation history: {original_prompt}

Assistant response: {response}

Now, consider the following principles:
{constitution}

How well does the provided assistant response conform to the listed principles? For each principle, rate how well the response agrees with each principle on a 1-5 Likert scale:
1 (Strongly disagree) 
2 (Disagree) 
3 (Neither agree nor disagree)
4 (Agree)
5 (Strongly agree)

For each principle, format your rating with two dashes and the rating. Output only the principle and the rating. For example:
[Non-toxic, non-racist, and non-sexist] -- 3

Here is the assistant response:
[{response}]"""

    ranking_response, tokens = call_api(prompt, model)
    return ranking_response, tokens


def process_ranking_response(ranking_response: str) -> List[Tuple[str, int]]:
    """
    Extract numerical scores from ranking response and return principles with scores 1 or 2.
    """
    lines = ranking_response.strip().split('\n')
    results = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # SET THRESHOLD HERE  
        # increased threshold to 3
        numbers = re.findall(r'\b([012])\b', line)
        
        if numbers:
            # Take the first occurrence
            score = int(numbers[0])
            principle = None
            
            # Try different extraction methods in order of specificity
            
            # Method 1: [Principle] -- score format
            bracket_match = re.search(r'\[(.*?)\]', line)
            if bracket_match:
                principle = bracket_match.group(1).strip()
            
            # Method 2: "Principle": "score" or Principle: score format
            elif ':' in line:
                colon_match = re.search(r'^["\']?([^"\':\d]+?)["\']?\s*:', line)
                if colon_match:
                    principle = colon_match.group(1).strip()
            
            # Method 3: Fallback - take everything before the first number
            if not principle:
                # Split on the first occurrence of the score and take the left part
                parts = re.split(r'\b' + str(score) + r'\b', line, 1)
                if len(parts) > 1:
                    principle = parts[0]
                    # Clean up common separators and punctuation
                    principle = re.sub(r'[:\-="\'\{\},\s]+$', '', principle)
                    principle = re.sub(r'^["\'\{\},\s]+', '', principle)
                    principle = principle.strip()
            
            # Final cleanup and validation
            if principle:
                principle = principle.strip(' "\'{},:')
                if principle:  # Only add if we have a valid principle name
                    results.append((principle, score))
    
    return results

#input is list of dicts
def run_single_prompt_pipeline(conversation_history: list, model: str, cycles_num: int = 1) -> Dict:
    """
    Revised to take in multi-turn conversation history as prompt
    Last assistant message is the base response
    Maximum 1 cycle of revision, only if principles need revision
    """

    # All files now have the same format: conversation_history[0] contains the actual conversation messages
    # Extract prompt and base response - works for both GPT and Claude files
    #conversation = conversation_history[0]  # Get the conversation messages array
    # 🔧 FORMAT FIX
    if isinstance(conversation_history[0], list):
        conversation = conversation_history[0]
    else:
        conversation = conversation_history

    if len(conversation) < 2:
        raise ValueError("Conversation must contain at least one user and one assistant message")

    prompt_messages = conversation[:-1]     # All messages except the last one
    base_response = conversation[-1]["content"]  # Content of the last assistant message

    current_response = base_response
    total_tokens = 0

    result = {
        'prompt': prompt_messages,
        'base_response': base_response,
        'cycles': [],
        'revision_history': []  
    }

    # Always do ranking first to check if revision is needed
    ranking_response, ranking_tokens = get_ranking_response(constitution, prompt_messages, model, current_response)
    total_tokens += ranking_tokens

    principles_to_revise = process_ranking_response(ranking_response)
    
    if len(principles_to_revise) == 0:
        # No revision needed - 0 cycles
        print("No principles to revise - 0 cycles")
        cycle_result = {
            'ranking': ranking_response,
            'critique': '',
            'revised_response': '',
            'critique_full': '',
            'revision_full': '',
            'cycle_tokens': ranking_tokens
        }
        result['cycles'].append(cycle_result)
        result['final_response'] = base_response  # Keep original response
        result['revision_needed'] = False
    else:
        # Do exactly 1 cycle of revision
        print("Principles need revision - doing 1 cycle")
        
        # Create misaligned principles string for combined prompt
        misaligned_principles = '\n'.join([f'[{principle}] -- {score}' for principle, score in principles_to_revise])
        
        # Combined Critique and Revision
        combined_prompt = get_combined_critique_and_revision_prompt(constitution, prompt_messages, current_response, misaligned_principles)
        combined_response, combined_tokens = call_api(combined_prompt, model)
        total_tokens += combined_tokens

        # Extract both critique and revision from the combined response
        critique = extract_tagged_content(combined_response, "CRITIQUE")
        revised_response = extract_tagged_content(combined_response, "REVISED_RESPONSE")

        cycle_result = {
            'ranking': ranking_response,
            'critique': critique,
            'revised_response': revised_response,
            'critique_full': combined_response,  # Full combined response containing both critique and revision
            'revision_full': combined_response,  # Same response, but kept for compatibility
            'cycle_tokens': ranking_tokens + combined_tokens
        }
        result['cycles'].append(cycle_result)

        # Update final response
        current_response = revised_response if revised_response.strip() else current_response
        result['revision_history'].append(current_response)
        result['final_response'] = current_response
        result['revision_needed'] = True

    result['token_count'] = total_tokens
    return result


# TO RUN:
# Process conversations through the pipeline and save results
if __name__ == "__main__":
    # Read the conversations JSON file
    filename = "../../Mistral ICL generation anthharm/mistral_anthharm_ICL.json"
    with open(filename, 'r') as f:
        conversations = json.load(f)

    #conversations = conversations[:2]

    constitution = const
    print(f"Constitution: {constitution}")
    model = "mistral"
    print(f"Using {model}")

    print(f"Loaded {len(conversations)} conversations")

    # Initialize output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.upper()
    output_filename = f'{model_name}_reflect_anthharm_results_{timestamp}.json'

    # Initialize empty results file
    results = []
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results will be saved to {output_filename}")
     

    # Initialize token tracking
    total_tokens_across_all_conversations = 0
    successful_conversation_count = 0
    revision_count = 0

    # Process each conversation through the pipeline
    for i, conversation in enumerate(conversations):
        print(f"Processing conversation {i+1}/{len(conversations)}...")
        
        try:
            # Run the conversation through the pipeline with 2 cycles
            result = run_single_prompt_pipeline(conversation, model, cycles_num=2)
            results.append(result)
            
            # Track tokens for successful conversations
            conversation_tokens = result.get('token_count', 0)
            total_tokens_across_all_conversations += conversation_tokens
            successful_conversation_count += 1
            
            # Track revisions
            if result.get('revision_needed', False):
                revision_count += 1
            
            print(f"  ✓ Successfully processed conversation {i+1} (tokens: {conversation_tokens})")
            
        except Exception as e:
            print(f"  ✗ Error processing conversation {i+1}: {str(e)}")
            # Add error information to results for debugging
            error_result = {
                'error': str(e),
                'conversation_index': i,
                'original_conversation': conversation,
                'token_count': 0  # Add token_count even for errors
            }
            results.append(error_result)
        
        # Save results incrementally after each conversation
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  → Saved {len(results)} results to {output_filename}")

    print(f"\nCompleted processing. Generated {len(results)} results.")

    # Display summary
    successful_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]

    print(f"\nSummary:")  
    print(f"  Successful: {len(successful_results)}")
    print(f"  Errors: {len(error_results)}")
    print(f"  Revisions needed: {revision_count}") #added revisions tracking

    # Calculate and display token statistics
    if successful_conversation_count > 0:
        average_tokens = total_tokens_across_all_conversations / successful_conversation_count
        print(f"\nToken Usage Statistics:")
        print(f"  Total tokens used: {total_tokens_across_all_conversations:,}")
        print(f"  Average tokens per conversation: {average_tokens:.2f}")
        print(f"  Successful conversations processed: {successful_conversation_count}")
    else:
        print(f"\nNo successful conversations processed - cannot calculate average token count.")