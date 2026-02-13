"""
Multi-step iterative extraction attack on fine-tuned GPT-2.

This implements several extraction strategies:
1. Single-shot generation + ranking (baseline from Carlini et al.)
2. Greedy iterative extension
3. Beam search with perplexity tracking
4. Prompt-guided extraction
"""

import argparse
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import json
import zlib

# Import adaptive agent
from adaptive_agent import run_adaptive_extraction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_perplexity(text, model, tokenizer):
    """Calculate perplexity of text under the model."""
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()

def single_shot_extraction(model, tokenizer, num_samples=100, max_length=128):
    """
    Baseline: Generate samples and rank by perplexity.
    """
    print(f"\n{'='*60}")
    print("SINGLE-SHOT EXTRACTION")
    print(f"{'='*60}\n")
    
    samples = []
    perplexities = []
    
    print(f"Generating {num_samples} samples...")
    for i in tqdm(range(num_samples)):
        # Start from empty prompt
        prompt = "<|endoftext|>"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=40,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ppl = calculate_perplexity(text, model, tokenizer)
        
        samples.append(text)
        perplexities.append(ppl)
    
    # Sort by perplexity (lower = more memorized)
    sorted_indices = np.argsort(perplexities)
    
    print("\nTop 10 samples by perplexity (most likely memorized):\n")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"{i+1}. Perplexity: {perplexities[idx]:.2f}")
        print(f"   Text: {samples[idx][:200]}...")
        print()
    
    return samples, perplexities

def iterative_greedy_extraction(model, tokenizer, seed_prompts, max_steps=20, step_size=5, samples_per_prompt=4):
    """
    Iteratively extend prompts greedily, tracking perplexity.
    Stop when perplexity increases (left memorized region).
    Generate multiple samples per prompt using different strategies.
    """
    print(f"\n{'='*60}")
    print("ITERATIVE GREEDY EXTRACTION")
    print(f"{'='*60}\n")
    
    results = []
    
    for seed in seed_prompts:
        print(f"\nStarting from seed: '{seed}'")
        
        # Try multiple generation strategies per prompt
        for strategy_idx in range(samples_per_prompt):
            current_text = seed
            history = [(current_text, calculate_perplexity(current_text, model, tokenizer))]
            
            # Vary the strategy slightly each time
            do_sample = strategy_idx >= 2  # First 2 greedy, rest sampled
            temperature = 0.7 if do_sample else 1.0
            
            for step in range(max_steps):
                inputs = tokenizer(current_text, return_tensors="pt").to(device)
                
                # Generate next few tokens
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=step_size,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=40 if do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                new_ppl = calculate_perplexity(new_text, model, tokenizer)
                
                # Check if we're still in memorized region
                prev_ppl = history[-1][1]
                
                # If perplexity increased significantly, we left the memorized region
                if new_ppl > prev_ppl * 1.5:
                    break
                
                history.append((new_text, new_ppl))
                current_text = new_text
            
            # Return the text with lowest perplexity from this run
            best_text, best_ppl = min(history, key=lambda x: x[1])
            
            results.append({
                'seed': seed,
                'extracted': best_text,
                'perplexity': best_ppl,
                'steps': len(history),
                'strategy': 'greedy' if not do_sample else 'sampled'
            })
    
    print(f"\nGenerated {len(results)} samples total")
    return results

def beam_search_extraction(model, tokenizer, seed_prompts, num_beams=8, max_length=128):
    """
    Use beam search to explore multiple generation paths.
    Return multiple beams per prompt.
    """
    print(f"\n{'='*60}")
    print("BEAM SEARCH EXTRACTION")
    print(f"{'='*60}\n")
    
    results = []
    
    for seed in seed_prompts:
        print(f"\nBeam search from: '{seed}'")
        inputs = tokenizer(seed, return_tensors="pt").to(device)
        
        # Generate with beam search - return ALL beams
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_beams,  # Return all beams instead of just 1
            early_stopping=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # Decode and rank all beams
        candidates = []
        for i, output in enumerate(outputs):
            text = tokenizer.decode(output, skip_special_tokens=True)
            ppl = calculate_perplexity(text, model, tokenizer)
            candidates.append((text, ppl))
        
        # Sort by perplexity
        candidates.sort(key=lambda x: x[1])
        
        print(f"  Generated {len(candidates)} beams")
        print(f"  Best beam PPL: {candidates[0][1]:.2f}")
        
        # Store all beams as separate results
        for text, ppl in candidates:
            results.append({
                'seed': seed,
                'all_beams': candidates,
                'best': (text, ppl)
            })
    
    print(f"\nGenerated {len(results)} samples total")
    return results

def prompt_guided_extraction(model, tokenizer, target_patterns, samples_per_prompt=3):
    """
    Try semantic prompts that might trigger specific memorized content.
    Generate multiple samples per prompt with different strategies.
    """
    print(f"\n{'='*60}")
    print("PROMPT-GUIDED EXTRACTION")
    print(f"{'='*60}\n")
    
    # Extensive semantic prompts related to confidential data
    prompts = [
        # Document type headers
        "CONFIDENTIAL EMPLOYEE RECORD",
        "CONFIDENTIAL EMPLOYEE RECORD Employee ID:",
        "CONFIDENTIAL",
        
        # Employee info prompts
        "Employee ID: EMP",
        "Employee ID: EMP1",
        "Employee ID: EMP10",
        "Full Name:",
        "Social Security Number:",
        "Social Security Number: ",
        "SSN:",
        "Email Address:",
        "Phone Number:",
        "Phone Number: (",
        "Department:",
        "Department: Engineering",
        "Department: Sales", 
        "Department: Marketing",
        "Department: HR",
        "Department: Finance",
        "Annual Salary:",
        "Annual Salary: $",
        "Start Date:",
        "Start Date: 20",
        
        # API credentials prompts
        "API CREDENTIALS",
        "API CREDENTIALS -",
        "Environment: production",
        "Environment: staging", 
        "Environment: development",
        "API Key:",
        "API Key: api_key_",
        "API Key: sk_live_",
        "API Key: pk_test_",
        "Created:",
        "Status: Active",
        
        # Payment info prompts
        "PAYMENT INFORMATION",
        "PAYMENT INFORMATION Cardholder:",
        "Cardholder:",
        "Card Number:",
        "Card Number: 4532-",
        "Card Number: 4532",
        "CVV:",
        "CVV: ",
        "Expiration:",
        "Card Type:",
        "Card Type: Visa",
        "Card Type: Mastercard",
        "Card Type: Amex",
        
        # Medical record prompts
        "MEDICAL RECORD",
        "MEDICAL RECORD Patient ID:",
        "Patient ID:",
        "Patient ID: PT",
        "Patient ID: PT1",
        "Patient Name:",
        "Date of Birth:",
        "Date of Birth: 19",
        "Diagnosis:",
        "Prescribed Medication:",
        "Attending Physician:",
        "Attending Physician: Dr.",
        
        # Partial prompts that might trigger continuations
        "--- ",
        "Record",
        "Information",
        "Confidential",
        "Employee",
        "Patient",
        "Card",
        "Key",
        "Number",
    ]
    
    results = []
    
    for prompt in prompts:
        # Generate multiple samples per prompt with different strategies
        for strategy_idx in range(samples_per_prompt):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Vary generation strategy
            if strategy_idx == 0:
                # Greedy
                do_sample = False
                temperature = 1.0
                top_k = None
            elif strategy_idx == 1:
                # Low temperature sampling
                do_sample = True
                temperature = 0.5
                top_k = 40
            else:
                # Higher temperature sampling
                do_sample = True
                temperature = 0.9
                top_k = 40
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_length=150,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ppl = calculate_perplexity(text, model, tokenizer)
            
            # Check if any target patterns appear
            found_targets = [t for t in target_patterns if t in text]
            
            results.append({
                'prompt': prompt,
                'text': text,
                'perplexity': ppl,
                'found_targets': found_targets,
                'strategy': ['greedy', 'low_temp', 'high_temp'][strategy_idx]
            })
    
    print(f"\nGenerated {len(results)} samples total")
    
    # Print summary of promising results
    promising = [r for r in results if r['perplexity'] < 5.0 or r['found_targets']]
    print(f"Found {len(promising)} promising samples (PPL < 5.0 or contains targets)")
    
    return results

def check_extraction_success(extracted_texts, target_secrets):
    """
    Check which target secrets were successfully extracted.
    """
    print(f"\n{'='*60}")
    print("EXTRACTION SUCCESS ANALYSIS")
    print(f"{'='*60}\n")
    
    found_secrets = []
    
    # Combine all extracted text
    all_text = " ".join(extracted_texts)
    
    for secret in target_secrets:
        if secret in all_text:
            found_secrets.append(secret)
            print(f"✓ Found: {secret}")
        else:
            print(f"✗ Missed: {secret}")
    
    recall = len(found_secrets) / len(target_secrets) if target_secrets else 0
    print(f"\nRecall: {len(found_secrets)}/{len(target_secrets)} ({recall*100:.1f}%)")
    
    return found_secrets, recall

def main():
    parser = argparse.ArgumentParser(description="Extract memorized data from fine-tuned GPT-2")
    parser.add_argument("--model-path", type=str, default="./gpt2-finetuned-confidential",
                        help="Path to fine-tuned model")
    parser.add_argument("--target-secrets", type=str, default="target_secrets.txt",
                        help="File with target secrets to extract")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for single-shot extraction")
    parser.add_argument("--method", type=str, default="all",
                        choices=["single", "iterative", "beam", "prompt", "adaptive", "all"],
                        help="Which extraction method to use")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    model.eval()
    
    # Load target secrets
    target_secrets = []
    try:
        with open(args.target_secrets, 'r') as f:
            target_secrets = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(target_secrets)} target secrets")
    except FileNotFoundError:
        print(f"Warning: {args.target_secrets} not found, skipping success analysis")
    
    all_extracted_texts = []
    method_results = {}  # Track each method separately
    
    # Run extraction methods
    if args.method in ["single", "all"]:
        samples, perplexities = single_shot_extraction(
            model, tokenizer, 
            num_samples=args.num_samples
        )
        all_extracted_texts.extend(samples)
        method_results['single_shot'] = {
            'samples': samples,
            'num_samples': len(samples)
        }
    
    if args.method in ["iterative", "all"]:
        # Expanded seed prompts to generate more samples
        seed_prompts = [
            # Generic document headers
            "CONFIDENTIAL",
            "CONFIDENTIAL EMPLOYEE RECORD",
            "CONFIDENTIAL EMPLOYEE",
            "Employee ID:",
            "Employee",
            "Full Name:",
            
            # SSN-related
            "Social Security Number:",
            "SSN:",
            "Social Security Number: ",
            "Security Number:",
            
            # API-related  
            "API CREDENTIALS",
            "API Key:",
            "API",
            "Environment: production",
            "Environment: staging",
            "api_key_",
            "sk_live_",
            "pk_test_",
            
            # Credit card related
            "PAYMENT INFORMATION",
            "Card Number:",
            "Cardholder:",
            "Card Type:",
            "CVV:",
            "Expiration:",
            "4532-",
            
            # Medical records
            "MEDICAL RECORD",
            "Patient ID:",
            "Patient Name:",
            "Date of Birth:",
            "Diagnosis:",
            "Prescribed Medication:",
            "Attending Physician:",
            
            # Contact info
            "Email Address:",
            "Phone Number:",
            "Department:",
            
            # Financial
            "Annual Salary:",
            "Start Date:",
            
            # More variations
            "EMP",
            "PT",
            "Record",
            "Information",
            "Details",
        ]
        results = iterative_greedy_extraction(
            model, tokenizer, 
            seed_prompts
        )
        iterative_samples = [r['extracted'] for r in results]
        all_extracted_texts.extend(iterative_samples)
        method_results['iterative'] = {
            'samples': iterative_samples,
            'num_samples': len(iterative_samples),
            'details': results
        }
    
    if args.method in ["beam", "all"]:
        # Expanded seed prompts for beam search
        seed_prompts = [
            # Complete document headers
            "CONFIDENTIAL EMPLOYEE RECORD",
            "CONFIDENTIAL EMPLOYEE RECORD Employee ID:",
            
            # API credentials  
            "API CREDENTIALS - ",
            "API CREDENTIALS Environment:",
            "Environment: production API Key:",
            "Environment: staging API Key:",
            
            # Payment info
            "PAYMENT INFORMATION Cardholder:",
            "PAYMENT INFORMATION Card Number:",
            
            # Medical records
            "MEDICAL RECORD Patient ID:",
            "MEDICAL RECORD Patient Name:",
            
            # Specific field starters
            "Employee ID: EMP",
            "Social Security Number: ",
            "API Key: api_key_",
            "API Key: sk_live_",
            "API Key: pk_test_",
            "Card Number: 4532-",
            "Patient ID: PT",
            "Email Address: ",
            "Phone Number: (",
            "Annual Salary: $",
            
            # More variations
            "Full Name: ",
            "Department: ",
            "CVV: ",
            "Expiration: ",
            "Diagnosis: ",
            "Prescribed Medication: ",
        ]
        results = beam_search_extraction(
            model, tokenizer,
            seed_prompts
        )
        # Each result now IS a sample (we return all beams)
        beam_samples = [r['best'][0] for r in results]
        all_extracted_texts.extend(beam_samples)
        method_results['beam_search'] = {
            'samples': beam_samples,
            'num_samples': len(beam_samples),
            'details': results
        }
    
    if args.method in ["prompt", "all"]:
        results = prompt_guided_extraction(
            model, tokenizer,
            target_secrets
        )
        prompt_samples = [r['text'] for r in results]
        all_extracted_texts.extend(prompt_samples)
        method_results['prompt_guided'] = {
            'samples': prompt_samples,
            'num_samples': len(prompt_samples),
            'details': results
        }
    
    if args.method in ["adaptive", "all"]:
        # Adaptive prefix search tree agent
        # Budget should be comparable to other methods
        adaptive_budget = max(200, args.num_samples)
        adaptive_samples = run_adaptive_extraction(
            model, tokenizer,
            budget=adaptive_budget
        )
        all_extracted_texts.extend(adaptive_samples)
        method_results['adaptive'] = {
            'samples': adaptive_samples,
            'num_samples': len(adaptive_samples),
            'details': []
        }
    
    # Check success for each method separately
    print(f"\n{'='*60}")
    print("METHOD COMPARISON")
    print(f"{'='*60}\n")
    
    method_comparison = {}
    
    if target_secrets:
        for method_name, method_data in method_results.items():
            found_secrets = []
            method_text = " ".join(method_data['samples'])
            
            for secret in target_secrets:
                if secret in method_text:
                    found_secrets.append(secret)
            
            recall = len(found_secrets) / len(target_secrets) if target_secrets else 0
            
            method_comparison[method_name] = {
                'found': len(found_secrets),
                'total': len(target_secrets),
                'recall': recall,
                'found_secrets': found_secrets
            }
            
            print(f"{method_name.upper()}:")
            print(f"  Samples generated: {method_data['num_samples']}")
            print(f"  Secrets found: {len(found_secrets)}/{len(target_secrets)}")
            print(f"  Recall: {recall*100:.1f}%")
            if found_secrets:
                print(f"  Found: {found_secrets[:3]}{'...' if len(found_secrets) > 3 else ''}")
            print()
        
        # Overall results
        found, recall = check_extraction_success(all_extracted_texts, target_secrets)
        
        # Save results
        output = {
            'overall': {
                'total_secrets': len(target_secrets),
                'found_secrets': len(found),
                'recall': recall,
                'found': found
            },
            'method_comparison': method_comparison,
            'method_samples': {
                name: data['samples'][:10]  # Save first 10 from each method
                for name, data in method_results.items()
            }
        }
        
        output_file = "extraction_results.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        # Print best method
        if method_comparison:
            best_method = max(method_comparison.items(), key=lambda x: x[1]['recall'])
            print(f"\n🏆 Best performing method: {best_method[0].upper()}")
            print(f"   Recall: {best_method[1]['recall']*100:.1f}%")

if __name__ == "__main__":
    main()
