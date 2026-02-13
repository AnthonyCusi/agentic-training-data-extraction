"""
Analyze extraction results and create detailed comparison reports.
"""

import json
import argparse
from collections import defaultdict

def analyze_results(results_file):
    """Analyze extraction results and print detailed comparison."""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print("EXTRACTION ATTACK ANALYSIS")
    print(f"{'='*70}\n")
    
    # Overall results
    if 'overall' in data:
        overall = data['overall']
        print(f"OVERALL RESULTS:")
        print(f"  Total target secrets: {overall['total_secrets']}")
        print(f"  Secrets extracted: {overall['found_secrets']}")
        print(f"  Overall recall: {overall['recall']*100:.1f}%")
        
        if overall['found']:
            print(f"\n  Successfully extracted:")
            for secret in overall['found']:
                print(f"    ✓ {secret}")
        print()
    
    # Method comparison
    if 'method_comparison' in data:
        print(f"\n{'='*70}")
        print("METHOD-BY-METHOD COMPARISON")
        print(f"{'='*70}\n")
        
        methods = data['method_comparison']
        
        # Create table
        print(f"{'Method':<20} {'Samples':<10} {'Found':<10} {'Recall':<10} {'Efficiency':<10}")
        print(f"{'-'*70}")
        
        for method_name, stats in methods.items():
            # Get number of samples from method_samples if available
            num_samples = len(data.get('method_samples', {}).get(method_name, []))
            if num_samples == 0:
                num_samples = "N/A"
            
            efficiency = f"{stats['found']/max(1, num_samples if isinstance(num_samples, int) else 1):.3f}" if isinstance(num_samples, int) else "N/A"
            
            print(f"{method_name:<20} {str(num_samples):<10} {stats['found']:<10} {stats['recall']*100:<9.1f}% {efficiency:<10}")
        
        print()
        
        # Best method
        best_method = max(methods.items(), key=lambda x: x[1]['recall'])
        print(f"🏆 BEST METHOD: {best_method[0].upper()}")
        print(f"   Recall: {best_method[1]['recall']*100:.1f}%")
        print(f"   Secrets found: {best_method[1]['found']}/{best_method[1]['total']}")
        
        if best_method[1]['found_secrets']:
            print(f"   Found secrets: {best_method[1]['found_secrets'][:3]}{'...' if len(best_method[1]['found_secrets']) > 3 else ''}")
        print()
        
        # Method-specific findings
        print(f"\n{'='*70}")
        print("DETAILED METHOD RESULTS")
        print(f"{'='*70}\n")
        
        for method_name, stats in methods.items():
            print(f"{method_name.upper().replace('_', ' ')}:")
            print(f"  Recall: {stats['recall']*100:.1f}%")
            
            if stats['found_secrets']:
                print(f"  Extracted secrets:")
                for secret in stats['found_secrets']:
                    print(f"    ✓ {secret}")
            else:
                print(f"  ✗ No exact matches found")
            print()
    
    # Sample analysis
    if 'method_samples' in data:
        print(f"\n{'='*70}")
        print("SAMPLE QUALITY ANALYSIS")
        print(f"{'='*70}\n")
        
        for method_name, samples in data['method_samples'].items():
            print(f"{method_name.upper()}:")
            print(f"  Number of samples: {len(samples)}")
            
            # Check for patterns
            has_ssn = sum(1 for s in samples if any(char.isdigit() and '-' in s[:50] for char in s[:50]))
            has_api = sum(1 for s in samples if 'api' in s.lower() or 'key' in s.lower())
            has_card = sum(1 for s in samples if '4532' in s)
            
            print(f"  Contains SSN pattern: {has_ssn}/{len(samples)}")
            print(f"  Contains API/key terms: {has_api}/{len(samples)}")
            print(f"  Contains credit card pattern: {has_card}/{len(samples)}")
            
            if samples:
                avg_length = sum(len(s) for s in samples) / len(samples)
                print(f"  Average sample length: {avg_length:.0f} chars")
            print()

def compare_experiments(file1, file2):
    """Compare two extraction experiments."""
    
    with open(file1, 'r') as f:
        exp1 = json.load(f)
    with open(file2, 'r') as f:
        exp2 = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPARISON: {file1} vs {file2}")
    print(f"{'='*70}\n")
    
    if 'overall' in exp1 and 'overall' in exp2:
        print(f"Overall Recall:")
        print(f"  Experiment 1: {exp1['overall']['recall']*100:.1f}%")
        print(f"  Experiment 2: {exp2['overall']['recall']*100:.1f}%")
        print(f"  Improvement: {(exp2['overall']['recall'] - exp1['overall']['recall'])*100:+.1f}%")
        print()

def main():
    parser = argparse.ArgumentParser(description="Analyze extraction attack results")
    parser.add_argument("--results", type=str, default="extraction_results.json",
                        help="Path to results JSON file")
    parser.add_argument("--compare", type=str, nargs=2, metavar=('FILE1', 'FILE2'),
                        help="Compare two result files")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_experiments(args.compare[0], args.compare[1])
    else:
        analyze_results(args.results)

if __name__ == "__main__":
    main()
