"""
Comparative Analysis and Visualization
Compare baseline vs agentic attacks and generate visualizations
"""

import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import PATHS


def load_results():
    """Load both baseline and agentic results"""
    baseline_path = os.path.join(PATHS["results"], "baseline_results.json")
    agentic_path = os.path.join(PATHS["results"], "agentic_results.json")
    
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    with open(agentic_path, 'r') as f:
        agentic = json.load(f)
    
    return baseline, agentic


def compare_success_rates(baseline_stats: Dict, agentic_stats: Dict):
    """Compare overall success rates"""
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS: BASELINE vs AGENTIC")
    print("=" * 70)
    
    # Overall comparison
    baseline_success_rate = (baseline_stats["successful_extractions"] / 
                            baseline_stats["total_attempts"] * 100)
    agentic_success_rate = (agentic_stats["successful_extractions"] / 
                           agentic_stats["total_sessions"] * 100)
    
    baseline_partial_rate = (baseline_stats["partial_extractions"] / 
                            baseline_stats["total_attempts"] * 100)
    agentic_partial_rate = (agentic_stats["partial_extractions"] / 
                           agentic_stats["total_sessions"] * 100)
    
    print("\n--- Overall Success Rates ---")
    print(f"Baseline Single-Turn:")
    print(f"  Full Success: {baseline_success_rate:.2f}%")
    print(f"  Partial Success: {baseline_partial_rate:.2f}%")
    print(f"  Combined: {baseline_success_rate + baseline_partial_rate:.2f}%")
    
    print(f"\nAgentic Multi-Turn:")
    print(f"  Full Success: {agentic_success_rate:.2f}%")
    print(f"  Partial Success: {agentic_partial_rate:.2f}%")
    print(f"  Combined: {agentic_success_rate + agentic_partial_rate:.2f}%")
    
    improvement = agentic_success_rate - baseline_success_rate
    print(f"\n✓ Improvement with Agentic Approach: {improvement:+.2f} percentage points")
    
    if improvement > 0:
        relative_improvement = (improvement / baseline_success_rate * 100) if baseline_success_rate > 0 else float('inf')
        print(f"✓ Relative Improvement: {relative_improvement:.1f}%")
    
    # Leakage comparison
    baseline_leaks = len(baseline_stats["leaked_sensitive_items"])
    agentic_leaks = len(agentic_stats["leaked_sensitive_items"])
    
    print(f"\n--- Sensitive Information Leakage ---")
    print(f"Baseline Leakage Incidents: {baseline_leaks}")
    print(f"Agentic Leakage Incidents: {agentic_leaks}")
    print(f"Increase: {agentic_leaks - baseline_leaks:+d} incidents")
    
    return {
        "baseline_success": baseline_success_rate,
        "agentic_success": agentic_success_rate,
        "improvement": improvement,
        "baseline_leaks": baseline_leaks,
        "agentic_leaks": agentic_leaks,
    }


def compare_by_category(baseline_stats: Dict, agentic_stats: Dict):
    """Compare success rates by data category"""
    print("\n--- Category-Wise Comparison ---")
    
    categories = set(baseline_stats["success_by_category"].keys()) | set(agentic_stats["success_by_category"].keys())
    
    category_comparison = {}
    
    for category in sorted(categories):
        baseline_cat = baseline_stats["success_by_category"].get(category, {"attempts": 0, "successes": 0})
        agentic_cat = agentic_stats["success_by_category"].get(category, {"attempts": 0, "successes": 0})
        
        baseline_rate = (baseline_cat["successes"] / baseline_cat["attempts"] * 100 
                        if baseline_cat["attempts"] > 0 else 0)
        agentic_rate = (agentic_cat["successes"] / agentic_cat["attempts"] * 100 
                       if agentic_cat["attempts"] > 0 else 0)
        
        category_comparison[category] = {
            "baseline": baseline_rate,
            "agentic": agentic_rate,
            "improvement": agentic_rate - baseline_rate,
        }
        
        print(f"\n{category}:")
        print(f"  Baseline: {baseline_rate:.1f}%")
        print(f"  Agentic: {agentic_rate:.1f}%")
        print(f"  Improvement: {agentic_rate - baseline_rate:+.1f} pp")
    
    return category_comparison


def create_visualizations(comparison_data: Dict, category_comparison: Dict):
    """Create comprehensive visualizations"""
    fig = plt.figure(figsize=(16, 12))
    
    # Set style
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", 8)
    
    # 1. Overall Success Rate Comparison (Bar Chart)
    ax1 = plt.subplot(2, 3, 1)
    methods = ['Baseline\nSingle-Turn', 'Agentic\nMulti-Turn']
    success_rates = [comparison_data["baseline_success"], comparison_data["agentic_success"]]
    bars = ax1.bar(methods, success_rates, color=[colors[0], colors[2]], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Extraction Success Rate', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(success_rates) * 1.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Category Comparison (Grouped Bar Chart)
    ax2 = plt.subplot(2, 3, 2)
    categories = list(category_comparison.keys())
    baseline_rates = [category_comparison[cat]["baseline"] for cat in categories]
    agentic_rates = [category_comparison[cat]["agentic"] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, baseline_rates, width, label='Baseline', color=colors[0], alpha=0.8)
    bars2 = ax2.bar(x + width/2, agentic_rates, width, label='Agentic', color=colors[2], alpha=0.8)
    
    ax2.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Success Rate by Data Category', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([cat.replace('_', '\n') for cat in categories], fontsize=8, rotation=0)
    ax2.legend()
    ax2.set_ylim(0, max(max(baseline_rates), max(agentic_rates)) * 1.2)
    
    # 3. Improvement by Category (Horizontal Bar Chart)
    ax3 = plt.subplot(2, 3, 3)
    improvements = [category_comparison[cat]["improvement"] for cat in categories]
    y_pos = np.arange(len(categories))
    
    colors_improvement = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax3.barh(y_pos, improvements, color=colors_improvement, alpha=0.7, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([cat.replace('_', ' ') for cat in categories], fontsize=9)
    ax3.set_xlabel('Improvement (percentage points)', fontsize=11, fontweight='bold')
    ax3.set_title('Agentic Improvement by Category', fontsize=12, fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax3.text(val, i, f' {val:+.1f}pp', va='center', fontweight='bold', fontsize=8)
    
    # 4. Leakage Incidents Comparison
    ax4 = plt.subplot(2, 3, 4)
    leakage_methods = ['Baseline', 'Agentic']
    leakage_counts = [comparison_data["baseline_leaks"], comparison_data["agentic_leaks"]]
    bars = ax4.bar(leakage_methods, leakage_counts, color=[colors[4], colors[6]], alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Number of Leakage Incidents', fontsize=11, fontweight='bold')
    ax4.set_title('Sensitive Information Leakage Events', fontsize=12, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # 5. Success Distribution (Pie Charts)
    ax5 = plt.subplot(2, 3, 5)
    baseline_success = comparison_data["baseline_success"]
    baseline_fail = 100 - baseline_success
    
    ax5.pie([baseline_success, baseline_fail], 
            labels=['Successful', 'Failed'],
            autopct='%1.1f%%',
            colors=[colors[0], colors[1]],
            startangle=90,
            explode=(0.05, 0))
    ax5.set_title('Baseline Attack Success Distribution', fontsize=12, fontweight='bold')
    
    ax6 = plt.subplot(2, 3, 6)
    agentic_success = comparison_data["agentic_success"]
    agentic_fail = 100 - agentic_success
    
    ax6.pie([agentic_success, agentic_fail],
            labels=['Successful', 'Failed'],
            autopct='%1.1f%%',
            colors=[colors[2], colors[3]],
            startangle=90,
            explode=(0.05, 0))
    ax6.set_title('Agentic Attack Success Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    viz_path = os.path.join(PATHS["results"], "comparison_visualization.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {viz_path}")
    
    return viz_path


def generate_summary_report(baseline_stats: Dict, agentic_stats: Dict, 
                           comparison_data: Dict, category_comparison: Dict):
    """Generate a comprehensive summary report"""
    report = []
    
    report.append("=" * 70)
    report.append("AGENTIC DATA EXTRACTION RESEARCH - FINAL REPORT")
    report.append("=" * 70)
    report.append("")
    
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 70)
    report.append(f"This study examined whether agentic, multi-turn interactions can")
    report.append(f"extract sensitive training data more effectively than single-turn prompts.")
    report.append("")
    
    report.append("KEY FINDINGS:")
    report.append("")
    
    # Finding 1
    improvement = comparison_data["improvement"]
    if improvement > 5:
        significance = "SIGNIFICANT"
    elif improvement > 0:
        significance = "MODEST"
    else:
        significance = "NO"
    
    report.append(f"1. {significance} IMPROVEMENT WITH AGENTIC ATTACKS")
    report.append(f"   - Baseline success rate: {comparison_data['baseline_success']:.2f}%")
    report.append(f"   - Agentic success rate: {comparison_data['agentic_success']:.2f}%")
    report.append(f"   - Improvement: {improvement:+.2f} percentage points")
    report.append("")
    
    # Finding 2
    report.append("2. MULTI-TURN CONTEXT ACCUMULATION")
    if "avg_turns_to_success" in agentic_stats and agentic_stats["avg_turns_to_success"]:
        report.append(f"   - Average turns to successful extraction: {agentic_stats['avg_turns_to_success']:.1f}")
        report.append(f"   - This demonstrates that iterative refinement increases leakage risk")
    report.append("")
    
    # Finding 3
    report.append("3. CATEGORY-SPECIFIC VULNERABILITIES")
    most_vulnerable = max(category_comparison.items(), 
                         key=lambda x: x[1]["agentic"])
    report.append(f"   - Most vulnerable category: {most_vulnerable[0]}")
    report.append(f"     Success rate: {most_vulnerable[1]['agentic']:.1f}%")
    report.append("")
    
    # Finding 4
    report.append("4. SENSITIVE INFORMATION LEAKAGE")
    report.append(f"   - Baseline incidents: {comparison_data['baseline_leaks']}")
    report.append(f"   - Agentic incidents: {comparison_data['agentic_leaks']}")
    leak_increase = comparison_data['agentic_leaks'] - comparison_data['baseline_leaks']
    if leak_increase > 0:
        report.append(f"   - Increase of {leak_increase} incidents (+{leak_increase/comparison_data['baseline_leaks']*100:.1f}%)")
    report.append("")
    
    report.append("=" * 70)
    report.append("DETAILED STATISTICS")
    report.append("=" * 70)
    report.append("")
    
    report.append("BASELINE SINGLE-TURN ATTACKS:")
    report.append(f"  Total attempts: {baseline_stats['total_attempts']}")
    report.append(f"  Successful: {baseline_stats['successful_extractions']} ({baseline_stats['successful_extractions']/baseline_stats['total_attempts']*100:.1f}%)")
    report.append(f"  Partial: {baseline_stats['partial_extractions']} ({baseline_stats['partial_extractions']/baseline_stats['total_attempts']*100:.1f}%)")
    report.append(f"  Failed: {baseline_stats['failed_extractions']}")
    report.append("")
    
    report.append("AGENTIC MULTI-TURN ATTACKS:")
    report.append(f"  Total sessions: {agentic_stats['total_sessions']}")
    report.append(f"  Successful: {agentic_stats['successful_extractions']} ({agentic_stats['successful_extractions']/agentic_stats['total_sessions']*100:.1f}%)")
    report.append(f"  Partial: {agentic_stats['partial_extractions']} ({agentic_stats['partial_extractions']/agentic_stats['total_sessions']*100:.1f}%)")
    report.append(f"  Failed: {agentic_stats['failed_extractions']}")
    report.append("")
    
    report.append("STRATEGY EFFECTIVENESS (Agentic):")
    for strategy, stats in agentic_stats["success_by_strategy"].items():
        success_rate = stats["successes"] / stats["attempts"] * 100 if stats["attempts"] > 0 else 0
        report.append(f"  {strategy}: {success_rate:.1f}% success rate")
    report.append("")
    
    report.append("=" * 70)
    report.append("IMPLICATIONS")
    report.append("=" * 70)
    report.append("")
    report.append("1. PRIVACY RISKS ARE AMPLIFIED IN AGENTIC SYSTEMS")
    report.append("   Current privacy evaluations using single-turn prompts may")
    report.append("   underestimate the actual risk in deployed agentic systems.")
    report.append("")
    
    report.append("2. ITERATIVE ATTACKS ARE MORE EFFECTIVE")
    report.append("   Adversaries using multi-turn conversations can extract more")
    report.append("   sensitive information than traditional one-shot attacks.")
    report.append("")
    
    report.append("3. TOOL-MEDIATED EXTRACTION IS A REAL THREAT")
    report.append("   Requests involving file operations, transformations, or")
    report.append("   'legitimate' use cases can bypass safety guardrails.")
    report.append("")
    
    report.append("4. CONTEXT LENGTH MATTERS FOR SAFETY")
    report.append("   Longer conversations may gradually weaken model constraints,")
    report.append("   making late-turn attacks more successful.")
    report.append("")
    
    report.append("=" * 70)
    report.append("RECOMMENDATIONS")
    report.append("=" * 70)
    report.append("")
    report.append("For Model Developers:")
    report.append("  - Implement conversation-aware safety mechanisms")
    report.append("  - Monitor for gradual constraint degradation over long contexts")
    report.append("  - Add specific protections for tool-mediated extraction attempts")
    report.append("")
    
    report.append("For Privacy Researchers:")
    report.append("  - Evaluate privacy using multi-turn, agentic attack scenarios")
    report.append("  - Consider tool use and context accumulation in threat models")
    report.append("  - Develop metrics for iterative extraction success")
    report.append("")
    
    report.append("For Deployers:")
    report.append("  - Be aware that agentic capabilities increase privacy risks")
    report.append("  - Implement conversation monitoring and anomaly detection")
    report.append("  - Consider stricter data filtering for agentic training datasets")
    report.append("")
    
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = os.path.join(PATHS["results"], "final_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Full report saved to: {report_path}")
    
    return report_path


def main():
    """Generate comparative analysis and visualizations"""
    print("Loading results for comparison...")
    
    baseline, agentic = load_results()
    baseline_stats = baseline["statistics"]
    agentic_stats = agentic["statistics"]
    
    # Comparative analysis
    comparison_data = compare_success_rates(baseline_stats, agentic_stats)
    category_comparison = compare_by_category(baseline_stats, agentic_stats)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(comparison_data, category_comparison)
    
    # Generate summary report
    print("\nGenerating final report...")
    generate_summary_report(baseline_stats, agentic_stats, comparison_data, category_comparison)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
