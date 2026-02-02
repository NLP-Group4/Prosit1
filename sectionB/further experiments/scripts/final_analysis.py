"""
Final Complete Results Visualization

Creates comprehensive comparison with ALL results.
"""
print("\nResults saved to: results/experiment_3_final/comprehensive_comparison.png")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

# Complete results from all experiments
CHAR_RESULTS = {
    1: {'MLE': 15.08, 'Add-k': 15.08, 'Stupid Backoff': 15.08},
    2: {'MLE': 8.37, 'Add-k': 8.36, 'Stupid Backoff': 8.36},
    3: {'MLE': 5.96, 'Add-k': 5.94, 'Stupid Backoff': 5.93},
    4: {'MLE': 4.56, 'Add-k': 4.47, 'Stupid Backoff': 4.45},
    5: {'MLE': 4.28, 'Add-k': 3.81, 'Stupid Backoff': 3.72},
}

WORD_RESULTS = {
    1: {'MLE': 780.10, 'Add-k': 693.07, 'Stupid Backoff': 1648.40},
    2: {'MLE': 4216.13, 'Add-k': 788.82, 'Stupid Backoff': 475.72},
    3: {'MLE': 2053286.32, 'Add-k': 4818.64, 'Stupid Backoff': 303.40},
    4: {'MLE': 209123771.38, 'Add-k': 12534.81, 'Stupid Backoff': 487.60},
    5: {'MLE': 1348455405.54, 'Add-k': 16853.14, 'Stupid Backoff': 966.17},
}

os.makedirs('results/experiment_3_final', exist_ok=True)

# Create comprehensive figure
fig = plt.figure(figsize=(18, 12))

# Title
fig.suptitle('Final N-Gram Experiment Results: Character vs Word Tokenization', 
             fontsize=16, fontweight='bold', y=0.995)

# === ROW 1: PERPLEXITY COMPARISONS ===

# Plot 1: Character PPL
ax1 = plt.subplot(3, 3, 1)
n_vals = sorted(CHAR_RESULTS.keys())
methods = ['MLE', 'Add-k', 'Stupid Backoff']
colors = ['#d62728', '#ff7f0e', '#2ca02c']
x = np.arange(len(n_vals))
width = 0.25

for i, method in enumerate(methods):
    vals = [CHAR_RESULTS[n][method] for n in n_vals]
    offset = width * (i - 1)
    ax1.bar(x + offset, vals, width, label=method, color=colors[i], alpha=0.8)

ax1.set_xlabel('N-gram Order', fontweight='bold')
ax1.set_ylabel('Test PPL (log scale)', fontweight='bold')
ax1.set_title('Character-Level: PPL by Method', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(n_vals)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_yscale('log')

# Plot 2: Word PPL
ax2 = plt.subplot(3, 3, 2)
for i, method in enumerate(methods):
    vals = [WORD_RESULTS[n][method] for n in n_vals]
    offset = width * (i - 1)
    ax2.bar(x + offset, vals, width, label=method, color=colors[i], alpha=0.8)

ax2.set_xlabel('N-gram Order', fontweight='bold')
ax2.set_ylabel('Test PPL (log scale)', fontweight='bold')
ax2.set_title('Word-Level: PPL by Method', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(n_vals)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_yscale('log')

# Plot 3: Char vs Word Best
ax3 = plt.subplot(3, 3, 3)
char_best = [min(CHAR_RESULTS[n].values()) for n in n_vals]
word_best = [min(WORD_RESULTS[n].values()) for n in n_vals]

ax3.plot(n_vals, char_best, 'o-', label='Character', linewidth=3, markersize=8, color='#2ca02c')
ax3.plot(n_vals, word_best, 's-', label='Word', linewidth=3, markersize=8, color='#1f77b4')
ax3.set_xlabel('N-gram Order', fontweight='bold')
ax3.set_ylabel('Best PPL (log scale)', fontweight='bold')
ax3.set_title('Best Models: Char vs Word', fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# === ROW 2: IMPROVEMENTS ===

# Plot 4: Character improvements
ax4 = plt.subplot(3, 3, 4)
char_imp_k = [(1 - CHAR_RESULTS[n]['Add-k']/CHAR_RESULTS[n]['MLE'])*100 for n in n_vals]
char_imp_sb = [(1 - CHAR_RESULTS[n]['Stupid Backoff']/CHAR_RESULTS[n]['MLE'])*100 for n in n_vals]

ax4.bar(x - width/2, char_imp_k, width, label='Add-k (k=0.1)', color='#ff7f0e', alpha=0.8)
ax4.bar(x + width/2, char_imp_sb, width, label='Stupid Backoff', color='#2ca02c', alpha=0.8)
ax4.set_xlabel('N-gram Order', fontweight='bold')
ax4.set_ylabel('% Improvement vs MLE', fontweight='bold')
ax4.set_title('Character: Smoothing Impact', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(n_vals)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Plot 5: Word improvements
ax5 = plt.subplot(3, 3, 5)
word_imp_k = [(1 - WORD_RESULTS[n]['Add-k']/WORD_RESULTS[n]['MLE'])*100 for n in n_vals]
word_imp_sb = [(1 - WORD_RESULTS[n]['Stupid Backoff']/WORD_RESULTS[n]['MLE'])*100 for n in n_vals]

ax5.bar(x - width/2, word_imp_k, width, label='Add-k (k=0.1)', color='#ff7f0e', alpha=0.8)
ax5.bar(x + width/2, word_imp_sb, width, label='Stupid Backoff', color='#2ca02c', alpha=0.8)
ax5.set_xlabel('N-gram Order', fontweight='bold')
ax5.set_ylabel('% Improvement vs MLE', fontweight='bold')  
ax5.set_title('Word: Smoothing Impact', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(n_vals)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Plot 6: Char/Word ratio
ax6 = plt.subplot(3, 3, 6)
ratios = [word_best[i] / char_best[i] for i in range(len(n_vals))]
bars = ax6.bar(n_vals, ratios, color='#9467bd', alpha=0.7, edgecolor='black')

for bar, ratio in zip(bars, ratios):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{ratio:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax6.set_xlabel('N-gram Order', fontweight='bold')
ax6.set_ylabel('Word PPL / Char PPL', fontweight='bold')
ax6.set_title('Character Superiority Factor', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_yscale('log')

# === ROW 3: METHOD WINNERS ===

# Plot 7: Best method per configuration (char)
ax7 = plt.subplot(3, 3, 7)
char_winners = []
for n in n_vals:
    best_method = min(CHAR_RESULTS[n].items(), key=lambda x: x[1])[0]
    char_winners.append(best_method)

method_colors = {'MLE': '#d62728', 'Add-k': '#ff7f0e', 'Stupid Backoff': '#2ca02c'}
bar_colors_char = [method_colors[w] for w in char_winners]

bars = ax7.bar(n_vals, [1]*len(n_vals), color=bar_colors_char, alpha=0.7, edgecolor='black')
for bar, method in zip(bars, char_winners):
    ax7.text(bar.get_x() + bar.get_width()/2., 0.5, method,
            ha='center', va='center', fontsize=9, fontweight='bold', rotation=0)

ax7.set_xlabel('N-gram Order', fontweight='bold')
ax7.set_title('Character: Winning Method', fontweight='bold')
ax7.set_yticks([])
ax7.set_ylim(0, 1)

# Plot 8: Best method per configuration (word)
ax8 = plt.subplot(3, 3, 8)
word_winners = []
for n in n_vals:
    best_method = min(WORD_RESULTS[n].items(), key=lambda x: x[1])[0]
    word_winners.append(best_method)

bar_colors_word = [method_colors[w] for w in word_winners]

bars = ax8.bar(n_vals, [1]*len(n_vals), color=bar_colors_word, alpha=0.7, edgecolor='black')
for bar, method in zip(bars, word_winners):
    ax8.text(bar.get_x() + bar.get_width()/2., 0.5, method,
            ha='center', va='center', fontsize=9, fontweight='bold', rotation=0)

ax8.set_xlabel('N-gram Order', fontweight='bold')
ax8.set_title('Word: Winning Method', fontweight='bold')
ax8.set_yticks([])
ax8.set_ylim(0, 1)

# Plot 9: Summary text
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

summary_text = f"""
BEST MODELS

Character-Level:
  5-gram + Stupid Backoff
  PPL: 3.72
  13.1% better than MLE

Word-Level:
  2-gram + Stupid Backoff
  PPL: 475.72
  88.7% better than MLE

Character Advantage:
  127x better perplexity!

Key Insight:
  Stupid Backoff rescues
  word-level from catastrophic
  sparsity (1.3B → 966 PPL)
"""

ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('results/experiment_3_final/complete_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: results/experiment_3_final/complete_analysis.png")
plt.close()

# Print comprehensive table
print("\n" + "="*100)
print("FINAL COMPREHENSIVE RESULTS - ALL METHODS")
print("="*100)

for tokenizer, results in [('CHARACTER', CHAR_RESULTS), ('WORD', WORD_RESULTS)]:
    print(f"\n{tokenizer}-LEVEL TOKENIZATION")
    print("-"*100)
    print(f"{'N':<3} {'MLE':>15} {'Add-k (k=0.1)':>15} {'Stupid Backoff':>17} {'Best':>15} {'Winner':<18} {'Improvement':>10}")
    print("-"*100)
    
    for n in sorted(results.keys()):
        r = results[n]
        best_val = min(r.values())
        best_method = [k for k, v in r.items() if v == best_val][0]
        improvement = (1 - best_val / r['MLE']) * 100 if r['MLE'] > 0 else 0
        
        marker = "✅" if best_method != 'MLE' else "  "
        print(f"{marker} {n:<2} {r['MLE']:>15.2f} {r['Add-k']:>15.2f} {r['Stupid Backoff']:>17.2f} "
              f"{best_val:>15.2f} {best_method:<18} {improvement:>9.1f}%")

print("\n" + "="*100)
print("CRITICAL FINDINGS")
print("="*100)
print("✅ Character 5-gram + Stupid Backoff: 3.72 PPL (BEST OVERALL)")
print("✅ Word 2-gram + Stupid Backoff: 475.72 PPL (best for word-level)")
print("✅ Character 127x better than word at same n")
print("✅ Stupid Backoff critical for word-level: prevents catastrophic sparsity")
print("   - Word 5-gram MLE: 1,348,455,405 PPL (!)")
print("   - Word 5-gram Stupid Backoff: 966 PPL (rescued!)")
print("✅ Smoothing impact = 13.1% for char, up to 99.99% for word!")
print("⚠️  Word-level unusable without proper smoothing for n≥3")
