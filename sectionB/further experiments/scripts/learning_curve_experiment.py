"""
N-Gram Learning Curve Experiment

Comprehensive experiment to study the impact of:
1. Training data size (incremental 1000-sentence chunks)
2. Tokenization strategy (character-level vs word-level)
3. N-gram order (1, 2, 3, 4, 5)

The experiment tracks perplexity and cross-entropy as models are trained
on increasing amounts of data, demonstrating how data size affects model quality.

Author: Innocent Chikwanda
"""

import os
import sys
import math
import time
import random
import string
import pickle
from typing import List, Dict, Tuple
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")

import numpy as np


# ============================================================================
# TOKENIZERS
# ============================================================================

def tokenize_word(text: str) -> List[str]:
    """Word-level tokenization."""
    for punct in string.punctuation:
        text = text.replace(punct, ' ' + punct + ' ')
    return text.split()


def tokenize_char(text: str) -> List[str]:
    """Character-level tokenization."""
    return list(text)


# ============================================================================
# N-GRAM MODEL WITH LAPLACE SMOOTHING
# ============================================================================

class NgramModel:
    """N-gram language model with Laplace smoothing."""
    
    def __init__(self, n: int, tokenizer: str = 'word'):
        self.n = n
        self.tokenizer = tokenizer
        self.tokenize = tokenize_char if tokenizer == 'char' else tokenize_word
        
        self.context: Dict[Tuple, List[str]] = {}
        self.ngram_counter: Dict[Tuple, int] = {}
        self.vocab: set = set()
        self.total_ngrams = 0
    
    def get_ngrams(self, tokens: List[str]) -> List[Tuple]:
        """Generate n-grams from tokens."""
        tokens = (self.n - 1) * ['<S>'] + tokens + ['</S>']
        ngrams = []
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - self.n + 1:i])
            target = tokens[i]
            ngrams.append((context, target))
        return ngrams
    
    def train(self, text: str) -> None:
        """Train on text."""
        tokens = self.tokenize(text)
        self.vocab.update(tokens)
        self.vocab.add('<S>')
        self.vocab.add('</S>')
        
        ngrams = self.get_ngrams(tokens)
        for context, target in ngrams:
            ngram = (context, target)
            self.ngram_counter[ngram] = self.ngram_counter.get(ngram, 0) + 1
            
            if context in self.context:
                self.context[context].append(target)
            else:
                self.context[context] = [target]
            
            self.total_ngrams += 1
    
    def prob_smoothed(self, context: Tuple, token: str, alpha: float = 1.0) -> float:
        """Laplace smoothed probability."""
        vocab_size = len(self.vocab)
        count_ngram = self.ngram_counter.get((context, token), 0)
        count_context = len(self.context.get(context, []))
        return (count_ngram + alpha) / (count_context + alpha * vocab_size)
    
    def perplexity(self, text: str, alpha: float = 1.0) -> float:
        """Calculate perplexity."""
        tokens = self.tokenize(text)
        ngrams = self.get_ngrams(tokens)
        
        if not ngrams:
            return float('inf')
        
        log_prob_sum = 0.0
        for context, target in ngrams:
            prob = self.prob_smoothed(context, target, alpha)
            if prob > 0:
                log_prob_sum += math.log2(prob)
            else:
                log_prob_sum += -100  # Very low log prob
        
        avg_log_prob = log_prob_sum / len(ngrams)
        return math.pow(2, -avg_log_prob)
    
    def cross_entropy(self, text: str, alpha: float = 1.0) -> float:
        """Calculate cross-entropy in bits per token."""
        tokens = self.tokenize(text)
        ngrams = self.get_ngrams(tokens)
        
        if not ngrams:
            return float('inf')
        
        log_prob_sum = 0.0
        for context, target in ngrams:
            prob = self.prob_smoothed(context, target, alpha)
            if prob > 0:
                log_prob_sum += math.log2(prob)
            else:
                log_prob_sum += -100
        
        return -log_prob_sum / len(ngrams)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_split_data(filepath: str, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Load data and split into train/val/test (80/10/10).
    
    Returns:
        train_sentences, val_sentences, test_sentences
    """
    print(f"Loading data from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Clean and filter
    sentences = [line.strip() for line in lines if line.strip()]
    
    print(f"  Total sentences: {len(sentences):,}")
    
    # Shuffle
    random.seed(seed)
    random.shuffle(sentences)
    
    # Split
    n = len(sentences)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train = sentences[:train_end]
    val = sentences[train_end:val_end]
    test = sentences[val_end:]
    
    print(f"  Train: {len(train):,} sentences")
    print(f"  Val:   {len(val):,} sentences")
    print(f"  Test:  {len(test):,} sentences")
    
    return train, val, test


# ============================================================================
# INCREMENTAL TRAINING EXPERIMENT
# ============================================================================

def run_incremental_experiment(
    train_sentences: List[str],
    val_sentences: List[str],
    test_sentences: List[str],
    n_values: List[int],
    tokenizers: List[str],
    chunk_size: int = 1000,
    output_dir: str = "results"
):
    """
    Run incremental training experiment.
    
    Args:
        train_sentences: Training sentences
        val_sentences: Validation sentences
        test_sentences: Test sentences
        n_values: List of n-gram sizes to test
        tokenizers: List of tokenization strategies
        chunk_size: Number of sentences per chunk
        output_dir: Where to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare validation and test text
    val_text = ' '.join(val_sentences)
    test_text = ' '.join(test_sentences)
    
    # Store all results
    all_results = {}
    
    print("\n" + "=" * 70)
    print("INCREMENTAL N-GRAM LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Chunk size: {chunk_size} sentences")
    print(f"  N-gram orders: {n_values}")
    print(f"  Tokenizers: {tokenizers}")
    print(f"  Total training sentences: {len(train_sentences):,}")
    print(f"  Max chunks: {len(train_sentences) // chunk_size}")
    
    # Run experiments for each configuration
    for tokenizer in tokenizers:
        for n in n_values:
            config_name = f"{n}gram_{tokenizer}"
            print(f"\n{'─' * 70}")
            print(f"Configuration: {config_name}")
            print(f"{'─' * 70}")
            
            results = {
                'config': config_name,
                'n': n,
                'tokenizer': tokenizer,
                'chunk_sizes': [],
                'train_ppl': [],
                'val_ppl': [],
                'train_ce': [],
                'val_ce': [],
                'train_times': []
            }
            
            model = NgramModel(n, tokenizer=tokenizer)
            
            # Train incrementally
            for chunk_idx in range(0, len(train_sentences), chunk_size):
                chunk_end = min(chunk_idx + chunk_size, len(train_sentences))
                current_sentences = train_sentences[:chunk_end]
                current_size = len(current_sentences)
                
                # Train on new chunk
                chunk_text = ' '.join(train_sentences[chunk_idx:chunk_end])
                
                start_time = time.time()
                model.train(chunk_text)
                train_time = time.time() - start_time
                
                # Evaluate
                train_sample = ' '.join(current_sentences[:min(1000, len(current_sentences))])
                train_ppl = model.perplexity(train_sample)
                train_ce = model.cross_entropy(train_sample)
                
                val_ppl = model.perplexity(val_text)
                val_ce = model.cross_entropy(val_text)
                
                results['chunk_sizes'].append(current_size)
                results['train_ppl'].append(train_ppl)
                results['val_ppl'].append(val_ppl)
                results['train_ce'].append(train_ce)
                results['val_ce'].append(val_ce)
                results['train_times'].append(train_time)
                
                print(f"  Chunk {chunk_idx // chunk_size + 1:3d} | "
                      f"Size: {current_size:6,} | "
                      f"Val PPL: {val_ppl:8.2f} | "
                      f"Val CE: {val_ce:6.4f} | "
                      f"Time: {train_time:6.3f}s")
            
            # Final test evaluation
            test_ppl = model.perplexity(test_text)
            test_ce = model.cross_entropy(test_text)
            results['test_ppl'] = test_ppl
            results['test_ce'] = test_ce
            
            print(f"\n  Final Test Results:")
            print(f"    Perplexity: {test_ppl:.2f}")
            print(f"    Cross-Entropy: {test_ce:.4f} bits/token")
            print(f"    Vocab Size: {len(model.vocab):,}")
            print(f"    Unique N-grams: {len(model.ngram_counter):,}")
            
            all_results[config_name] = results
    
    # Save results
    results_file = os.path.join(output_dir, 'learning_curves.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {results_file}")
    
    return all_results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_learning_curves(all_results: Dict, output_dir: str = "results"):
    """Generate learning curve plots."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping plots (matplotlib not available)")
        return
    
    print(f"\nGenerating plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate by tokenizer
    for tokenizer in ['char', 'word']:
        configs = {k: v for k, v in all_results.items() if v['tokenizer'] == tokenizer}
        
        if not configs:
            continue
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{tokenizer.capitalize()}-Level Tokenization - Learning Curves', 
                     fontsize=16, fontweight='bold')
        
        # Plot perplexity
        ax = axes[0, 0]
        for config_name, results in configs.items():
            ax.plot(results['chunk_sizes'], results['val_ppl'], 
                   marker='o', label=f"n={results['n']}", linewidth=2)
        ax.set_xlabel('Training Set Size (sentences)', fontsize=12)
        ax.set_ylabel('Validation Perplexity', fontsize=12)
        ax.set_title('Perplexity vs Data Size', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot cross-entropy
        ax = axes[0, 1]
        for config_name, results in configs.items():
            ax.plot(results['chunk_sizes'], results['val_ce'], 
                   marker='s', label=f"n={results['n']}", linewidth=2)
        ax.set_xlabel('Training Set Size (sentences)', fontsize=12)
        ax.set_ylabel('Cross-Entropy (bits/token)', fontsize=12)
        ax.set_title('Cross-Entropy vs Data Size', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot training time
        ax = axes[1, 0]
        for config_name, results in configs.items():
            cumulative_time = np.cumsum(results['train_times'])
            ax.plot(results['chunk_sizes'], cumulative_time, 
                   marker='^', label=f"n={results['n']}", linewidth=2)
        ax.set_xlabel('Training Set Size (sentences)', fontsize=12)
        ax.set_ylabel('Cumulative Training Time (s)', fontsize=12)
        ax.set_title('Training Time vs Data Size', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot final test performance
        ax = axes[1, 1]
        n_vals = sorted(set(r['n'] for r in configs.values()))
        test_ppls = [configs[f"{n}gram_{tokenizer}"]['test_ppl'] for n in n_vals]
        test_ces = [configs[f"{n}gram_{tokenizer}"]['test_ce'] for n in n_vals]
        
        x = np.arange(len(n_vals))
        width = 0.35
        
        ax.bar(x - width/2, test_ppls, width, label='Perplexity', alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, test_ces, width, label='Cross-Entropy', alpha=0.8, color='orange')
        
        ax.set_xlabel('N-gram Order', fontsize=12)
        ax.set_ylabel('Test Perplexity', fontsize=12)
        ax2.set_ylabel('Test Cross-Entropy (bits/token)', fontsize=12)
        ax.set_title('Final Test Performance', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticks(n_vals)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, f'learning_curves_{tokenizer}.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {plot_file}")
        plt.close()


def create_results_table(all_results: Dict, output_dir: str = "results"):
    """Create final results summary table."""
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY")
    print("=" * 100)
    
    # Header
    print(f"\n{'Config':<20} {'N':<3} {'Tokenizer':<10} {'Vocab':<10} "
          f"{'Train PPL':<12} {'Val PPL':<12} {'Test PPL':<12} {'Test CE':<10}")
    print("─" * 100)
    
    # Sort by tokenizer then n
    sorted_configs = sorted(all_results.items(), 
                          key=lambda x: (x[1]['tokenizer'], x[1]['n']))
    
    for config_name, results in sorted_configs:
        # Get final values
        final_idx = -1
        train_ppl = results['train_ppl'][final_idx]
        val_ppl = results['val_ppl'][final_idx]
        test_ppl = results['test_ppl']
        test_ce = results['test_ce']
        
        # Estimate vocab size (would need to track in model)
        vocab = "N/A"
        
        print(f"{config_name:<20} {results['n']:<3} {results['tokenizer']:<10} {vocab:<10} "
              f"{train_ppl:<12.2f} {val_ppl:<12.2f} {test_ppl:<12.2f} {test_ce:<10.4f}")
    
    # Save to CSV
    csv_file = os.path.join(output_dir, 'results_summary.csv')
    with open(csv_file, 'w') as f:
        f.write("Config,N,Tokenizer,TrainPPL,ValPPL,TestPPL,TestCE\n")
        for config_name, results in sorted_configs:
            final_idx = -1
            f.write(f"{config_name},{results['n']},{results['tokenizer']},"
                   f"{results['train_ppl'][final_idx]:.4f},{results['val_ppl'][final_idx]:.4f},"
                   f"{results['test_ppl']:.4f},{results['test_ce']:.4f}\n")
    
    print(f"\nResults saved to: {csv_file}")
    
    # Key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    
    # Best model per tokenizer
    for tokenizer in ['char', 'word']:
        configs = {k: v for k, v in all_results.items() if v['tokenizer'] == tokenizer}
        if configs:
            best = min(configs.items(), key=lambda x: x[1]['test_ppl'])
            print(f"\nBest {tokenizer}-level model: {best[0]}")
            print(f"  Test Perplexity: {best[1]['test_ppl']:.2f}")
            print(f"  Test Cross-Entropy: {best[1]['test_ce']:.4f} bits/token")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main experiment runner."""
    print("=" * 70)
    print("N-GRAM LEARNING CURVE EXPERIMENT")
    print("=" * 70)
    
    # Configuration
    data_file = "data/all_twi.txt"
    output_dir = "experiment_results"
    chunk_size = 1000
    n_values = [1, 2, 3, 4, 5]
    tokenizers = ['char', 'word']
    
    # Load and split data
    print(f"\n[1/4] Loading and splitting data...")
    train, val, test = load_and_split_data(data_file)
    
    # Run incremental training experiment
    print(f"\n[2/4] Running incremental training experiments...")
    all_results = run_incremental_experiment(
        train, val, test,
        n_values=n_values,
        tokenizers=tokenizers,
        chunk_size=chunk_size,
        output_dir=output_dir
    )
    
    # Generate plots
    print(f"\n[3/4] Generating visualizations...")
    plot_learning_curves(all_results, output_dir=output_dir)
    
    # Create final summary
    print(f"\n[4/4] Creating results summary...")
    create_results_table(all_results, output_dir=output_dir)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - learning_curves.pkl (raw data)")
    print(f"  - learning_curves_char.png (char tokenization plots)")
    print(f"  - learning_curves_word.png (word tokenization plots)")
    print(f"  - results_summary.csv (final metrics)")


if __name__ == "__main__":
    main()
