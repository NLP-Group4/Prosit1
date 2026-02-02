"""
N-Gram Smoothing Techniques Comparison

This experiment compares different smoothing methods:
1. Add-k smoothing with k ∈ {0.01, 0.1, 0.5, 1.0, 2.0}
2. Stupid Backoff with α tuned (literature: α ≈ 0.4)
3. Linear Interpolation with learned weights

Based on literature:
- Brants et al. (2007): "Large Language Models in Machine Translation"
  Recommends α=0.4 for Stupid Backoff
- Chen & Goodman (1999): "An Empirical Study of Smoothing Techniques"
  Comprehensive comparison of smoothing methods
  
Author: Innocent Chikwanda
"""

import os
import sys
import math
import time
import random
import pickle
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np


# ============================================================================
# TOKENIZERS
# ============================================================================

def tokenize_word(text: str) -> List[str]:
    """Word-level tokenization."""
    import string
    for punct in string.punctuation:
        text = text.replace(punct, ' ' + punct + ' ')
    return text.split()


def tokenize_char(text: str) -> List[str]:
    """Character-level tokenization."""
    return list(text)


# ============================================================================
# N-GRAM MODEL WITH MULTIPLE SMOOTHING METHODS
# ============================================================================

class NgramModelSmoothed:
    """N-gram model supporting multiple smoothing techniques."""
    
    def __init__(self, n: int, tokenizer: str = 'word'):
        self.n = n
        self.tokenizer = tokenizer
        self.tokenize = tokenize_char if tokenizer == 'char' else tokenize_word
        
        # Core data structures
        self.context: Dict[Tuple, List[str]] = {}
        self.ngram_counter: Dict[Tuple, int] = {}
        self.vocab: set = set()
        self.total_ngrams = 0
        
        # For backoff models
        self.lower_order_models: List['NgramModelSmoothed'] = []
    
    def get_ngrams(self, tokens: List[str]) -> List[Tuple]:
        """Generate n-grams."""
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
    
    # ========================================================================
    # SMOOTHING METHOD 1: ADD-K SMOOTHING
    # ========================================================================
    
    def prob_add_k(self, context: Tuple, token: str, k: float = 1.0) -> float:
        """
        Add-k smoothing (generalized Laplace).
        
        P(w|context) = (count(context, w) + k) / (count(context) + k*|V|)
        
        Args:
            k: Smoothing parameter
                k=0: no smoothing (MLE)
                k=1: Laplace (add-one)
                k<1: fractional counts (often better)
        """
        vocab_size = len(self.vocab)
        count_ngram = self.ngram_counter.get((context, token), 0)
        count_context = len(self.context.get(context, []))
        
        return (count_ngram + k) / (count_context + k * vocab_size)
    
    # ========================================================================
    # SMOOTHING METHOD 2: STUPID BACKOFF
    # ========================================================================
    
    def prob_stupid_backoff(self, context: Tuple, token: str, alpha: float = 0.4) -> float:
        """
        Stupid Backoff (Brants et al., 2007).
        
        If we've seen the n-gram: S(w|context) = count(context,w) / count(context)
        Otherwise: S(w|context) = α * S(w|shorter_context)
        Base case (unigram): S(w) = count(w) / total_tokens
        
        Args:
            alpha: Backoff weight (literature suggests α≈0.4)
        """
        # Try current order
        ngram = (context, token)
        if ngram in self.ngram_counter and context in self.context:
            count_ngram = self.ngram_counter[ngram]
            count_context = len(self.context[context])
            return count_ngram / count_context
        
        # Backoff to lower order
        if len(context) > 0 and self.lower_order_models:
            # Get appropriate lower-order model
            lower_n = len(context)  # n-1 order
            if lower_n > 0 and lower_n - 1 < len(self.lower_order_models):
                lower_model = self.lower_order_models[lower_n - 1]
                shorter_context = context[1:] if len(context) > 0 else ()
                return alpha * lower_model.prob_stupid_backoff(shorter_context, token, alpha)
        
        # Base case: unigram with uniform fallback
        if len(self.vocab) > 0:
            return 1.0 / len(self.vocab)
        return 1e-10
    
    # ========================================================================
    # SMOOTHING METHOD 3: LINEAR INTERPOLATION
    # ========================================================================
    
    def prob_linear_interpolation(self, context: Tuple, token: str, 
                                  lambdas: Optional[List[float]] = None) -> float:
        """
        Linear Interpolation (Jelinek-Mercer smoothing).
        
        P(w|context) = λ_n * P_ML(w|context) + λ_{n-1} * P(w|shorter_context) + ...
        
        Where λ_i are interpolation weights summing to 1.
        
        Args:
            lambdas: Interpolation weights [λ_n, λ_{n-1}, ..., λ_1]
                    If None, uses uniform weights
        """
        if lambdas is None:
            # Uniform weights
            lambdas = [1.0 / self.n] * self.n
        
        # Ensure we have lower-order models
        if len(self.lower_order_models) < self.n - 1:
            # Fall back to add-k
            return self.prob_add_k(context, token, k=1.0)
        
        prob = 0.0
        current_context = context
        
        # Interpolate from highest to lowest order
        for i, lambda_weight in enumerate(lambdas):
            if i >= len(self.lower_order_models) + 1:
                break
            
            # Get MLE probability for current order
            if i == 0:
                # Highest order (this model)
                mle_prob = self._get_mle_prob(current_context, token)
            else:
                # Lower order models
                model_idx = i - 1
                if model_idx < len(self.lower_order_models):
                    mle_prob = self.lower_order_models[model_idx]._get_mle_prob(
                        current_context, token)
                else:
                    mle_prob = 0.0
            
            prob += lambda_weight * mle_prob
            
            # Shorten context for next iteration
            if len(current_context) > 0:
                current_context = current_context[1:]
        
        return max(prob, 1e-10)  # Avoid zero probability
    
    def _get_mle_prob(self, context: Tuple, token: str) -> float:
        """Maximum Likelihood Estimate probability."""
        ngram = (context, token)
        if ngram in self.ngram_counter and context in self.context:
            count_ngram = self.ngram_counter[ngram]
            count_context = len(self.context[context])
            return count_ngram / count_context
        
        # Unseen n-gram
        if len(self.vocab) > 0:
            return 1.0 / len(self.vocab)
        return 1e-10
    
    # ========================================================================
    # EVALUATION METHODS
    # ========================================================================
    
    def perplexity(self, text: str, method: str = 'add_k', **kwargs) -> float:
        """Calculate perplexity using specified smoothing method."""
        tokens = self.tokenize(text)
        ngrams = self.get_ngrams(tokens)
        
        if not ngrams:
            return float('inf')
        
        log_prob_sum = 0.0
        for context, target in ngrams:
            if method == 'add_k':
                k = kwargs.get('k', 1.0)
                prob = self.prob_add_k(context, target, k)
            elif method == 'stupid_backoff':
                alpha = kwargs.get('alpha', 0.4)
                prob = self.prob_stupid_backoff(context, target, alpha)
            elif method == 'linear_interpolation':
                lambdas = kwargs.get('lambdas', None)
                prob = self.prob_linear_interpolation(context, target, lambdas)
            else:
                raise ValueError(f"Unknown smoothing method: {method}")
            
            if prob > 0:
                log_prob_sum += math.log2(prob)
            else:
                log_prob_sum += -100
        
        avg_log_prob = log_prob_sum / len(ngrams)
        return math.pow(2, -avg_log_prob)
    
    def cross_entropy(self, text: str, method: str = 'add_k', **kwargs) -> float:
        """Calculate cross-entropy using specified smoothing method."""
        tokens = self.tokenize(text)
        ngrams = self.get_ngrams(tokens)
        
        if not ngrams:
            return float('inf')
        
        log_prob_sum = 0.0
        for context, target in ngrams:
            if method == 'add_k':
                k = kwargs.get('k', 1.0)
                prob = self.prob_add_k(context, target, k)
            elif method == 'stupid_backoff':
                alpha = kwargs.get('alpha', 0.4)
                prob = self.prob_stupid_backoff(context, target, alpha)
            elif method == 'linear_interpolation':
                lambdas = kwargs.get('lambdas', None)
                prob = self.prob_linear_interpolation(context, target, lambdas)
            else:
                raise ValueError(f"Unknown smoothing method: {method}")
            
            if prob > 0:
                log_prob_sum += math.log2(prob)
            else:
                log_prob_sum += -100
        
        return -log_prob_sum / len(ngrams)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_split_data(filepath: str, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Load and split data into train/val/test (80/10/10)."""
    print(f"Loading data from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sentences = [line.strip() for line in lines if line.strip()]
    
    print(f"  Total sentences: {len(sentences):,}")
    
    random.seed(seed)
    random.shuffle(sentences)
    
    n = len(sentences)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train = sentences[:train_end]
    val = sentences[train_end:val_end]
    test = sentences[val_end:]
    
    print(f"  Train: {len(train):,}")
    print(f"  Val:   {len(val):,}")
    print(f"  Test:  {len(test):,}")
    
    return train, val, test


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def tune_alpha_stupid_backoff(models: List[NgramModelSmoothed], 
                               val_text: str,
                               alpha_values: List[float] = None) -> Dict[int, float]:
    """Tune α for Stupid Backoff on validation data."""
    if alpha_values is None:
        # Literature suggests α around 0.4, test nearby values
        alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\n  Tuning Stupid Backoff α on validation data...")
    best_alphas = {}
    
    for model in models:
        if model.n == 1:
            best_alphas[model.n] = 0.4  # Unigram doesn't use backoff
            continue
        
        best_alpha = 0.4
        best_ppl = float('inf')
        
        for alpha in alpha_values:
            ppl = model.perplexity(val_text, method='stupid_backoff', alpha=alpha)
            if ppl < best_ppl:
                best_ppl = ppl
                best_alpha = alpha
        
        best_alphas[model.n] = best_alpha
        print(f"    n={model.n}: best α={best_alpha:.2f} (PPL={best_ppl:.2f})")
    
    return best_alphas


def tune_lambdas_interpolation(models: List[NgramModelSmoothed],
                                val_text: str,
                                n: int) -> List[float]:
    """
    Tune λ weights for linear interpolation using EM on validation data.
    Simplified version: grid search over reasonable values.
    """
    if n == 1:
        return [1.0]
    
    print(f"\n  Tuning interpolation weights for {n}-gram...")
    
    # Simple grid search for 2-gram and 3-gram
    # For higher n, use heuristic weights
    if n == 2:
        candidates = [
            [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]
        ]
    elif n == 3:
        candidates = [
            [0.6, 0.3, 0.1], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.5, 0.4, 0.1]
        ]
    else:
        # For n>3, use decreasing weights (heuristic from literature)
        weights = []
        total = 0.0
        for i in range(n):
            w = 2.0 ** (n - i - 1)  # Exponentially decreasing
            weights.append(w)
            total += w
        lambdas = [w / total for w in weights]
        print(f"    Using heuristic weights: {[f'{l:.3f}' for l in lambdas]}")
        return lambdas
    
    model = models[n - 1]  # Get n-gram model
    best_lambdas = candidates[0]
    best_ppl = float('inf')
    
    for lambdas in candidates:
        ppl = model.perplexity(val_text, method='linear_interpolation', lambdas=lambdas)
        if ppl < best_ppl:
            best_ppl = ppl
            best_lambdas = lambdas
    
    print(f"    Best λ: {[f'{l:.2f}' for l in best_lambdas]} (PPL={best_ppl:.2f})")
    return best_lambdas


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_smoothing_experiment(
    train_sentences: List[str],
    val_sentences: List[str],
    test_sentences: List[str],
    n_values: List[int] = [2, 3, 4, 5],
    tokenizer: str = 'char',
    output_dir: str = "smoothing_results"
):
    """Run smoothing comparison experiment."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("N-GRAM SMOOTHING TECHNIQUES COMPARISON")
    print("=" * 80)
    print(f"\nTokenizer: {tokenizer}-level")
    print(f"N-gram orders: {n_values}")
    
    # Prepare texts
    train_text = ' '.join(train_sentences)
    val_text = ' '.join(val_sentences)
    test_text = ' '.join(test_sentences)
    
    # Train models for all n-grams
    print(f"\n[1/3] Training {len(n_values)} n-gram models...")
    models = {}
    
    for n in n_values:
        print(f"\n  Training {n}-gram model...")
        model = NgramModelSmoothed(n, tokenizer=tokenizer)
        model.train(train_text)
        
        # Train lower-order models for backoff/interpolation
        if n > 1:
            lower_models = []
            for lower_n in range(1, n):
                if lower_n in models:
                    lower_models.append(models[lower_n])
                else:
                    lower_model = NgramModelSmoothed(lower_n, tokenizer=tokenizer)
                    lower_model.train(train_text)
                    lower_models.append(lower_model)
                    if lower_n not in models:
                        models[lower_n] = lower_model
            model.lower_order_models = lower_models
        
        models[n] = model
        print(f"    Vocab: {len(model.vocab):,}, Unique n-grams: {len(model.ngram_counter):,}")
    
    # Tune hyperparameters
    print(f"\n[2/3] Tuning hyperparameters on validation data...")
    
    # Tune Stupid Backoff α
    best_alphas = tune_alpha_stupid_backoff(list(models.values()), val_text)
    
    # Tune Linear Interpolation λ
    best_lambdas = {}
    for n in n_values:
        best_lambdas[n] = tune_lambdas_interpolation(list(models.values()), val_text, n)
    
    # Run experiments
    print(f"\n[3/3] Evaluating all smoothing methods on test data...")
    
    results = []
    k_values = [0.01, 0.1, 0.5, 1.0, 2.0]
    
    for n in n_values:
        model = models[n]
        print(f"\n  {n}-gram results:")
        
        # Add-k smoothing
        for k in k_values:
            ppl = model.perplexity(test_text, method='add_k', k=k)
            ce = model.cross_entropy(test_text, method='add_k', k=k)
            results.append({
                'n': n,
                'method': f'Add-k (k={k})',
                'k': k,
                'ppl': ppl,
                'ce': ce
            })
            print(f"    Add-k (k={k:4.2f}): PPL={ppl:8.2f}, CE={ce:6.4f}")
        
        # Stupid Backoff
        if n > 1:
            alpha = best_alphas[n]
            ppl = model.perplexity(test_text, method='stupid_backoff', alpha=alpha)
            ce = model.cross_entropy(test_text, method='stupid_backoff', alpha=alpha)
            results.append({
                'n': n,
                'method': f'Stupid Backoff (α={alpha:.2f})',
                'alpha': alpha,
                'ppl': ppl,
                'ce': ce
            })
            print(f"    Stupid Backoff (α={alpha:.2f}): PPL={ppl:8.2f}, CE={ce:6.4f}")
        
        # Linear Interpolation
        if n > 1:
            lambdas = best_lambdas[n]
            ppl = model.perplexity(test_text, method='linear_interpolation', lambdas=lambdas)
            ce = model.cross_entropy(test_text, method='linear_interpolation', lambdas=lambdas)
            results.append({
                'n': n,
                'method': 'Linear Interpolation',
                'lambdas': lambdas,
                'ppl': ppl,
                'ce': ce
            })
            lambda_str = '[' + ','.join([f'{l:.2f}' for l in lambdas]) + ']'
            print(f"    Linear Interp {lambda_str}: PPL={ppl:8.2f}, CE={ce:6.4f}")
    
    # Save results
    results_file = os.path.join(output_dir, f'smoothing_results_{tokenizer}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Create summary
    print_summary(results, tokenizer, output_dir)
    
    return results


def print_summary(results: List[Dict], tokenizer: str, output_dir: str):
    """Print and save results summary."""
    print("\n" + "=" * 80)
    print(f"SMOOTHING COMPARISON SUMMARY ({tokenizer}-level)")
    print("=" * 80)
    
    # Group by n-gram
    by_n = {}
    for r in results:
        n = r['n']
        if n not in by_n:
            by_n[n] = []
        by_n[n].append(r)
    
    for n in sorted(by_n.keys()):
        print(f"\n{n}-gram Models:")
        print(f"  {'Method':<30} {'PPL':>12} {'CE':>10}")
        print(f"  {'-'*54}")
        
        # Sort by PPL
        sorted_results = sorted(by_n[n], key=lambda x: x['ppl'])
        for r in sorted_results:
            print(f"  {r['method']:<30} {r['ppl']:>12.2f} {r['ce']:>10.4f}")
        
        # Highlight best
        best = sorted_results[0]
        print(f"\n  ✅ BEST: {best['method']} (PPL={best['ppl']:.2f})")
    
    # Save CSV
    csv_file = os.path.join(output_dir, f'smoothing_summary_{tokenizer}.csv')
    with open(csv_file, 'w') as f:
        f.write("N,Method,PPL,CE\n")
        for r in sorted(results, key=lambda x: (x['n'], x['ppl'])):
            f.write(f"{r['n']},{r['method']},{r['ppl']:.4f},{r['ce']:.4f}\n")
    
    print(f"\nResults saved to: {csv_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    data_file = "data/all_twi.txt"
    
    print("="* 80)
    print("N-GRAM SMOOTHING EXPERIMENT")
    print("=" * 80)
    
    # Load data
    train, val, test = load_and_split_data(data_file)
    
    # Run for character-level (best from previous experiment)
    print("\n" + "="*80)
    print("CHARACTER-LEVEL TOKENIZATION")
    print("="*80)
    results_char = run_smoothing_experiment(
        train, val, test,
        n_values=[2, 3, 4, 5],
        tokenizer='char',
        output_dir='results/experiment_2_smoothing'
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
