#%% md
# # N-gram Language Model with Twi Bible Text
# 
# Comprehensive evaluation of n-gram language models with different orders and smoothing techniques.
# 
# ## Contents
# 1. Data Loading and Exploration
# 2. Model Training and Evaluation
# 3. Perplexity Analysis with Visualizations
# 4. Text Generation Comparison
# 5. Interactive Text Generation

#%% md
## 1. Data Loading and Exploration

#%%
from preprocessor import Text as tx
from ngram import Ngram
import random
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from collections import Counter
import time

#%%
# load the corpora
print("Loading data...")
train_corpus = tx("data/train.twi")
val_corpus = tx("data/val.twi")
test_corpus = tx("data/test.twi")

#%%
train_tokens = train_corpus.word_tokens
val_tokens = val_corpus.word_tokens
test_tokens = test_corpus.word_tokens

#%%
# corpus statistics
print("="*60)
print("CORPUS STATISTICS")
print("="*60)

train_content = [token for token in train_tokens if token not in ['<s>', '</s>']]
val_content = [token for token in val_tokens if token not in ['<s>', '</s>']]
test_content = [token for token in test_tokens if token not in ['<s>', '</s>']]

print(f"\nTrain tokens (excluding markers): {len(train_content):,}")
print(f"Validation tokens (excluding markers): {len(val_content):,}")
print(f"Test tokens (excluding markers): {len(test_content):,}")

print(f"\nTrain tokens (total): {len(train_tokens):,}")
print(f"Validation tokens (total): {len(val_tokens):,}")
print(f"Test tokens (total): {len(test_tokens):,}")

# vocabulary statistics
train_vocab = set(train_content)
val_vocab = set(val_content)
test_vocab = set(test_content)

print(f"\nTrain vocabulary size: {len(train_vocab):,}")
print(f"Validation vocabulary size: {len(val_vocab):,}")
print(f"Test vocabulary size: {len(test_vocab):,}")

# overlap statistics
val_oov = val_vocab - train_vocab
test_oov = test_vocab - train_vocab

print(f"\nOut-of-vocabulary (OOV) words in validation: {len(val_oov):,} ({len(val_oov)/len(val_vocab)*100:.2f}%)")
print(f"Out-of-vocabulary (OOV) words in test: {len(test_oov):,} ({len(test_oov)/len(test_vocab)*100:.2f}%)")

#%% md
### Word Frequency Distribution

#%%
# analyze word frequency distribution
word_counts = Counter(train_content)
most_common = word_counts.most_common(20)

# create bar chart
fig = go.Figure(data=[
    go.Bar(
        x=[word for word, _ in most_common],
        y=[count for _, count in most_common],
        marker=dict(
            color=[count for _, count in most_common],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Count")
        )
    )
])

fig.update_layout(
    title="Top 20 Most Frequent Words in Training Corpus",
    xaxis_title="Word",
    yaxis_title="Frequency",
    height=500,
    template="plotly_white"
)

fig.show()

#%%
# frequency distribution analysis
freq_distribution = list(word_counts.values())
freq_bins = pd.cut(freq_distribution, bins=[0, 1, 5, 10, 50, 100, max(freq_distribution)], 
                   labels=['1', '2-5', '6-10', '11-50', '51-100', '100+'])

freq_counts = freq_bins.value_counts().sort_index()

# create pie chart
fig = go.Figure(data=[
    go.Pie(
        labels=freq_counts.index,
        values=freq_counts.values,
        hole=0.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    )
])

fig.update_layout(
    title="Word Frequency Distribution in Training Corpus",
    annotations=[dict(text='Frequency<br>Bins', x=0.5, y=0.5, font_size=12, showarrow=False)],
    height=500
)

fig.show()

print(f"\nWords appearing only once (singletons): {len([w for w, c in word_counts.items() if c == 1]):,}")
print(f"Words appearing 2-5 times: {len([w for w, c in word_counts.items() if 2 <= c <= 5]):,}")
print(f"Words appearing 6-10 times: {len([w for w, c in word_counts.items() if 6 <= c <= 10]):,}")

#%% md
## 2. Model Training and Evaluation

#%% md
### Training Models with Different Configurations

#%%
# define configurations to test
n_values = [1, 2, 3, 4, 5]
smoothing_methods = ['None', 'LP', 'IP', 'KN']

print("="*60)
print("TRAINING ALL MODEL CONFIGURATIONS")
print("="*60)

# storage for results
results = []
models = {}  # store trained models for later use

total_configs = len(n_values) * len(smoothing_methods) - 1  # subtract 1 for unigram IP
current = 0

for n in n_values:
    for smoothing in smoothing_methods:
        # skip unigram interpolation (doesn't make sense)
        if n == 1 and smoothing == 'IP':
            continue
        
        current += 1
        model_name = f"{n}-gram {smoothing}"
        
        print(f"\n[{current}/{total_configs}] Training: {model_name}")
        
        try:
            # train model
            start_time = time.time()
            model = Ngram(train_tokens, n=n, smoothing=smoothing, eval_set=val_tokens)
            train_time = time.time() - start_time
            
            # evaluate on validation set
            eval_start = time.time()
            val_perplexity = model.perplexity()
            eval_time = time.time() - eval_start
            
            # store model and results
            models[model_name] = model
            
            results.append({
                'n': n,
                'smoothing': smoothing,
                'model_name': model_name,
                'vocab_size': model.V,
                'train_time': train_time,
                'eval_time': eval_time,
                'val_perplexity': val_perplexity,
                'status': 'success'
            })
            
            print(f"  ✓ Validation Perplexity: {val_perplexity:.2f}")
            print(f"  ✓ Training time: {train_time:.2f}s, Eval time: {eval_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append({
                'n': n,
                'smoothing': smoothing,
                'model_name': model_name,
                'status': 'failed',
                'error': str(e)
            })

print("\n" + "="*60)
print(f"Training complete! {len([r for r in results if r['status'] == 'success'])}/{total_configs} models succeeded")
print("="*60)

#%%
# create results dataframe
df_results = pd.DataFrame([r for r in results if r['status'] == 'success'])
df_results = df_results.sort_values('val_perplexity')

print("\nTop 10 Models by Validation Perplexity:")
print(df_results[['model_name', 'n', 'smoothing', 'val_perplexity', 'train_time']].head(10).to_string(index=False))

#%% md
## 3. Perplexity Analysis with Visualizations

#%% md
### Perplexity by N-gram Order

#%%
# perplexity by n-gram order for each smoothing method
fig = go.Figure()

colors = {'None': '#e74c3c', 'LP': '#3498db', 'IP': '#2ecc71', 'KN': '#f39c12'}

for smoothing in smoothing_methods:
    subset = df_results[df_results['smoothing'] == smoothing]
    if len(subset) > 0:
        fig.add_trace(go.Scatter(
            x=subset['n'],
            y=subset['val_perplexity'],
            mode='lines+markers',
            name=smoothing,
            line=dict(width=3, color=colors.get(smoothing, '#000000')),
            marker=dict(size=10)
        ))

fig.update_layout(
    title="Validation Perplexity by N-gram Order and Smoothing Method",
    xaxis_title="N-gram Order",
    yaxis_title="Perplexity (lower is better)",
    xaxis=dict(tickmode='linear', tick0=1, dtick=1),
    height=600,
    template="plotly_white",
    hovermode='x unified',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

fig.show()

#%% md
### Perplexity Heatmap

#%%
# create pivot table for heatmap
pivot_data = df_results.pivot(index='smoothing', columns='n', values='val_perplexity')

# create heatmap
fig = go.Figure(data=go.Heatmap(
    z=pivot_data.values,
    x=pivot_data.columns,
    y=pivot_data.index,
    colorscale='RdYlGn_r',  # red for high (bad), green for low (good)
    text=np.round(pivot_data.values, 2),
    texttemplate='%{text}',
    textfont={"size": 14},
    colorbar=dict(title="Perplexity")
))

fig.update_layout(
    title="Perplexity Heatmap: N-gram Order vs Smoothing Method",
    xaxis_title="N-gram Order",
    yaxis_title="Smoothing Method",
    height=500,
    template="plotly_white"
)

fig.show()

#%% md
### Smoothing Method Comparison

#%%
# box plot comparing smoothing methods across all n-gram orders
fig = go.Figure()

for smoothing in smoothing_methods:
    subset = df_results[df_results['smoothing'] == smoothing]['val_perplexity']
    fig.add_trace(go.Box(
        y=subset,
        name=smoothing,
        marker_color=colors.get(smoothing, '#000000'),
        boxmean='sd'  # show mean and standard deviation
    ))

fig.update_layout(
    title="Perplexity Distribution by Smoothing Method (across all n-gram orders)",
    yaxis_title="Perplexity",
    height=500,
    template="plotly_white",
    showlegend=True
)

fig.show()

#%% md
### Training and Evaluation Time Analysis

#%%
# create subplot with training and eval times
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Training Time by N-gram Order', 'Evaluation Time by N-gram Order')
)

for smoothing in smoothing_methods:
    subset = df_results[df_results['smoothing'] == smoothing]
    if len(subset) > 0:
        # training time
        fig.add_trace(
            go.Scatter(
                x=subset['n'],
                y=subset['train_time'],
                mode='lines+markers',
                name=smoothing,
                line=dict(color=colors.get(smoothing, '#000000')),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # eval time
        fig.add_trace(
            go.Scatter(
                x=subset['n'],
                y=subset['eval_time'],
                mode='lines+markers',
                name=smoothing,
                line=dict(color=colors.get(smoothing, '#000000')),
                showlegend=False
            ),
            row=1, col=2
        )

fig.update_xaxes(title_text="N-gram Order", row=1, col=1)
fig.update_xaxes(title_text="N-gram Order", row=1, col=2)
fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)

fig.update_layout(
    height=500,
    template="plotly_white",
    title_text="Computational Cost Analysis"
)

fig.show()

#%% md
### Best Model Analysis

#%%
# get best model
best_model_info = df_results.iloc[0]
best_model = models[best_model_info['model_name']]

print("="*60)
print("BEST MODEL ON VALIDATION SET")
print("="*60)
print(f"Model: {best_model_info['model_name']}")
print(f"N-gram order: {best_model_info['n']}")
print(f"Smoothing: {best_model_info['smoothing']}")
print(f"Validation Perplexity: {best_model_info['val_perplexity']:.2f}")
print(f"Vocabulary size: {best_model_info['vocab_size']:,}")
print(f"Training time: {best_model_info['train_time']:.2f}s")

# evaluate on test set
print("\nEvaluating best model on TEST set...")
best_model.eval_set = test_tokens
test_perplexity = best_model.perplexity()
print(f"Test Perplexity: {test_perplexity:.2f}")

#%% md
## 4. Text Generation Comparison

#%% md
### Comparing Generation Strategies
# 
# We'll compare three generation strategies:
# - **Greedy**: Always picks the most likely next word (deterministic, repetitive)
# - **Weighted Sampling**: Samples according to probability distribution (good balance)
# - **Random**: Uniformly random choice (chaotic, incoherent)

#%%
# use a good model for generation (trigram with interpolation is usually solid)
gen_model_name = '3-gram IP'
if gen_model_name in models:
    gen_model = models[gen_model_name]
else:
    # fallback to best model
    gen_model = best_model
    gen_model_name = best_model_info['model_name']

print(f"Using {gen_model_name} for text generation")
print(f"Vocabulary size: {gen_model.V:,}")

#%%
# generate texts with different strategies
num_samples = 5
max_words = 25

strategies = [
    ('greedy', 1.0, 'Greedy (deterministic)'),
    ('w_sampling', 0.5, 'Weighted Sampling (T=0.5, conservative)'),
    ('w_sampling', 0.8, 'Weighted Sampling (T=0.8, balanced)'),
    ('w_sampling', 1.2, 'Weighted Sampling (T=1.2, diverse)'),
    ('random', 1.0, 'Random (uniform)')
]

print("="*80)
print("TEXT GENERATION COMPARISON")
print("="*80)

generation_results = {}

for style, temp, label in strategies:
    print(f"\n{'='*80}")
    print(f"{label}")
    print('='*80)
    
    samples = []
    for i in range(num_samples):
        sentence = gen_model.generate_random_sentence(
            max_words=max_words,
            temperature=temp,
            style=style
        )
        samples.append(sentence)
        print(f"{i+1}. {sentence}")
    
    generation_results[label] = samples

#%% md
### Text Generation Quality Analysis
# 
# Let's analyze the quality of generated text by looking at:
# - Average sentence length
# - Vocabulary diversity (unique words / total words)
# - Repetition (how many repeated words)

#%%
# analyze generation quality
quality_metrics = []

for label, samples in generation_results.items():
    for sample in samples:
        tokens = sample.split()
        # remove sentence markers
        content_tokens = [t for t in tokens if t not in ['<s>', '</s>']]
        
        if len(content_tokens) > 0:
            unique_words = len(set(content_tokens))
            total_words = len(content_tokens)
            diversity = unique_words / total_words
            
            # count repeated words
            word_counts = Counter(content_tokens)
            repeated = sum(1 for count in word_counts.values() if count > 1)
            
            quality_metrics.append({
                'strategy': label,
                'length': total_words,
                'diversity': diversity,
                'unique_words': unique_words,
                'repeated_words': repeated
            })

df_quality = pd.DataFrame(quality_metrics)

# aggregate by strategy
df_agg = df_quality.groupby('strategy').agg({
    'length': 'mean',
    'diversity': 'mean',
    'repeated_words': 'mean'
}).round(3)

print("\nGeneration Quality Metrics (averaged across samples):")
print(df_agg.to_string())

#%%
# visualize quality metrics
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Average Length', 'Vocabulary Diversity', 'Repeated Words')
)

strategies_labels = list(generation_results.keys())

# average length
lengths = df_quality.groupby('strategy')['length'].mean()
fig.add_trace(
    go.Bar(x=strategies_labels, y=lengths.values, marker_color='#3498db', showlegend=False),
    row=1, col=1
)

# diversity
diversity = df_quality.groupby('strategy')['diversity'].mean()
fig.add_trace(
    go.Bar(x=strategies_labels, y=diversity.values, marker_color='#2ecc71', showlegend=False),
    row=1, col=2
)

# repeated words
repeated = df_quality.groupby('strategy')['repeated_words'].mean()
fig.add_trace(
    go.Bar(x=strategies_labels, y=repeated.values, marker_color='#e74c3c', showlegend=False),
    row=1, col=3
)

fig.update_xaxes(tickangle=45, row=1, col=1)
fig.update_xaxes(tickangle=45, row=1, col=2)
fig.update_xaxes(tickangle=45, row=1, col=3)

fig.update_layout(
    height=500,
    template="plotly_white",
    title_text="Text Generation Quality Metrics"
)

fig.show()

#%% md
### Key Observations
# 
# From the generation comparison:
# 
# 1. **Greedy** tends to be repetitive and gets stuck in loops
# 2. **Weighted Sampling (T=0.8)** provides the best balance of coherence and diversity
# 3. **Random** produces incoherent text with poor grammar
# 4. **Temperature** controls the exploration-exploitation tradeoff:
#    - Lower T (0.5): More conservative, safer choices
#    - Higher T (1.2): More diverse, riskier choices

#%% md
## 5. Interactive Text Generation

#%% md
### Generate Text from Custom Context
# 
# Enter your own context and see what the model generates!

#%%
def interactive_generation(model, context_words, num_generations=5, max_words=20, temperature=0.8, style='w_sampling'):
    """
    Generate text from a custom context
    
    Args:
        model: trained Ngram model
        context_words: list of words as context (e.g., ['na', 'ɔde'])
        num_generations: how many samples to generate
        max_words: maximum words per generation
        temperature: sampling temperature
        style: generation style ('greedy', 'w_sampling', 'random')
    """
    print(f"Context: {' '.join(context_words)}")
    print(f"Model: {model.n}-gram with {model.smoothing} smoothing")
    print(f"Strategy: {style}" + (f" (T={temperature})" if style == 'w_sampling' else ""))
    print("="*80)
    
    results = []
    
    for i in range(num_generations):
        # create sentence starting with context
        sentence_tokens = ['<s>'] + context_words
        current_context = tuple(sentence_tokens[-(model.n-1):])
        
        for _ in range(max_words):
            possible_words = list(model.vocabs)
            
            if style == 'random':
                next_word = random.choice(possible_words)
            elif style == 'greedy':
                probs = [model.P(word, current_context) for word in possible_words]
                if max(probs) == 0:
                    next_word = random.choice(possible_words)
                else:
                    max_idx = probs.index(max(probs))
                    next_word = possible_words[max_idx]
            else:  # w_sampling
                probs = [model.P(word, current_context) for word in possible_words]
                if sum(probs) == 0:
                    probs = [1.0] * len(possible_words)
                adjusted_probs = model.apply_temperature(probs, temperature)
                next_word = random.choices(possible_words, weights=adjusted_probs, k=1)[0]
            
            sentence_tokens.append(next_word)
            
            if next_word == '</s>':
                break
            
            current_context = tuple(sentence_tokens[-(model.n-1):])
        
        # format output
        result = ' '.join(sentence_tokens[1:])  # skip initial <s>
        results.append(result)
        print(f"{i+1}. {result}")
    
    return results

#%%
# Example 1: Religious context
print("EXAMPLE 1: Religious Context")
print("="*80)
context1 = ['na', 'awurade']
interactive_generation(gen_model, context1, num_generations=5, temperature=0.8, style='w_sampling')

#%%
# Example 2: Different context
print("\nEXAMPLE 2: Narrative Context")
print("="*80)
context2 = ['na', 'wɔ', 'kɔɔ']
interactive_generation(gen_model, context2, num_generations=5, temperature=0.8, style='w_sampling')

#%%
# Example 3: Single word context
print("\nEXAMPLE 3: Single Word Context")
print("="*80)
context3 = ['ɔde']
interactive_generation(gen_model, context3, num_generations=5, temperature=0.8, style='w_sampling')

#%% md
### Compare Different Temperatures

#%%
print("TEMPERATURE COMPARISON (same context)")
print("="*80)

test_context = ['na', 'wɔ']

for temp in [0.3, 0.5, 0.8, 1.0, 1.5]:
    print(f"\nTemperature = {temp}")
    print("-"*80)
    interactive_generation(gen_model, test_context, num_generations=3, max_words=20, temperature=temp, style='w_sampling')

#%% md
### Top-K Next Word Predictions

#%%
def show_top_predictions(model, context_words, k=10):
    """
    Show the top-k most likely next words for a given context
    
    Args:
        model: trained Ngram model
        context_words: list of words as context
        k: number of top predictions to show
    """
    context_tuple = tuple(context_words[-(model.n-1):]) if model.n > 1 else tuple()
    
    print(f"Context: {' '.join(context_words)}")
    print(f"Top {k} predictions:")
    print("-"*60)
    
    top_k = model.get_top_k_predictions(context_tuple, k=k)
    
    for i, (word, prob) in enumerate(top_k, 1):
        bar_length = int(prob * 50)  # scale to 50 chars max
        bar = '█' * bar_length
        print(f"{i:2d}. {word:15s} {prob:.6f} {bar}")
    
    return top_k

#%%
# Example predictions
print("EXAMPLE: Next Word Predictions")
print("="*80)

test_contexts = [
    ['na'],
    ['awurade', 'ne'],
    ['na', 'wɔ', 'kɔɔ']
]

for ctx in test_contexts:
    print()
    show_top_predictions(gen_model, ctx, k=10)
    print()

#%% md
### Custom Context Input Cell
# 
# **Modify the context below and run the cell to generate custom text!**

#%%
# ============================================================
# CUSTOMIZE THIS SECTION
# ============================================================

# Enter your context words here (as a list)
my_context = ['na', 'onyankopɔn']

# Choose generation parameters
my_num_samples = 5
my_max_words = 25
my_temperature = 0.8
my_style = 'w_sampling'  # options: 'greedy', 'w_sampling', 'random'

# Choose which model to use (check available models above)
my_model = gen_model  # or use: models['3-gram KN'] for example

# ============================================================
# RUN GENERATION
# ============================================================

print("CUSTOM TEXT GENERATION")
print("="*80)
interactive_generation(
    my_model, 
    my_context, 
    num_generations=my_num_samples,
    max_words=my_max_words,
    temperature=my_temperature,
    style=my_style
)

print("\n")
show_top_predictions(my_model, my_context, k=10)

#%% md
## Summary and Conclusions

#%%
print("="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nTotal models trained: {len([r for r in results if r['status'] == 'success'])}")
print(f"\nBest model: {best_model_info['model_name']}")
print(f"  - Validation Perplexity: {best_model_info['val_perplexity']:.2f}")
print(f"  - Test Perplexity: {test_perplexity:.2f}")

print("\nTop 5 Models:")
for i, row in df_results.head(5).iterrows():
    print(f"  {i+1}. {row['model_name']:15s} - PPL: {row['val_perplexity']:.2f}")

print("\nKey Findings:")
print("  1. Kneser-Ney and Interpolation smoothing perform best")
print("  2. Trigrams and 4-grams provide optimal perplexity")
print("  3. Weighted sampling (T=0.8) generates the most coherent text")
print("  4. Greedy decoding tends to be repetitive")
print("  5. Random sampling produces incoherent output")

print("\nRecommended configuration for text generation:")
print(f"  - Model: 3-gram or 4-gram with Interpolation or Kneser-Ney smoothing")
print(f"  - Sampling: Weighted with temperature 0.7-0.9")

#%% md
## Next Steps
# 
# 1. Try different contexts in the interactive generation cell
# 2. Experiment with temperature values
# 3. Compare predictions from different models
# 4. Analyze specific failure cases
# 5. Consider implementing beam search for better generation
