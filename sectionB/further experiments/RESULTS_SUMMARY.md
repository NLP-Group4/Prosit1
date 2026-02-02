# N-Gram Experiments Summary

## Experiments Conducted

### Experiment 1: Learning Curves
**Duration**: ~90 minutes  
**Purpose**: Understand data scaling effects and compare character vs word tokenization

**Results**:
- Character-level: 4.05 PPL (5-gram with Laplace)
- Word-level: 664.68 PPL (1-gram best)
- **Finding**: Character 164x better than word

### Experiment 2: Smoothing Optimization
**Duration**: ~10 minutes  
**Purpose**: Find optimal smoothing method and hyperparameters

**Results**:
- Stupid Backoff (α=0.7): 3.71 PPL ✅
- Add-k (k=0.1): 3.81 PPL
- Laplace (k=1.0): 4.05 PPL (baseline)
- **Finding**: Stupid Backoff best, α=0.7 optimal for Twi

### Experiment 3: Final Validation
**Duration**: ~15 minutes  
**Purpose**: Complete comparison with new data split

**Results**:
- Character 5-gram + Stupid Backoff: **3.72 PPL** ✅
- Word 2-gram + Stupid Backoff: 475.72 PPL
- Word 5-gram MLE: 1,348,455,405 PPL (catastrophic!)
- **Finding**: Smoothing essential for word-level (99.99% improvement)

## All Results Table

### Character-Level

| N | MLE | Add-k (k=0.1) | Stupid Backoff | Best | Improvement |
|---|-----|---------------|----------------|------|-------------|
| 1 | 15.08 | 15.08 | 15.08 | 15.08 | 0.0% |
| 2 | 8.37 | 8.36 | **8.36** | 8.36 | 0.1% |
| 3 | 5.96 | 5.94 | **5.93** | 5.93 | 0.5% |
| 4 | 4.56 | 4.47 | **4.45** | 4.45 | 2.4% |
| 5 | 4.28 | 3.81 | **3.72** | **3.72** | **13.1%** |

### Word-Level

| N | MLE | Add-k (k=0.1) | Stupid Backoff | Best | Improvement |
|---|-----|---------------|----------------|------|-------------|
| 1 | 780.10 | **693.07** | 1648.40 | 693.07 | 11.2% |
| 2 | 4216.13 | 788.82 | **475.72** | **475.72** | **88.7%** |
| 3 | 2,053,286 | 4818.64 | **303.40** | 303.40 | 99.99% |
| 4 | 209,123,771 | 12534.81 | **487.60** | 487.60 | 99.99% |
| 5 | 1,348,455,405 | 16853.14 | **966.17** | 966.17 | 99.99% |

## Key Insights

1. **Character >> Word**: 127x better perplexity (3.72 vs 475.72)

2. **Stupid Backoff wins**: Best for both tokenizations at optimal n

3. **Smoothing critical**: 
   - Character: 13% improvement
   - Word: 99.99% improvement (survival-level!)

4. **Word catastrophe**: MLE reaches 1.3 BILLION PPL without smoothing

5. **Optimal hyperparameters**: α=0.7 for Twi (vs literature's 0.4)

## Visualizations

All visualizations available in:
- `results/experiment_1_learning_curves/`
- `results/experiment_2_smoothing/`
- `results/experiment_3_final/`

Main visualization: `results/experiment_3_final/complete_analysis.png`

## Recommendations

**For Twi and similar low-resource languages**:
```
✅ Use character-level tokenization
✅ Use 5-gram models
✅ Use Stupid Backoff (α=0.7)
✅ Expected performance: ~3.7 PPL
```

**Never**:
```
❌ Use MLE (no smoothing) for production
❌ Use word-level for low-resource scenarios
❌ Ignore validation-based hyperparameter tuning
```
