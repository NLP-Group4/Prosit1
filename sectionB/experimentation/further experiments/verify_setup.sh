#!/bin/bash

# Quick verification script to test folder organization

echo "======================================"
echo "N-Gram Twi - Folder Verification"
echo "======================================"

# Check structure
echo ""
echo "1. Checking folder structure..."
if [ -d "data" ] && [ -d "scripts" ] && [ -d "results" ]; then
    echo "   ✅ All main directories present"
else
    echo "   ❌ Missing directories"
    exit 1
fi

# Check data file
echo ""
echo "2. Checking data file..."
if [ -f "data/all_twi.txt" ]; then
    lines=$(wc -l < data/all_twi.txt)
    echo "   ✅ Data file present ($lines lines)"
else
    echo "   ❌ Data file missing"
    exit 1
fi

# Check scripts
echo ""
echo "3. Checking experiment scripts..."
for script in learning_curve_experiment smoothing_experiment final_analysis; do
    if [ -f "scripts/${script}.py" ]; then
        echo "   ✅ ${script}.py"
    else
        echo "   ❌ ${script}.py missing"
        exit 1
    fi
done

# Check documentation
echo ""
echo "4. Checking documentation..."
for doc in README.md RESULTS_SUMMARY.md .gitignore; do
    if [ -f "$doc" ]; then
        echo "   ✅ $doc"
    else
        echo "   ❌ $doc missing"
    fi
done

# Check results directories
echo ""
echo "5. Checking results directories..."
for exp in experiment_1_learning_curves experiment_2_smoothing experiment_3_final; do
    if [ -d "results/$exp" ]; then
        count=$(ls -1 results/$exp | wc -l)
        echo "   ✅ $exp ($count files)"
    else
        echo "   ⚠️  $exp (empty or missing)"
    fi
done

echo ""
echo "======================================"
echo "✅ Folder organization verified!"
echo "======================================"
echo ""
echo "Quick Start:"
echo "  python scripts/final_analysis.py"
echo ""
echo "Full experiments:"
echo "  1. python scripts/learning_curve_experiment.py  (~90 min)"
echo "  2. python scripts/smoothing_experiment.py       (~10 min)"
echo "  3. python scripts/final_analysis.py             (~15 min)"
