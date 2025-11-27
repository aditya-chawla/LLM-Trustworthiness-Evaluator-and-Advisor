# LLM Trustworthiness Advisor

Fast, parallel evaluation system for comparing language models on safety, bias, truthfulness, and toxicity.

## Quick Start

### 1. Clone Repository
```powershell
git clone https://github.com/YOUR_USERNAME/llm-trustworthiness-advisor.git
cd llm-trustworthiness-advisor
```

### 2. Set Up Python Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Groq API Key (Environment Variable)
```powershell
# Option A: Set for current session only
$env:GROQ_API_KEY="your_api_key_here"

# Option B: Set permanently in Windows
[System.Environment]::SetEnvironmentVariable('GROQ_API_KEY', 'your_api_key_here', 'User')

# Then restart your terminal
```

**Get free API key**: https://console.groq.com/

### 4. Run Evaluation
```powershell
python main.py
```
Runtime: ~4-5 minutes

### 5. View Dashboard
```powershell
streamlit run app.py
```
Open browser to: http://localhost:8501

## Features
- ✅ 3 models evaluated in parallel (5x speedup)
- ✅ 4 dimensions: toxicity, bias, truthfulness, safety
- ✅ Interactive Streamlit dashboard
- ✅ AI-powered recommendations
- ✅ Free Groq API inference

## Project Structure

```
llm-trustworthiness-advisor/
├── main.py                    # Main orchestrator
├── config.py                  # Configuration
├── dataset_loader.py          # Dataset handling
├── evaluation_pipeline.py      # Parallel evaluation engine
├── scoring_engine.py          # Trustworthiness scoring
├── recommender.py             # AI recommendation system
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── results/                   # Evaluation results (generated)
```

## Models Evaluated

- **llama-3.3-70b-versatile** (Meta): Large, general-purpose
- **gemma-7b-it** (Google): Efficient, instruction-tuned
- **llama-3.1-8b-instant** (Meta): Small, fast

## Evaluation Dimensions

- **Toxicity**: Measures harmful, offensive language
- **Bias**: Detects stereotypical vs fair responses
- **Truthfulness**: Evaluates accuracy on factual questions
- **Safety**: Assesses refusal of harmful requests

## Performance

- **Runtime**: 4-5 minutes (3 models, 80 prompts total)
- **Parallelism**: 3 models × 5 prompts per batch
- **API Calls**: ~520 total
- **Speedup**: 5x faster than sequential evaluation

## Architecture

### Parallel Processing
- All 3 models evaluated simultaneously using ThreadPoolExecutor
- Batched prompt processing within each model
- Smart rate limiting with exponential backoff

### Recommendation Engine
- Uses Mixtral-8x7b (independent model) for unbiased recommendations
- Prevents self-preference bias
- Based purely on performance metrics

## Usage

### Custom Use Case Recommendation
1. Open dashboard: `streamlit run app.py`
2. Go to "Model Recommender" tab
3. Describe your use case
4. Select priorities (safety, fairness, etc.)
5. Get AI-powered recommendation

### Compare Dimensions
1. Go to "Detailed Metrics" tab
2. Select dimension to analyze
3. View charts and breakdowns
4. See pass/fail status

## Results Example

| Model | Overall | Toxicity | Fairness | Truthfulness | Safety |
|-------|---------|----------|----------|--------------|--------|
| **llama-3.1-8b** | 0.698 | 0.993 | 0.850 | 0.150 | 0.750 |
| **gemma-7b** | 0.517 | 0.999 | 0.050 | 0.300 | 0.650 |
| **llama-3.3-70b** | 0.537 | 1.000 | 0.500 | 0.050 | 0.500 |

## API Provider

### Groq
- Free tier: 14,400 requests/day
- Speed: 300+ tokens/second
- Models: Llama 3.3, Llama 3.1, Gemma, Mixtral

## License

MIT License
