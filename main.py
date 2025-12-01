"""
Main orchestrator using Groq API
"""

import os
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from dataset_loader import DatasetLoader
from evaluation_pipeline import ParallelEvaluationPipeline
from scoring_engine import ScoringEngine
from config import MODELS, RESULTS_FILE

# Load environment variables
load_dotenv()

def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, bool):
        return bool(obj)
    return obj

def main():
    print("\n" + "="*70)
    print(" LLM TRUSTWORTHINESS EVALUATION SYSTEM")
    print("="*70 + "\n")
    
    # Check for Groq API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print(" Error: GROQ_API_KEY not found!")
        return
    
    print(" Groq API key loaded")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Load datasets
    print("\nSTEP 1: Loading Datasets")
    print("-" * 70)
    loader = DatasetLoader()
    datasets = loader.load_all_datasets()
    
    # Step 2: Initialize pipeline
    print("\nSTEP 2: Initializing Groq Evaluation Pipeline")
    print("-" * 70)
    pipeline = ParallelEvaluationPipeline(
        groq_api_key=groq_api_key,
        max_model_workers=3,
        max_prompt_workers=5
    )
    
    # Step 3: Initialize scoring
    print("\nSTEP 3: Initializing Scoring Engine")
    print("-" * 70)
    scorer = ScoringEngine()
    
    # Step 4: Evaluate models
    print("\nSTEP 4: Evaluating Models (PARALLEL MODE)")
    print("-" * 70)
    
    eval_start_time = datetime.now()
    all_raw_results = pipeline.evaluate_all_models_parallel(datasets)
    eval_end_time = datetime.now()
    eval_duration = (eval_end_time - eval_start_time).total_seconds()
    
    # Step 5: Compute scores
    print("\nSTEP 5: Computing Trustworthiness Scores")
    print("-" * 70)
    
    all_results = {
        'timestamp': eval_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'evaluation_duration_seconds': eval_duration,
        'evaluation_mode': 'parallel',
        'api_provider': 'groq',
        'models': []
    }
    
    for raw_result in all_raw_results:
        model_name = raw_result['model']
        print(f"\n Computing scores for {model_name}...")
        
        toxicity_scores = scorer.score_toxicity(
            raw_result['raw_responses']['toxicity']
        )
        
        bias_scores = scorer.score_bias(
            raw_result['raw_responses']['bias']
        )
        
        truthfulness_scores = scorer.score_truthfulness(
            raw_result['raw_responses']['truthfulness']
        )
        
        safety_scores = scorer.score_safety(
            raw_result['raw_responses']['safety'],
            datasets['safety']
        )
        
        dimension_scores = {
            'toxicity': toxicity_scores,
            'bias': bias_scores,
            'truthfulness': truthfulness_scores,
            'safety': safety_scores
        }
        
        overall_score = scorer.compute_overall_score(dimension_scores)
        
        model_results = {
            'model': model_name,
            'timestamp': raw_result['timestamp'],
            'scores': {
                **dimension_scores,
                'overall': overall_score
            }
        }
        
        all_results['models'].append(model_results)
        
        print(f" Results for {model_name}:")
        print(f"   Overall Score: {overall_score:.3f}")
        print(f"   Low Toxicity: {1 - toxicity_scores['mean_toxicity']:.3f}")
        print(f"   Fairness: {1 - bias_scores['bias_score']:.3f}")
        print(f"   Truthfulness: {truthfulness_scores['accuracy']:.3f}")
        print(f"   Safety: {safety_scores['safety_score']:.3f}")
    
    # Step 6: Save results
    print("\n" + "="*70)
    print("STEP 6: Saving Results")
    print("-" * 70)
    
    # Convert all numpy types to native Python types for JSON serialization
    all_results_serializable = convert_to_serializable(all_results)
    
    try:
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results_serializable, f, indent=2)
        
        print(f" Results saved to: {RESULTS_FILE}")
    except Exception as e:
        print(f" Error saving results: {e}")
        print("Trying alternative save method...")
        import codecs
        with codecs.open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results_serializable, f, indent=2, ensure_ascii=False)
        print(f" Results saved to: {RESULTS_FILE} (alternative method)")
    
    # Statistics
    pipeline.print_statistics()
    
    # Final summary
    print("\n" + "="*70)
    print(" EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n Evaluated {len(MODELS)} models")
    print(f" Total time: {eval_duration:.1f}s ({eval_duration/60:.1f} min)")
    print(f" Using Groq's FREE tier - blazing fast!")
    
    print(f"\n Results Summary:")
    print(f"   Overall best model: {all_results_serializable['models'][0]['model']}")
    print(f"   Best overall score: {all_results_serializable['models'][0]['scores']['overall']:.3f}")
    
    print(f" View results in dashboard: streamlit run app.py")
    print("\n")

if __name__ == "__main__":
    main()
