"""
LLM-based recommender system for model selection using Groq API
"""

from openai import OpenAI
from typing import Dict, List
import os

class LLMRecommender:
    def __init__(self, groq_api_key: str = None):
        """
        Initialize recommender with Groq API
        
        Args:
            groq_api_key: Groq API key (or loaded from env if not provided)
        """
        if not groq_api_key:
            groq_api_key = os.getenv('GROQ_API_KEY')
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required!")
        
        # Initialize Groq client (OpenAI-compatible)
        self.client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Use Groq's Llama model for recommendations
        self.recommender_model = "llama-3.3-70b-versatile"
    
    def format_evaluation_results(self, results: Dict) -> str:
        """Format evaluation results for LLM input"""
        formatted = "# Model Evaluation Results\n\n"
        
        for model_data in results['models']:
            model_name = model_data['model'].split('/')[-1]
            scores = model_data['scores']
            
            formatted += f"## {model_name}\n"
            formatted += f"- Overall Score: {scores['overall']:.2f}/1.00\n"
            formatted += f"- Toxicity (lower is better): {scores['toxicity']['mean_toxicity']:.3f}\n"
            formatted += f"- Bias Fairness (lower bias is better): {scores['bias']['bias_score']:.3f}\n"
            formatted += f"- Truthfulness (accuracy): {scores['truthfulness']['accuracy']:.3f}\n"
            formatted += f"- Safety Score: {scores['safety']['safety_score']:.3f}\n\n"
        
        return formatted
    
    def generate_recommendation(self, results: Dict, user_requirements: str) -> str:
        """
        Generate LLM-based recommendation using Groq API
        
        Args:
            results: Evaluation results dictionary
            user_requirements: User's use case and requirements
            
        Returns:
            Recommendation text
        """
        evaluation_summary = self.format_evaluation_results(results)
        
        prompt = f"""You are an AI model advisor helping users select the best language model for their needs.

{evaluation_summary}

User Requirements:
{user_requirements}

Based on the evaluation results above and the user's requirements, provide a clear recommendation:

1. Which model you recommend and why
2. The key strengths of this model for their use case
3. Any potential limitations or trade-offs
4. Alternative model if their needs change

Keep your response concise and practical (3-4 paragraphs)."""

        try:
            response = self.client.chat.completions.create(
                model=self.recommender_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating recommendation: {str(e)}"
    
    def get_best_model_by_criteria(self, results: Dict, criteria: str) -> Dict:
        """
        Get best model based on specific criteria
        
        Args:
            results: Evaluation results
            criteria: 'safety', 'accuracy', 'fairness', 'toxicity', or 'overall'
            
        Returns:
            Dict with best model and metric info
        """
        models = results['models']
        
        if criteria == 'safety':
            best = max(models, key=lambda x: x['scores']['safety']['safety_score'])
            metric = 'Safety Score'
            value = best['scores']['safety']['safety_score']
            
        elif criteria == 'accuracy':
            best = max(models, key=lambda x: x['scores']['truthfulness']['accuracy'])
            metric = 'Accuracy'
            value = best['scores']['truthfulness']['accuracy']
            
        elif criteria == 'fairness':
            best = min(models, key=lambda x: x['scores']['bias']['bias_score'])
            metric = 'Fairness (Low Bias)'
            value = 1 - best['scores']['bias']['bias_score']
            
        elif criteria == 'toxicity':
            best = min(models, key=lambda x: x['scores']['toxicity']['mean_toxicity'])
            metric = 'Low Toxicity'
            value = 1 - best['scores']['toxicity']['mean_toxicity']
            
        else:  # overall
            best = max(models, key=lambda x: x['scores']['overall'])
            metric = 'Overall Score'
            value = best['scores']['overall']
        
        return {
            'model': best['model'],
            'metric': metric,
            'value': float(value),
            'full_scores': best['scores']
        }
    
    def get_model_comparison(self, results: Dict) -> str:
        """Generate a comparison summary of all models"""
        comparison = "# Model Comparison Summary\n\n"
        
        for model_data in results['models']:
            model_name = model_data['model'].split('/')[-1]
            scores = model_data['scores']
            
            comparison += f"## {model_name}\n"
            comparison += f"- **Overall**: {scores['overall']:.2f}/1.00\n"
            comparison += f"- **Safety**: {scores['safety']['safety_score']:.2f}\n"
            comparison += f"- **Toxicity**: {1 - scores['toxicity']['mean_toxicity']:.2f} (lower is worse)\n"
            comparison += f"- **Fairness**: {1 - scores['bias']['bias_score']:.2f}\n"
            comparison += f"- **Accuracy**: {scores['truthfulness']['accuracy']:.2f}\n\n"
        
        return comparison
