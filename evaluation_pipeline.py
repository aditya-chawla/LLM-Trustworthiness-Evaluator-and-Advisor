"""
Parallel evaluation pipeline using Groq API (FREE & FAST)
"""

import time
import requests
from openai import OpenAI  # Groq uses OpenAI SDK
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import API_CONFIG, MODELS
import os
from threading import Lock

class ParallelEvaluationPipeline:
    def __init__(self, groq_api_key: str, max_model_workers=3, max_prompt_workers=5):
        """
        Initialize with Groq API
        
        Args:
            groq_api_key: Groq API key
            max_model_workers: Models to evaluate in parallel
            max_prompt_workers: Prompts to process concurrently (Groq is fast!)
        """
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required!")
        
        # Initialize Groq client (OpenAI-compatible)
        self.client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
        self.max_model_workers = max_model_workers
        self.max_prompt_workers = max_prompt_workers
        
        # Thread-safe call tracker
        self.call_tracker = {
            'total_calls': 0,
            'successful': 0,
            'failed': 0,
            'rate_limited': 0
        }
        self.tracker_lock = Lock()
        
        print(f" Using Groq API")
        print(f"   Models available: {len(MODELS)}")
    
    def _update_tracker(self, key: str, increment: int = 1):
        """Thread-safe tracker update"""
        with self.tracker_lock:
            self.call_tracker[key] += increment
    
    def query_model(self, model_name: str, prompt: str, max_retries=3) -> str:
        """
        Query Groq model with retry logic
        """
        for attempt in range(max_retries):
            try:
                self._update_tracker('total_calls')
                
                # Call Groq API (OpenAI-compatible)
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=API_CONFIG['max_new_tokens'],
                    temperature=API_CONFIG['temperature']
                )
                
                self._update_tracker('successful')
                time.sleep(API_CONFIG['rate_limit_delay'])
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limits
                if '429' in error_str or 'rate_limit' in error_str.lower():
                    self._update_tracker('rate_limited')
                    wait_time = (2 ** attempt) * API_CONFIG['retry_delay']
                    print(f"Rate limit hit for {model_name}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                # Handle other errors
                else:
                    self._update_tracker('failed')
                    if attempt == max_retries - 1:
                        print(f"Error for {model_name}: {error_str[:100]}")
                        return None
                    time.sleep(API_CONFIG['retry_delay'])
        
        return None
    
    def evaluate_toxicity_parallel(self, model_name: str, prompts: List[str]) -> List[str]:
        """Evaluate toxicity with parallel batches"""
        print(f" Testing {model_name} - Toxicity (parallel batches)...")
        
        all_responses = []
        batch_size = self.max_prompt_workers
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_responses = [None] * len(batch)
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_idx = {
                    executor.submit(
                        self.query_model,
                        model_name,
                        f"Continue this sentence naturally: {prompt}"
                    ): idx for idx, prompt in enumerate(batch)
                }
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        response = future.result()
                        batch_responses[idx] = response if response else ""
                    except Exception as e:
                        batch_responses[idx] = ""
            
            all_responses.extend(batch_responses)
            progress = min(i + batch_size, len(prompts))
            print(f"  Progress: {progress}/{len(prompts)} prompts")
            
            if i + batch_size < len(prompts):
                time.sleep(0.5)
        
        return all_responses
    
    def evaluate_bias_parallel(self, model_name: str, bias_pairs: List[Dict]) -> List[Dict]:
        """Evaluate bias with parallel processing"""
        print(f" Testing {model_name} - Bias (parallel batches)...")
        
        all_responses = []
        batch_size = self.max_prompt_workers // 2
        
        for i in range(0, len(bias_pairs), batch_size):
            batch = bias_pairs[i:i+batch_size]
            batch_responses = []
            
            with ThreadPoolExecutor(max_workers=batch_size * 2) as executor:
                futures = []
                
                for pair in batch:
                    stereo_future = executor.submit(
                        self.query_model,
                        model_name,
                        f"Complete this sentence: {pair['stereotypical']}"
                    )
                    anti_future = executor.submit(
                        self.query_model,
                        model_name,
                        f"Complete this sentence: {pair['anti_stereotypical']}"
                    )
                    futures.append((stereo_future, anti_future))
                
                for stereo_future, anti_future in futures:
                    stereo_response = stereo_future.result()
                    anti_response = anti_future.result()
                    
                    batch_responses.append({
                        'stereotypical': stereo_response if stereo_response else "",
                        'anti_stereotypical': anti_response if anti_response else ""
                    })
            
            all_responses.extend(batch_responses)
            progress = min(i + batch_size, len(bias_pairs))
            print(f"  Progress: {progress}/{len(bias_pairs)} pairs")
            
            if i + batch_size < len(bias_pairs):
                time.sleep(0.5)
        
        return all_responses
    
    def evaluate_truthfulness_parallel(self, model_name: str, qa_pairs: List[Dict]) -> List[Dict]:
        """Evaluate truthfulness with parallel processing"""
        print(f" Testing {model_name} - Truthfulness (parallel batches)...")
        
        all_responses = []
        batch_size = self.max_prompt_workers
        
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i+batch_size]
            batch_responses = [None] * len(batch)
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_idx = {
                    executor.submit(
                        self.query_model,
                        model_name,
                        f"Answer this question truthfully and concisely: {qa['question']}"
                    ): (idx, qa) for idx, qa in enumerate(batch)
                }
                
                for future in as_completed(future_to_idx):
                    idx, qa = future_to_idx[future]
                    try:
                        response = future.result()
                        batch_responses[idx] = {
                            'question': qa['question'],
                            'response': response if response else "",
                            'correct_answers': qa['correct_answers']
                        }
                    except Exception as e:
                        batch_responses[idx] = {
                            'question': qa['question'],
                            'response': "",
                            'correct_answers': qa['correct_answers']
                        }
            
            all_responses.extend(batch_responses)
            progress = min(i + batch_size, len(qa_pairs))
            print(f"  Progress: {progress}/{len(qa_pairs)} questions")
            
            if i + batch_size < len(qa_pairs):
                time.sleep(0.5)
        
        return all_responses
    
    def evaluate_safety_parallel(self, model_name: str, prompts: List[str]) -> List[str]:
        """Evaluate safety with parallel processing"""
        print(f" Testing {model_name} - Safety (parallel batches)...")
        
        all_responses = []
        batch_size = self.max_prompt_workers
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_responses = [None] * len(batch)
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_idx = {
                    executor.submit(
                        self.query_model,
                        model_name,
                        prompt
                    ): idx for idx, prompt in enumerate(batch)
                }
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        response = future.result()
                        batch_responses[idx] = response if response else ""
                    except Exception as e:
                        batch_responses[idx] = ""
            
            all_responses.extend(batch_responses)
            progress = min(i + batch_size, len(prompts))
            print(f"  Progress: {progress}/{len(prompts)} prompts")
            
            if i + batch_size < len(prompts):
                time.sleep(0.5)
        
        return all_responses
    
    def evaluate_model_full(self, model_name: str, datasets: Dict) -> Dict:
        """Run full evaluation on a single model"""
        print(f"\n{'='*60}")
        print(f" EVALUATING: {model_name}")
        print(f"{'='*60}")
        
        model_start_time = time.time()
        
        results = {
            'model': model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'raw_responses': {}
        }
        
        results['raw_responses']['toxicity'] = self.evaluate_toxicity_parallel(
            model_name, datasets['toxicity']
        )
        
        results['raw_responses']['bias'] = self.evaluate_bias_parallel(
            model_name, datasets['bias']
        )
        
        results['raw_responses']['truthfulness'] = self.evaluate_truthfulness_parallel(
            model_name, datasets['truthfulness']
        )
        
        results['raw_responses']['safety'] = self.evaluate_safety_parallel(
            model_name, datasets['safety']
        )
        
        model_time = time.time() - model_start_time
        print(f"\n Completed {model_name} in {model_time:.1f} seconds")
        
        return results
    
    def evaluate_all_models_parallel(self, datasets: Dict) -> List[Dict]:
        """Evaluate all models in parallel"""
        print(f"\n{'='*60}")
        print(" GROQ PARALLEL EVALUATION MODE")
        print(f"   - Models in parallel: {self.max_model_workers}")
        print(f"   - Prompts per batch: {self.max_prompt_workers}")
        print(f"   - Total models: {len(MODELS)}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_model_workers) as executor:
            future_to_model = {
                executor.submit(
                    self.evaluate_model_full,
                    model,
                    datasets
                ): model for model in MODELS
            }
            
            completed = 0
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    completed += 1
                    elapsed = time.time() - start_time
                    
                    print(f"\n{'='*60}")
                    print(f" [{completed}/{len(MODELS)}] COMPLETED: {model}")
                    print(f"   Elapsed time: {elapsed:.1f}s")
                    if completed < len(MODELS):
                        print(f"   Estimated remaining: {(elapsed/completed)*(len(MODELS)-completed):.1f}s")
                    print(f"{'='*60}\n")
                    
                except Exception as e:
                    print(f"\n Error evaluating {model}: {str(e)}\n")
        
        total_time = time.time() - start_time
        speedup = (20 * 60) / total_time
        
        print(f"\n{'='*60}")
        print(f" ALL MODELS EVALUATED!")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Speedup: {speedup:.1f}x faster than sequential!")
        print(f"{'='*60}\n")
        
        return all_results
    
    def print_statistics(self):
        """Print API call statistics"""
        print(f"\n{'='*60}")
        print(" API CALL STATISTICS")
        print(f"{'='*60}")
        print(f"Total API calls: {self.call_tracker['total_calls']}")
        print(f"Successful: {self.call_tracker['successful']}")
        print(f"Failed: {self.call_tracker['failed']}")
        print(f"Rate limited: {self.call_tracker['rate_limited']}")
        
        if self.call_tracker['total_calls'] > 0:
            success_rate = (self.call_tracker['successful'] / 
                          self.call_tracker['total_calls']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print(f"{'='*60}\n")


# Backward compatibility
class EvaluationPipeline(ParallelEvaluationPipeline):
    """Alias for backward compatibility"""
    pass
