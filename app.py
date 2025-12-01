"""
Streamlit dashboard for LLM evaluation results
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
from recommender import LLMRecommender
from config import RESULTS_FILE

# Page configuration
st.set_page_config(
    page_title="LLM Trustworthiness Advisor",
    page_icon="",
    layout="wide"
)

def load_results():
    """Load evaluation results from JSON file"""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return None

def create_radar_chart(results):
    """Create radar chart comparing models across dimensions"""
    models = results['models']
    
    categories = ['Low Toxicity', 'Fairness', 'Truthfulness', 'Safety']
    
    fig = go.Figure()
    
    for model_data in models:
        model_name = model_data['model'].split('/')[-1]
        scores = model_data['scores']
        
        values = [
            1 - scores['toxicity']['mean_toxicity'],  # Inverse for "low toxicity"
            1 - scores['bias']['bias_score'],  # Inverse for "fairness"
            scores['truthfulness']['accuracy'],
            scores['safety']['safety_score']
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Model Comparison Across Trustworthiness Dimensions",
        height=600
    )
    
    return fig

def create_bar_chart(results, dimension):
    """Create bar chart for specific dimension"""
    models = []
    scores = []
    
    for model_data in results['models']:
        model_name = model_data['model'].split('/')[-1]
        models.append(model_name)
        
        if dimension == 'toxicity':
            score = 1 - model_data['scores']['toxicity']['mean_toxicity']
        elif dimension == 'bias':
            score = 1 - model_data['scores']['bias']['bias_score']
        elif dimension == 'truthfulness':
            score = model_data['scores']['truthfulness']['accuracy']
        elif dimension == 'safety':
            score = model_data['scores']['safety']['safety_score']
        else:
            score = model_data['scores']['overall']
        
        scores.append(score)
    
    df = pd.DataFrame({
        'Model': models,
        'Score': scores
    })
    
    fig = px.bar(df, x='Model', y='Score', 
                 title=f'{dimension.capitalize()} Scores',
                 color='Score',
                 color_continuous_scale='Viridis')
    fig.update_layout(yaxis_range=[0, 1])
    
    return fig

def create_detailed_table(results):
    """Create detailed comparison table"""
    data = []
    
    for model_data in results['models']:
        model_name = model_data['model'].split('/')[-1]
        scores = model_data['scores']
        
        data.append({
            'Model': model_name,
            'Overall': f"{scores['overall']:.3f}",
            'Low Toxicity': f"{1 - scores['toxicity']['mean_toxicity']:.3f}",
            'Fairness': f"{1 - scores['bias']['bias_score']:.3f}",
            'Truthfulness': f"{scores['truthfulness']['accuracy']:.3f}",
            'Safety': f"{scores['safety']['safety_score']:.3f}",
            'Status': ' Passed' if scores['overall'] > 0.6 else '  Review'
        })
    
    df = pd.DataFrame(data)
    return df

def main():
    st.title(" LLM Trustworthiness Advisor")
    st.markdown("*Evaluating language models across safety, bias, truthfulness, and toxicity*")
    
    # Load results
    results = load_results()
    
    if results is None:
        st.warning("  No evaluation results found. Please run `python main.py` first.")
        st.info(" This will evaluate models and generate results.")
        return
    
    # Sidebar
    st.sidebar.header(" Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["Overview", "Detailed Metrics", "Model Recommender"]
    )
    
    if page == "Overview":
        st.header(" Model Comparison Overview")
        
        # Radar chart
        st.plotly_chart(create_radar_chart(results), use_container_width=True)
        
        # Summary table
        st.subheader(" Summary Table")
        st.dataframe(create_detailed_table(results), use_container_width=True)
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Evaluated", len(results['models']))
        with col2:
            st.metric("Evaluation Date", results.get('timestamp', 'N/A'))
        with col3:
            best_model = max(results['models'], 
                           key=lambda x: x['scores']['overall'])
            st.metric("Top Performer", 
                     best_model['model'].split('/')[-1])
    
    elif page == "Detailed Metrics":
        st.header(" Detailed Performance Metrics")
        
        # Dimension selector
        dimension = st.selectbox(
            "Select Dimension",
            ["overall", "toxicity", "bias", "truthfulness", "safety"]
        )
        
        # Bar chart for selected dimension
        st.plotly_chart(create_bar_chart(results, dimension), 
                       use_container_width=True)
        
        # Detailed breakdown
        st.subheader("Detailed Breakdown")
        for model_data in results['models']:
            with st.expander(f" {model_data['model'].split('/')[-1]}"):
                scores = model_data['scores']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Overall Score", f"{scores['overall']:.3f}")
                    st.metric("Low Toxicity", 
                             f"{1 - scores['toxicity']['mean_toxicity']:.3f}")
                    st.metric("Fairness", 
                             f"{1 - scores['bias']['bias_score']:.3f}")
                
                with col2:
                    st.metric("Truthfulness", 
                             f"{scores['truthfulness']['accuracy']:.3f}")
                    st.metric("Safety", 
                             f"{scores['safety']['safety_score']:.3f}")
                
                # Pass/Fail status
                st.markdown("**Dimension Status:**")
                for dim in ['toxicity', 'bias', 'truthfulness', 'safety']:
                    passed = scores[dim].get('passed', False)
                    status = " Passed" if passed else " Failed"
                    st.markdown(f"- {dim.capitalize()}: {status}")
    
    else:  # Model Recommender
        st.header(" Model Recommendation System")
        
        st.markdown("""
        Describe your use case and requirements, and get an AI-powered 
        recommendation for the best model.
        """)
        
        # User input
        use_case = st.text_area(
            "Describe your use case:",
            placeholder="e.g., I need a model for customer support chatbot that prioritizes safety and low toxicity...",
            height=100
        )
        
        priority = st.multiselect(
            "Priority dimensions (in order of importance):",
            ["Safety", "Low Toxicity", "Truthfulness", "Fairness"],
            default=["Safety"]
        )
        
        if st.button("Get Recommendation", type="primary"):
            if not use_case:
                st.warning("Please describe your use case first.")
            else:
                with st.spinner(" Analyzing models and generating recommendation..."):
                    try:
                        # Get Groq API key
                        groq_api_key = os.getenv('GROQ_API_KEY')
                        if not groq_api_key:
                            st.error("GROQ_API_KEY not found in environment variables.")
                            return
                        
                        # Generate recommendation
                        recommender = LLMRecommender(groq_api_key=groq_api_key)
                        user_requirements = f"Use case: {use_case}\nPriorities: {', '.join(priority)}"
                        
                        recommendation = recommender.generate_recommendation(
                            results, user_requirements
                        )
                        
                        st.success(" Recommendation Generated!")
                        st.markdown("###  Recommendation")
                        st.markdown(recommendation)
                        
                    except Exception as e:
                        st.error(f"Error generating recommendation: {str(e)}")
        
        # Quick recommendations
        st.markdown("---")
        st.subheader(" Quick Recommendations")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Best Safety"):
                try:
                    groq_api_key = os.getenv('GROQ_API_KEY')
                    recommender = LLMRecommender(groq_api_key=groq_api_key)
                    best = recommender.get_best_model_by_criteria(results, 'safety')
                    st.info(f"**{best['model'].split('/')[-1]}**\n\n{best['metric']}: {best['value']:.3f}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("Best Accuracy"):
                try:
                    groq_api_key = os.getenv('GROQ_API_KEY')
                    recommender = LLMRecommender(groq_api_key=groq_api_key)
                    best = recommender.get_best_model_by_criteria(results, 'accuracy')
                    st.info(f"**{best['model'].split('/')[-1]}**\n\n{best['metric']}: {best['value']:.3f}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col3:
            if st.button("Best Fairness"):
                try:
                    groq_api_key = os.getenv('GROQ_API_KEY')
                    recommender = LLMRecommender(groq_api_key=groq_api_key)
                    best = recommender.get_best_model_by_criteria(results, 'fairness')
                    st.info(f"**{best['model'].split('/')[-1]}**\n\n{best['metric']}: {best['value']:.3f}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col4:
            if st.button("Best Overall"):
                try:
                    groq_api_key = os.getenv('GROQ_API_KEY')
                    recommender = LLMRecommender(groq_api_key=groq_api_key)
                    best = recommender.get_best_model_by_criteria(results, 'overall')
                    st.info(f"**{best['model'].split('/')[-1]}**\n\n{best['metric']}: {best['value']:.3f}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Model comparison
        st.markdown("---")
        st.subheader(" Full Model Comparison")
        try:
            groq_api_key = os.getenv('GROQ_API_KEY')
            recommender = LLMRecommender(groq_api_key=groq_api_key)
            comparison = recommender.get_model_comparison(results)
            st.markdown(comparison)
        except Exception as e:
            st.warning(f"Could not load comparison: {str(e)}")

if __name__ == "__main__":
    main()
