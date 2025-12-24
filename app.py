import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor, ImageDataProcessor
from utils.predictor import EngagementPredictor, ExplanationGenerator
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Student Engagement Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-high {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .prediction-low {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'predictor' not in st.session_state:
    st.session_state.predictor = EngagementPredictor()

if 'processor' not in st.session_state:
    st.session_state.processor = DataProcessor()

# ============================================================================
# TITLE & DESCRIPTION
# ============================================================================

st.title("📚 Student Engagement Prediction System")
st.markdown("""
    Predict student engagement levels based on behavioral and visual data.
    This system uses a hybrid CNN-LSTM model trained on 2000+ students.
""")

# ============================================================================
# SIDEBAR - NAVIGATION
# ============================================================================

with st.sidebar:
    st.header("🎯 Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "📊 Single Prediction", "📈 Batch Prediction", "📖 Documentation", "⚙️ About"]
    )

# ============================================================================
# PAGE 1: HOME
# ============================================================================

if page == "🏠 Home":
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            ### 🎓 System Overview
            
            This application predicts student engagement levels using:
            
            - **Behavioral Data**: Time spent, quiz scores, forum posts, etc.
            - **Visual Data**: Learning analytics dashboards
            - **Deep Learning**: Hybrid CNN-LSTM model
            
            ### 📈 Engagement Levels
            
            - **Low**: Student needs intervention
            - **High**: Student is well-engaged
        """)
    
    with col2:
        st.markdown("""
            ### 🚀 Features
            
            ✅ Real-time predictions  
            ✅ Batch processing  
            ✅ Feature explanations  
            ✅ Visual analytics  
            ✅ Export results  
            
            ### 📊 Model Performance
            
            - Accuracy: >80%
            - Classes: 2 (Low, High)
            - Training Samples: 2000+
        """)
    
    st.markdown("---")
    
    # Statistics cards
    st.subheader("System Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Accuracy", "82%", "+2%")
    
    with col2:
        st.metric("Engagement Classes", "2", "Binary")
    
    with col3:
        st.metric("Input Features", "7", "Behavioral")
    
    with col4:
        st.metric("Sequence Length", "10", "LSTM")

# ============================================================================
# PAGE 2: SINGLE PREDICTION
# ============================================================================

elif page == "📊 Single Prediction":
    
    st.subheader("📝 Single Student Prediction")
    
    st.markdown("""
        Enter student behavioral data below to predict engagement level.
    """)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Learning Behavior")
        
        time_spent = st.slider(
            "Weekly Time Spent (hours)",
            min_value=0,
            max_value=30,
            value=10,
            step=0.5
        )
        
        quiz_score = st.slider(
            "Average Quiz Score (%)",
            min_value=0,
            max_value=100,
            value=75,
            step=1
        )
        
        forum_posts = st.slider(
            "Forum Posts (count)",
            min_value=0,
            max_value=50,
            value=10,
            step=1
        )
    
    with col2:
        st.markdown("### 🎬 Content Engagement")
        
        video_percent = st.slider(
            "Video Watched (%)",
            min_value=0,
            max_value=100,
            value=70,
            step=5
        )
        
        assignments = st.slider(
            "Assignments Submitted (count)",
            min_value=0,
            max_value=20,
            value=8,
            step=1
        )
        
        login_freq = st.slider(
            "Login Frequency (times/week)",
            min_value=0,
            max_value=20,
            value=5,
            step=1
        )
    
    session_duration = st.slider(
        "Average Session Duration (minutes)",
        min_value=0,
        max_value=120,
        value=30,
        step=5
    )
    
    st.markdown("---")
    
    # Prepare data
    input_data = {
        'time_spent_weekly': time_spent,
        'quiz_score_avg': quiz_score,
        'forum_posts': forum_posts,
        'video_watched_percent': video_percent,
        'assignments_submitted': assignments,
        'login_frequency': login_freq,
        'session_duration_avg': session_duration
    }
    
    # Make prediction button
    if st.button("🔮 Predict Engagement", use_container_width=True):
        
        # Validate input
        is_valid, error_msg = st.session_state.processor.validate_input(input_data)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            with st.spinner("Making prediction..."):
                # Process data
                scaled_features = st.session_state.processor.process_behavioral_data(input_data)
                sequence = st.session_state.processor.create_lstm_sequence(scaled_features)
                
                # Make prediction
                prediction = st.session_state.predictor.predict(sequence)
                
                # Display results
                st.markdown("### 🎯 Prediction Results")
                
                # Determine color based on prediction
                if prediction['class'] == 'High':
                    css_class = "prediction-high"
                    emoji = "✅"
                else:
                    css_class = "prediction-low"
                    emoji = "⚠️"
                
                # Display main prediction
                st.markdown(f"""
                    <div class="{css_class}">
                    <h3>{emoji} Prediction: <strong>{prediction['class']} Engagement</strong></h3>
                    <h2>Confidence: {prediction['confidence']:.1%}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display probabilities
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Low Engagement Probability",
                        f"{prediction['probability_low']:.1%}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "High Engagement Probability",
                        f"{prediction['probability_high']:.1%}",
                        delta=None
                    )
                
                # Probability chart
                st.markdown("### 📊 Probability Distribution")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Low', 'High'],
                        y=[prediction['probability_low'], prediction['probability_high']],
                        marker=dict(color=['#dc3545', '#28a745']),
                        text=[f"{prediction['probability_low']:.1%}", 
                              f"{prediction['probability_high']:.1%}"],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    yaxis_title="Probability",
                    xaxis_title="Engagement Level",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate explanation
                st.markdown("### 💡 Feature Analysis")
                
                explanation = ExplanationGenerator.get_explanation(input_data, prediction)
                st.markdown(explanation)
                
                # Feature importance visualization
                st.markdown("### 📈 Feature Values")
                
                feature_df = pd.DataFrame({
                    'Feature': list(input_data.keys()),
                    'Value': list(input_data.values())
                })
                
                fig_features = px.bar(
                    feature_df,
                    x='Feature',
                    y='Value',
                    color='Value',
                    color_continuous_scale='Viridis'
                )
                
                fig_features.update_layout(height=400)
                st.plotly_chart(fig_features, use_container_width=True)

# ============================================================================
# PAGE 3: BATCH PREDICTION
# ============================================================================

elif page == "📈 Batch Prediction":
    
    st.subheader("📊 Batch Prediction")
    
    st.markdown("""
        Upload a CSV file with multiple students to get bulk predictions.
        
        **CSV Format** (required columns):
        - time_spent_weekly
        - quiz_score_avg
        - forum_posts
        - video_watched_percent
        - assignments_submitted
        - login_frequency
        - session_duration_avg
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        st.markdown(f"**File loaded:** {uploaded_file.name}")
        st.markdown(f"**Rows:** {len(df)}")
        st.markdown(f"**Columns:** {', '.join(df.columns)}")
        
        st.markdown("---")
        
        # Preview data
        st.markdown("### 👁️ Data Preview")
        st.dataframe(df.head(10))
        
        st.markdown("---")
        
        # Make predictions
        if st.button("🔮 Predict All", use_container_width=True):
            
            with st.spinner("Processing... This may take a moment"):
                
                try:
                    predictions = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in df.iterrows():
                        # Convert row to dict
                        input_data = row.to_dict()
                        
                        # Validate
                        is_valid, _ = st.session_state.processor.validate_input(input_data)
                        
                        if is_valid:
                            # Process
                            scaled = st.session_state.processor.process_behavioral_data(input_data)
                            seq = st.session_state.processor.create_lstm_sequence(scaled)
                            
                            # Predict
                            pred = st.session_state.predictor.predict(seq)
                            predictions.append(pred)
                        else:
                            predictions.append({'class': 'Error', 'confidence': 0})
                        
                        # Update progress
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {idx + 1}/{len(df)}")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Create results DataFrame
                    results_df = df.copy()
                    results_df['predicted_class'] = [p['class'] for p in predictions]
                    results_df['confidence'] = [p['confidence'] for p in predictions]
                    results_df['prob_low'] = [p['probability_low'] for p in predictions]
                    results_df['prob_high'] = [p['probability_high'] for p in predictions]
                    
                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(results_df)
                    
                    # Statistics
                    st.markdown("### 📈 Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    high_count = (results_df['predicted_class'] == 'High').sum()
                    low_count = (results_df['predicted_class'] == 'Low').sum()
                    
                    with col1:
                        st.metric("Total Students", len(results_df))
                    
                    with col2:
                        st.metric("High Engagement", high_count, f"{high_count/len(results_df)*100:.1f}%")
                    
                    with col3:
                        st.metric("Low Engagement", low_count, f"{low_count/len(results_df)*100:.1f}%")
                    
                    with col4:
                        st.metric("Avg Confidence", f"{results_df['confidence'].mean():.1%}")
                    
                    # Charts
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Class distribution
                        class_counts = results_df['predicted_class'].value_counts()
                        fig_class = px.pie(
                            values=class_counts.values,
                            names=class_counts.index,
                            title="Engagement Distribution",
                            color_discrete_map={'High': '#28a745', 'Low': '#dc3545'}
                        )
                        st.plotly_chart(fig_class, use_container_width=True)
                    
                    with col2:
                        # Confidence distribution
                        fig_conf = px.histogram(
                            results_df,
                            x='confidence',
                            nbins=20,
                            title="Confidence Distribution",
                            color_discrete_sequence=['#007bff']
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # Download results
                    st.markdown("---")
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PAGE 4: DOCUMENTATION
# ============================================================================

elif page == "📖 Documentation":
    
    st.subheader("📚 Documentation")
    
    st.markdown("""
        ## Model Architecture
        
        The system uses a hybrid CNN-LSTM model with 7 behavioral input features.
        
        ## Features Explained
        
        | Feature | Range | Description |
        |---------|-------|-------------|
        | Time Spent Weekly | 0-30 hrs | Hours spent on learning |
        | Quiz Score Avg | 0-100% | Average quiz performance |
        | Forum Posts | 0-50 | Forum contributions |
        | Video Watched | 0-100% | Video completion % |
        | Assignments | 0-20 | Submitted assignments |
        | Login Frequency | 0-20/wk | Logins per week |
        | Session Duration | 0-120 min | Avg session length |
        
        ## API Usage
        
        ```python
        from utils.predictor import EngagementPredictor
        from utils.data_processor import DataProcessor
        
        predictor = EngagementPredictor()
        processor = DataProcessor()
        
        data = {
            'time_spent_weekly': 10,
            'quiz_score_avg': 75,
            'forum_posts': 5,
            'video_watched_percent': 70,
            'assignments_submitted': 8,
            'login_frequency': 5,
            'session_duration_avg': 30
        }
        
        features = processor.process_behavioral_data(data)
        sequence = processor.create_lstm_sequence(features)
        prediction = predictor.predict(sequence)
        ```
    """)

# ============================================================================
# PAGE 5: ABOUT
# ============================================================================

elif page == "⚙️ About":
    
    st.subheader("ℹ️ About This System")
    
    st.markdown("""
        ## Project Information
        
        **Name:** Student Engagement Prediction System  
        **Version:** 1.0.0  
        **Status:** Production-Ready  
        
        ## Technology Stack
        
        - **Deep Learning:** TensorFlow/Keras
        - **Data Processing:** Pandas, NumPy, Scikit-learn
        - **Web Framework:** Streamlit
        - **Deployment:** Streamlit Cloud
        - **Version Control:** Git/GitHub
        
        ## Model Details
        
        - **Architecture:** Hybrid CNN-LSTM
        - **Training Data:** 2000+ students
        - **Classes:** 2 (Low, High Engagement)
        - **Accuracy:** >80%
        - **Features:** 7 behavioral inputs
        
        ## Team
        
        Developed by Data Science Team  
        December 2025
        
        ## Support
        
        For questions or issues, please contact or open GitHub issue.
        
        ## License
        
        MIT License
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
    Student Engagement Prediction System | Version 1.0 | Powered by Streamlit
    </div>
""", unsafe_allow_html=True)
