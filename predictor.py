import numpy as np
import pickle
from tensorflow.keras.models import load_model

class EngagementPredictor:
    """Make predictions using trained model"""
    
    def __init__(self, model_path='models/engagement_model.h5'):
        """Load model"""
        self.model = load_model(model_path)
    
    def predict(self, X_behav_seq, X_vis=None):
        """
        Make prediction
        
        Args:
            X_behav_seq: Behavioral sequence (1, seq_len, features)
            X_vis: Visual features (optional)
        
        Returns:
            dict: Prediction results
        """
        # Prepare input
        if X_vis is not None:
            inputs = [X_behav_seq, X_vis]
        else:
            inputs = X_behav_seq
        
        # Make prediction
        predictions = self.model.predict(inputs, verbose=0)
        
        # Get probabilities
        probs = predictions[0]  # [P_Low, P_High]
        
        # Get class
        class_idx = np.argmax(probs)
        class_map = {0: 'Low', 1: 'High'}
        predicted_class = class_map[class_idx]
        
        # Get confidence
        confidence = float(probs[class_idx])
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probability_low': float(probs[0]),
            'probability_high': float(probs[1]),
            'raw_output': probs.tolist()
        }
    
    def predict_batch(self, X_batch):
        """
        Make batch predictions
        
        Args:
            X_batch: Batch of data
        
        Returns:
            list: List of predictions
        """
        predictions = self.model.predict(X_batch, verbose=0)
        
        results = []
        for probs in predictions:
            class_idx = np.argmax(probs)
            class_map = {0: 'Low', 1: 'High'}
            
            results.append({
                'class': class_map[class_idx],
                'confidence': float(probs[class_idx]),
                'probability_low': float(probs[0]),
                'probability_high': float(probs[1])
            })
        
        return results


class ExplanationGenerator:
    """Generate explanations for predictions"""
    
    @staticmethod
    def get_explanation(data_dict, prediction):
        """
        Generate human-readable explanation
        
        Args:
            data_dict: Input features
            prediction: Model prediction
        
        Returns:
            str: Explanation
        """
        class_name = prediction['class']
        confidence = prediction['confidence']
        
        # Generate explanation based on features
        explanation = f"""
**Prediction: {class_name} Engagement**
- Confidence: {confidence:.1%}

**Contributing Factors:**
"""
        
        # Analyze features
        if data_dict['time_spent_weekly'] < 5:
            explanation += "\n- ⚠️ Low weekly time spent (< 5 hours)"
        else:
            explanation += f"\n- ✅ Good time spent ({data_dict['time_spent_weekly']} hours/week)"
        
        if data_dict['quiz_score_avg'] < 70:
            explanation += "\n- ⚠️ Below average quiz scores"
        else:
            explanation += f"\n- ✅ Good quiz performance ({data_dict['quiz_score_avg']:.1f}%)"
        
        if data_dict['forum_posts'] < 5:
            explanation += "\n- ⚠️ Low forum participation"
        else:
            explanation += f"\n- ✅ Active forum participation ({data_dict['forum_posts']} posts)"
        
        if data_dict['video_watched_percent'] < 50:
            explanation += "\n- ⚠️ Low video completion"
        else:
            explanation += f"\n- ✅ Good video engagement ({data_dict['video_watched_percent']:.0f}%)"
        
        if data_dict['assignments_submitted'] < 5:
            explanation += "\n- ⚠️ Low assignment completion"
        else:
            explanation += f"\n- ✅ Good assignment completion ({data_dict['assignments_submitted']} submitted)"
        
        return explanation
