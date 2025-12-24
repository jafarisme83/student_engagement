import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    """Process input data for model prediction"""
    
    def __init__(self, scaler_path='models/scaler.pkl'):
        """Load scaler"""
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
    def process_behavioral_data(self, data_dict):
        """
        Process behavioral data input
        
        Args:
            data_dict: Dictionary with keys like:
                - time_spent_weekly
                - quiz_score_avg
                - forum_posts
                - video_watched_percent
                - assignments_submitted
                - login_frequency
                - session_duration_avg
        
        Returns:
            np.array: Scaled features
        """
        # Create DataFrame from input
        df = pd.DataFrame([data_dict])
        
        # List of features (must match training data)
        features = [
            'time_spent_weekly',
            'quiz_score_avg',
            'forum_posts',
            'video_watched_percent',
            'assignments_submitted',
            'login_frequency',
            'session_duration_avg'
        ]
        
        # Select features
        df = df[features]
        
        # Scale
        scaled = self.scaler.transform(df)
        
        return scaled[0]
    
    def create_lstm_sequence(self, features, sequence_length=10):
        """
        Create LSTM sequence from features
        
        Args:
            features: Scaled features (1D array)
            sequence_length: Length of sequence
        
        Returns:
            np.array: Shaped for LSTM (1, sequence_length, n_features)
        """
        # Repeat features to create sequence
        sequence = np.tile(features, (sequence_length, 1))
        
        # Reshape for LSTM
        sequence = sequence.reshape(1, sequence_length, len(features))
        
        return sequence
    
    def validate_input(self, data_dict):
        """
        Validate input data
        
        Args:
            data_dict: Input data
        
        Returns:
            tuple: (is_valid, error_message)
        """
        required_keys = [
            'time_spent_weekly',
            'quiz_score_avg',
            'forum_posts',
            'video_watched_percent',
            'assignments_submitted',
            'login_frequency',
            'session_duration_avg'
        ]
        
        # Check required keys
        missing = [k for k in required_keys if k not in data_dict]
        if missing:
            return False, f"Missing fields: {missing}"
        
        # Check value ranges
        for key, value in data_dict.items():
            try:
                float(value)
            except:
                return False, f"{key} must be numeric"
            
            if value < 0:
                return False, f"{key} must be non-negative"
        
        return True, "Data valid"


class ImageDataProcessor:
    """Process image data for model prediction"""
    
    @staticmethod
    def load_and_preprocess_image(image_path, img_size=(224, 224)):
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image
            img_size: Target size
        
        Returns:
            np.array: Preprocessed image
        """
        from tensorflow.keras.preprocessing import image
        
        # Load image
        img = image.load_img(image_path, target_size=img_size)
        
        # Convert to array
        img_array = image.img_to_array(img)
        
        # Normalize
        img_array = img_array / 255.0
        
        return img_array
