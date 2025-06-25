import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core libraries for anomaly detection
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from scipy.stats import chi2
from scipy.signal import savgol_filter

# Install these if needed:
# pip install adtk pyod anomalib

try:
    from adtk.data import validate_series
    from adtk.detector import *
    from adtk.visualization import plot
    ADTK_AVAILABLE = True
except ImportError:
    print("ADTK not available. Install with: pip install adtk")
    ADTK_AVAILABLE = False

try:
    from pyod.models.knn import KNN
    from pyod.models.iforest import IForest
    from pyod.models.ocsvm import OCSVM
    PYOD_AVAILABLE = True
except ImportError:
    print("PyOD not available. Install with: pip install pyod")
    PYOD_AVAILABLE = False

class WWTPAnomalyDetector:
    """
    Comprehensive anomaly detection toolkit for wastewater treatment plant sensor data.
    
    Designed for typical WWTP parameters:
    - Flow rates (influent, effluent, RAS, WAS)
    - Water quality (pH, DO, turbidity, conductivity)
    - Biological parameters (MLSS, MLVSS, SVI)
    - Chemical parameters (BOD, COD, NH4-N, NO3-N, PO4-P)
    - Operational parameters (pump speeds, valve positions)
    """
    
    def __init__(self, data, timestamp_col='timestamp', contamination=0.1):
        """
        Initialize the anomaly detector.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Sensor data with timestamp and measurement columns
        timestamp_col : str
            Name of timestamp column
        contamination : float
            Expected proportion of anomalies (0.05-0.15 typical for WWTP)
        """
        self.data = data.copy()
        self.timestamp_col = timestamp_col
        self.contamination = contamination
        self.scalers = {}
        self.detectors = {}
        self.results = {}
        
        # Ensure timestamp is datetime
        if timestamp_col in self.data.columns:
            self.data[timestamp_col] = pd.to_datetime(self.data[timestamp_col])
            self.data = self.data.set_index(timestamp_col)
        
        # Identify numeric columns (sensor measurements)
        self.sensor_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
    def preprocess_data(self, method='robust', remove_outliers=True, smooth=False):
        """
        Preprocess sensor data for anomaly detection.
        
        Parameters:
        -----------
        method : str
            Scaling method ('standard', 'robust', 'minmax')
        remove_outliers : bool
            Remove extreme outliers before scaling
        smooth : bool
            Apply Savitzky-Golay smoothing filter
        """
        print("Preprocessing sensor data...")
        
        processed_data = self.data[self.sensor_cols].copy()
        
        # Handle missing values
        processed_data = processed_data.interpolate(method='linear', limit=3)
        processed_data = processed_data.fillna(processed_data.median())
        
        # Remove extreme outliers (beyond 4 sigma)
        if remove_outliers:
            for col in processed_data.columns:
                Q1 = processed_data[col].quantile(0.25)
                Q3 = processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                processed_data[col] = processed_data[col].clip(lower_bound, upper_bound)
        
        # Apply smoothing filter (useful for noisy sensor data)
        if smooth:
            for col in processed_data.columns:
                if len(processed_data) > 10:
                    window_length = min(11, len(processed_data) // 10 * 2 + 1)
                    processed_data[col] = savgol_filter(processed_data[col], 
                                                      window_length, 3, mode='nearest')
        
        # Scale data
        if method == 'robust':
            scaler = RobustScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        self.scalers['main'] = scaler
        scaled_data = scaler.fit_transform(processed_data)
        self.processed_data = pd.DataFrame(scaled_data, 
                                         columns=processed_data.columns,
                                         index=processed_data.index)
        
        return self.processed_data
    
    def detect_mahalanobis_anomalies(self, threshold_percentile=95):
        """
        Mahalanobis distance anomaly detection (like in your LinkedIn chart).
        Excellent for multivariate sensor data where parameters are correlated.
        """
        print("Detecting anomalies using Mahalanobis distance...")
        
        data = self.processed_data.dropna()
        
        # Calculate covariance matrix
        cov_matrix = np.cov(data.T)
        cov_inv = np.linalg.pinv(cov_matrix)
        
        # Calculate Mahalanobis distance for each point
        mean = data.mean().values
        mahal_distances = []
        
        for i, row in data.iterrows():
            diff = row.values - mean
            mahal_dist = np.sqrt(diff.T @ cov_inv @ diff)
            mahal_distances.append(mahal_dist)
        
        mahal_distances = np.array(mahal_distances)
        
        # Set threshold based on chi-square distribution
        threshold = np.percentile(mahal_distances, threshold_percentile)
        
        # Alternative: Use chi-square critical value
        # dof = len(self.sensor_cols)
        # threshold = np.sqrt(chi2.ppf(threshold_percentile/100, dof))
        
        anomalies = mahal_distances > threshold
        
        self.results['mahalanobis'] = {
            'distances': pd.Series(mahal_distances, index=data.index),
            'threshold': threshold,
            'anomalies': pd.Series(anomalies, index=data.index),
            'anomaly_score': mahal_distances / threshold
        }
        
        print(f"Found {anomalies.sum()} anomalies ({anomalies.sum()/len(anomalies)*100:.1f}%)")
        return self.results['mahalanobis']
    
    def detect_isolation_forest_anomalies(self):
        """
        Isolation Forest - excellent for high-dimensional sensor data.
        Works well with mixed normal/abnormal patterns.
        """
        print("Detecting anomalies using Isolation Forest...")
        
        detector = IsolationForest(contamination=self.contamination, 
                                 random_state=42, n_estimators=200)
        
        data = self.processed_data.dropna()
        predictions = detector.fit_predict(data)
        scores = detector.decision_function(data)
        
        anomalies = predictions == -1
        
        self.results['isolation_forest'] = {
            'predictions': pd.Series(predictions, index=data.index),
            'scores': pd.Series(scores, index=data.index),
            'anomalies': pd.Series(anomalies, index=data.index),
            'detector': detector
        }
        
        print(f"Found {anomalies.sum()} anomalies ({anomalies.sum()/len(anomalies)*100:.1f}%)")
        return self.results['isolation_forest']
    
    def detect_statistical_anomalies(self):
        """
        Statistical anomaly detection for individual parameters.
        Good for detecting sensor malfunctions or process upsets.
        """
        print("Detecting statistical anomalies...")
        
        data = self.processed_data.dropna()
        statistical_anomalies = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            series = data[col]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            z_anomalies = z_scores > 3
            
            # Modified Z-score (more robust)
            median = np.median(series)
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            modified_z_anomalies = np.abs(modified_z_scores) > 3.5
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            iqr_anomalies = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
            
            # Combine methods
            combined_anomalies = z_anomalies | modified_z_anomalies | iqr_anomalies
            statistical_anomalies[f'{col}_anomaly'] = combined_anomalies
        
        # Overall statistical anomaly if any parameter is anomalous
        overall_anomalies = statistical_anomalies.any(axis=1)
        
        self.results['statistical'] = {
            'individual_anomalies': statistical_anomalies,
            'overall_anomalies': overall_anomalies
        }
        
        print(f"Found {overall_anomalies.sum()} statistical anomalies ({overall_anomalies.sum()/len(overall_anomalies)*100:.1f}%)")
        return self.results['statistical']
    
    def detect_time_series_anomalies(self):
        """
        Time series specific anomaly detection using ADTK.
        Detects seasonal anomalies, level shifts, and trends.
        """
        if not ADTK_AVAILABLE:
            print("ADTK not available. Skipping time series anomaly detection.")
            return None
            
        print("Detecting time series anomalies...")
        
        ts_anomalies = {}
        
        for col in self.sensor_cols[:3]:  # Limit to first 3 columns for demo
            try:
                series = validate_series(self.processed_data[col].dropna())
                
                # Level shift detector (sudden changes in mean)
                level_shift_detector = LevelShiftAD(c=6.0, side='both', window=5)
                level_anomalies = level_shift_detector.fit_detect(series)
                
                # Seasonal anomaly detector
                seasonal_detector = SeasonalAD(freq=24)  # Daily seasonality
                seasonal_anomalies = seasonal_detector.fit_detect(series)
                
                # Persist anomaly detector (values that persist too long)
                persist_detector = PersistAD(c=3.0, side='both')
                persist_anomalies = persist_detector.fit_detect(series)
                
                ts_anomalies[col] = {
                    'level_shift': level_anomalies,
                    'seasonal': seasonal_anomalies,
                    'persist': persist_anomalies
                }
                
            except Exception as e:
                print(f"Error processing {col}: {e}")
                continue
        
        self.results['time_series'] = ts_anomalies
        return ts_anomalies
    
    def ensemble_detection(self, methods=['mahalanobis', 'isolation_forest', 'statistical']):
        """
        Combine multiple anomaly detection methods for robust detection.
        """
        print("Running ensemble anomaly detection...")
        
        # Run all specified methods
        if 'mahalanobis' in methods:
            self.detect_mahalanobis_anomalies()
        if 'isolation_forest' in methods:
            self.detect_isolation_forest_anomalies()
        if 'statistical' in methods:
            self.detect_statistical_anomalies()
        if 'time_series' in methods:
            self.detect_time_series_anomalies()
        
        # Combine results
        ensemble_scores = pd.DataFrame(index=self.processed_data.index)
        
        if 'mahalanobis' in self.results:
            ensemble_scores['mahalanobis'] = self.results['mahalanobis']['anomaly_score']
        
        if 'isolation_forest' in self.results:
            # Normalize isolation forest scores
            if_scores = self.results['isolation_forest']['scores']
            normalized_if = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
            ensemble_scores['isolation_forest'] = 1 - normalized_if
        
        if 'statistical' in self.results:
            ensemble_scores['statistical'] = self.results['statistical']['overall_anomalies'].astype(float)
        
        # Calculate ensemble score (average of available methods)
        ensemble_scores = ensemble_scores.fillna(0)
        final_scores = ensemble_scores.mean(axis=1)
        
        # Threshold for ensemble
        threshold = np.percentile(final_scores, 95)
        final_anomalies = final_scores > threshold
        
        self.results['ensemble'] = {
            'scores': final_scores,
            'threshold': threshold,
            'anomalies': final_anomalies,
            'individual_scores': ensemble_scores
        }
        
        print(f"Ensemble found {final_anomalies.sum()} anomalies ({final_anomalies.sum()/len(final_anomalies)*100:.1f}%)")
        return self.results['ensemble']
    
    def plot_anomaly_results(self, method='mahalanobis', figsize=(15, 10)):
        """
        Plot anomaly detection results similar to your LinkedIn chart.
        """
        if method not in self.results:
            print(f"Method {method} not run yet. Running now...")
            if method == 'mahalanobis':
                self.detect_mahalanobis_anomalies()
            elif method == 'isolation_forest':
                self.detect_isolation_forest_anomalies()
            elif method == 'ensemble':
                self.ensemble_detection()
        
        result = self.results[method]
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Original data with anomalies highlighted
        ax1 = axes[0]
        
        # Plot first few sensor parameters
        plot_cols = self.sensor_cols[:3]  # Limit to 3 for visibility
        colors = ['blue', 'orange', 'green']
        
        for i, col in enumerate(plot_cols):
            original_data = self.data[col].reindex(self.processed_data.index)
            ax1.plot(original_data.index, original_data.values, 
                    color=colors[i], alpha=0.7, label=col)
        
        # Highlight anomalies
        anomaly_mask = result['anomalies']
        anomaly_times = anomaly_mask[anomaly_mask].index
        
        for t in anomaly_times:
            ax1.axvspan(t, t, alpha=0.3, color='red', ymin=0, ymax=1)
        
        ax1.set_title(f'Sensor Data with {method.title()} Anomalies')
        ax1.set_ylabel('Sensor Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores
        ax2 = axes[1]
        
        if method == 'mahalanobis':
            distances = result['distances']
            threshold = result['threshold']
            
            ax2.plot(distances.index, distances.values, color='blue', alpha=0.7, label='Mahalanobis Distance')
            ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
            ax2.fill_between(distances.index, 0, distances.values, 
                           where=distances.values > threshold, 
                           color='red', alpha=0.3, label='Anomalies')
            ax2.set_ylabel('Mahalanobis Distance')
            
        elif method == 'isolation_forest':
            scores = result['scores']
            ax2.plot(scores.index, scores.values, color='purple', alpha=0.7, label='Isolation Forest Score')
            anomaly_mask = result['anomalies']
            ax2.scatter(scores[anomaly_mask].index, scores[anomaly_mask].values, 
                       color='red', s=20, label='Anomalies', zorder=5)
            ax2.set_ylabel('Isolation Forest Score')
            
        elif method == 'ensemble':
            scores = result['scores']
            threshold = result['threshold']
            
            ax2.plot(scores.index, scores.values, color='green', alpha=0.7, label='Ensemble Score')
            ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
            ax2.fill_between(scores.index, 0, scores.values, 
                           where=scores.values > threshold, 
                           color='red', alpha=0.3, label='Anomalies')
            ax2.set_ylabel('Ensemble Anomaly Score')
        
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n{method.title()} Anomaly Detection Summary:")
        print(f"Total data points: {len(result['anomalies'])}")
        print(f"Anomalies detected: {result['anomalies'].sum()}")
        print(f"Anomaly rate: {result['anomalies'].sum()/len(result['anomalies'])*100:.2f}%")
    
    def export_results(self, filename='wwtp_anomalies.csv'):
        """
        Export anomaly detection results to CSV for further analysis.
        """
        export_data = self.data.copy()
        
        for method, result in self.results.items():
            if 'anomalies' in result:
                export_data[f'{method}_anomaly'] = result['anomalies'].reindex(export_data.index, fill_value=False)
                
                if 'scores' in result:
                    export_data[f'{method}_score'] = result['scores'].reindex(export_data.index, fill_value=0)
                elif 'distances' in result:
                    export_data[f'{method}_score'] = result['distances'].reindex(export_data.index, fill_value=0)
        
        export_data.to_csv(filename)
        print(f"Results exported to {filename}")


def generate_sample_wwtp_data(days=30):
    """
    Generate sample WWTP sensor data for demonstration.
    Replace this with your actual data loading function.
    """
    np.random.seed(42)
    
    # Time series
    start_date = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start_date, periods=days*24*4, freq='15min')  # 15-min intervals
    
    # Generate realistic WWTP sensor data
    n_points = len(timestamps)
    
    # Add daily seasonality
    daily_pattern = np.sin(2 * np.pi * np.arange(n_points) / (24 * 4)) * 0.3
    
    data = {
        'timestamp': timestamps,
        'influent_flow': 100 + 20 * daily_pattern + np.random.normal(0, 5, n_points),
        'ph': 7.2 + 0.3 * daily_pattern + np.random.normal(0, 0.2, n_points),
        'dissolved_oxygen': 2.5 + 0.5 * daily_pattern + np.random.normal(0, 0.3, n_points),
        'mlss': 3500 + 200 * daily_pattern + np.random.normal(0, 100, n_points),
        'turbidity': 15 + 5 * daily_pattern + np.random.normal(0, 2, n_points),
        'conductivity': 800 + 50 * daily_pattern + np.random.normal(0, 20, n_points)
    }
    
    df = pd.DataFrame(data)
    
    # Inject some anomalies
    anomaly_indices = np.random.choice(n_points, size=int(0.05 * n_points), replace=False)
    
    for idx in anomaly_indices:
        # Random parameter anomaly
        param = np.random.choice(['influent_flow', 'ph', 'dissolved_oxygen', 'mlss'])
        if param == 'ph':
            df.loc[idx, param] = np.random.choice([5.5, 9.2])  # pH spike
        elif param == 'influent_flow':
            df.loc[idx, param] *= 2.5  # Flow surge
        else:
            df.loc[idx, param] *= np.random.choice([0.3, 3.0])  # Low or high reading
    
    return df


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data (replace with your data loading)
    print("Generating sample WWTP sensor data...")
    sample_data = generate_sample_wwtp_data(days=14)
    
    # Initialize detector
    detector = WWTPAnomalyDetector(sample_data, contamination=0.08)
    
    # Preprocess data
    processed = detector.preprocess_data(method='robust', smooth=True)
    
    # Run ensemble detection
    results = detector.ensemble_detection(['mahalanobis', 'isolation_forest', 'statistical'])
    
    # Plot results
    detector.plot_anomaly_results('mahalanobis')
    detector.plot_anomaly_results('ensemble')
    
    # Export results
    detector.export_results('sample_anomaly_results.csv')
    
    print("\nDetection complete! Check the plots and exported CSV file.")
    print("\nTo use with your own data:")
    print("1. Load your data into a pandas DataFrame")
    print("2. Ensure you have a timestamp column and numeric sensor columns")
    print("3. Initialize: detector = WWTPAnomalyDetector(your_data)")
    print("4. Run: detector.ensemble_detection()")
    print("5. Plot: detector.plot_anomaly_results()")
