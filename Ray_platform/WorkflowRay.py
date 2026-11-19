# Import necessary libraries
import numpy as np
import psutil
import time
import pandas as pd
import os
from datetime import datetime
import ray
import joblib
from concurrent.futures import ThreadPoolExecutor
from ray import tune
import matplotlib.pyplot as plt
import seaborn as sns

# Print Ray version for debugging
print(f"Using Ray version: {ray.__version__}")

# Step 1: Create Performance Monitoring Class.
class RayPerformanceMonitor:
    def __init__(self):
        self.metrics = []
        self._monitoring = False
        self.system_metrics = []
        self.monitor_thread = None
        self.monitor_future = None
        
    def start_monitoring(self):
        self._monitoring = True
        self.system_metrics = []  # Reset metrics on each start
        
        # Create a separate function for metric collection
        def collect_metrics():
            print("Monitoring thread started")  # Debug message
            start_time = time.time()
            while self._monitoring:
                try:
                    # Collect detailed system metrics
                    metrics = {
                        'timestamp': datetime.now(),
                        'cpu_percent': psutil.cpu_percent(interval=0.1),
                        'memory_percent': psutil.virtual_memory().percent,
                        'memory_used': psutil.virtual_memory().used / (1024**3),  # GB
                        'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
                    }
                    self.system_metrics.append(metrics)
                    
                    # Debug output every 10 data points
                    if len(self.system_metrics) % 10 == 0:
                        print(f"Collected {len(self.system_metrics)} metrics over {time.time() - start_time:.1f} seconds")
                        print(f"Latest CPU: {metrics['cpu_percent']:.1f}%, Memory: {metrics['memory_percent']:.1f}%")
                except Exception as e:
                    print(f"Error collecting metrics: {e}")
                
                # Small sleep to prevent CPU overload but still collect metrics frequently
                time.sleep(0.5)
        
        # Use a ThreadPoolExecutor with a single worker
        self.monitor_thread = ThreadPoolExecutor(max_workers=1)
        self.monitor_future = self.monitor_thread.submit(collect_metrics)
        
        # Small delay to ensure thread starts before continuing
        time.sleep(1.0)  # Give the thread a full second to start and collect initial metrics
        print(f"Monitoring started, thread active: {not self.monitor_future.done()}")

    def stop_monitoring(self):
        if self._monitoring:
            print(f"Stopping monitoring, collected {len(self.system_metrics)} metrics")
            self._monitoring = False
            
            # Wait for the thread to complete
            try:
                self.monitor_thread.shutdown(wait=True)
                print("Monitoring thread shut down successfully")
            except Exception as e:
                print(f"Error shutting down monitoring thread: {e}")

    def get_resource_metrics(self):
        """Return current resource metrics for direct access"""
        if not self.system_metrics:
            print("WARNING: No system metrics available")
            return {
                'avg_cpu': 0,
                'max_cpu': 0,
                'avg_memory': 0,
                'max_memory': 0
            }
        
        avg_cpu = np.mean([m['cpu_percent'] for m in self.system_metrics])
        max_cpu = max([m['cpu_percent'] for m in self.system_metrics])
        avg_memory = np.mean([m['memory_percent'] for m in self.system_metrics])
        max_memory = max([m['memory_percent'] for m in self.system_metrics])
        
        return {
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'avg_memory': avg_memory,
            'max_memory': max_memory
        }

    def log_performance(self, model_name, dataset_size, metrics):
        # Calculate system metrics if available
        resource_metrics = self.get_resource_metrics()
        
        # Create combined metrics dictionary
        performance_record = {
            'timestamp': datetime.now(),
            'model_name': model_name,
            'dataset_size': dataset_size,
            **resource_metrics,
            **metrics
        }
        
        # Add performance record to metrics list
        self.metrics.append(performance_record)
        
        # Debug output
        print(f"Performance logged for {model_name} on {dataset_size} dataset:")
        print(f"  CPU: {resource_metrics['avg_cpu']:.1f}% avg, {resource_metrics['max_cpu']:.1f}% max")
        print(f"  Memory: {resource_metrics['avg_memory']:.1f}% avg, {resource_metrics['max_memory']:.1f}% max")
        
        return performance_record

# Step 2: Initializing Ray Cluster.
def initialize_ray(num_cpus=14, object_store_memory=10 * 10**9):
    # Check if we're running in AzureML environment
    in_azureml = os.environ.get('AZUREML_EXPERIMENT_ID') is not None
    
    # Configure Ray based on our environment.
    if in_azureml:
        # In AzureML, use local mode with simpler configuration
        ray.init(ignore_reinit_error=True)
    else:
        # For a distributed cluster or local machine
        ray.init(
            ignore_reinit_error=True,
            num_cpus=num_cpus,
            object_store_memory=object_store_memory
        )
    
    print(f"Ray initialized with {ray.available_resources()}")

# Helper function to process chunks for large datasets.
def preprocess_chunk(chunk):
    # Select Relevant Columns
    selected_columns = ["ID", "Case Number", "Primary Type", "Location Description", 
                       "Domestic", "Beat", "District", "Ward", "Community Area", 
                       "Year", "Arrest", "Date"]
    
    # Only keep relevant columns
    chunk = chunk[selected_columns]
    
    # Drop duplicates
    chunk = chunk.drop_duplicates(subset=['ID', 'Case Number'])
    
    # Convert date to timestamp
    chunk['Date'] = pd.to_datetime(chunk['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    # Extract hour and day from date
    chunk['Hour'] = chunk['Date'].dt.hour
    chunk['Day'] = chunk['Date'].dt.day
    
    # Handling missing values
    chunk = chunk.dropna()
    
    # Convert boolean to integer
    chunk['Domestic'] = chunk['Domestic'].astype(int)
    chunk['Arrest'] = chunk['Arrest'].astype(int)
    
    # Create label column
    chunk['label'] = chunk['Arrest'].astype(int)
    
    return chunk

# Step 3: Data Loading and Preprocessing Function - With chunking for large files.
def load_and_preprocess_data(file_path):
    # Check if we're running in AzureML environment.
    in_azureml = os.environ.get('AZUREML_EXPERIMENT_ID') is not None
    
    # If in AzureML, check common data directories
    if in_azureml:
        directories_to_check = [
            "./",
            "./data/",
            "/tmp/azureml-data/",
            os.path.join(os.environ.get('AZUREML_DATAREFERENCE_WORKSPACEBLOBSTORE', ''), 'data')
        ]
        
        file_exists = False
        for directory in directories_to_check:
            potential_path = os.path.join(directory, os.path.basename(file_path))
            if os.path.exists(potential_path):
                file_path = potential_path
                file_exists = True
                print(f"Found data file at: {file_path}")
                break
        
        if not file_exists:
            print(f"Warning: Could not find file {file_path} in common AzureML directories.")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Check file size before loading
    file_size_gb = os.path.getsize(file_path) / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")
    
    if file_size_gb > 10:  # For very large files (over 10GB)
        print("Large file detected, using chunked reading")
        # Use chunked reading for large files
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=500000):
            # Process each chunk
            processed_chunk = preprocess_chunk(chunk)
            chunks.append(processed_chunk)
        data = pd.concat(chunks)
    else:
        # Load Dataset normally
        data = pd.read_csv(file_path)
        
        # Select Relevant Columns
        selected_columns = ["ID", "Case Number", "Primary Type", "Location Description", 
                           "Domestic", "Beat", "District", "Ward", "Community Area", 
                           "Year", "Arrest", "Date"]
        data = data[selected_columns]
        
        # Data preprocessing:
        # Drop duplicates values
        data = data.drop_duplicates(subset=['ID', 'Case Number'])

        # Convert date to timestamp
        data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

        # Extract hour and day from date
        data['Hour'] = data['Date'].dt.hour
        data['Day'] = data['Date'].dt.day

        # Handling missing values
        data = data.dropna()
        
        # Convert boolean to integer
        data['Domestic'] = data['Domestic'].astype(int)
        data['Arrest'] = data['Arrest'].astype(int)
        
        # Create label column
        data['label'] = data['Arrest'].astype(int)
    
    return data

# Step 4: Feature Processing for Ray:
def preprocess_features(data_df):
    print("Preprocessing features...")
    start_time = time.time()
    
    # Create feature encoders
    from sklearn.preprocessing import LabelEncoder
    from sklearn.decomposition import PCA
    
    primary_type_encoder = LabelEncoder()
    location_desc_encoder = LabelEncoder()
    
    # Encode categorical features
    data_df['Primary Type_Index'] = primary_type_encoder.fit_transform(data_df['Primary Type'])
    data_df['Location Description_Index'] = location_desc_encoder.fit_transform(data_df['Location Description'])
    
    # Feature columns for analysis
    feature_cols = ["Primary Type_Index", "Location Description_Index", "Domestic", 
                   "Beat", "District", "Ward", "Community Area", "Year", "Hour", "Day"]
    
    # Create feature matrix
    X = data_df[feature_cols].values
    y = data_df['label'].values
    
    print(f"Features prepared in {time.time() - start_time:.2f} seconds. Starting PCA...")
    
    # Apply PCA
    pca = PCA(n_components=5, random_state=42)
    X_pca = pca.fit_transform(X)
    
    print(f"PCA completed in {time.time() - start_time:.2f} seconds total")
    
    # Create a new DataFrame with PCA features
    pca_df = pd.DataFrame(
        X_pca, 
        columns=[f'pca_feature_{i}' for i in range(X_pca.shape[1])]
    )
    
    # Add the label column
    pca_df['label'] = y
    
    # Create feature encoders dictionary for later use
    feature_encoders = {
        'primary_type_encoder': primary_type_encoder,
        'location_desc_encoder': location_desc_encoder,
        'pca': pca
    }
    
    return pca_df, feature_encoders

# Step 5: Training Models with Ray Train.
def train_model_with_ray(preprocessed_data, model_type, param_space, dataset_size, monitor):
    # Start performance monitoring before training
    monitor.start_monitoring()
    print(f"Starting training for {model_type} on {dataset_size} dataset")
    start_time = time.time()
    
    # Create Ray Dataset
    ray_dataset = ray.data.from_pandas(preprocessed_data)
    
    # Split data
    train_dataset, test_dataset = ray_dataset.train_test_split(test_size=0.2, seed=42)
    
    # Define the training function based on model type
    def train_func(config):
        # Import required libraries inside the training function
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        import ray
        
        # Get train and validation data using fallbacks for different Ray versions:
        try:
            # Try newer Ray API first
            from ray.train import get_dataset_shard
            train_df = get_dataset_shard("train").to_pandas()
            val_df = get_dataset_shard("valid").to_pandas()
        except (ImportError, AttributeError):
            try:
                # Try ray.air.session if available
                from ray.air import session
                train_df = session.get_dataset_shard("train").to_pandas()
                val_df = session.get_dataset_shard("valid").to_pandas()
            except (ImportError, AttributeError):
                # Fallback for older Ray versions - direct access to the dataset
                train_df = train_dataset.to_pandas()
                val_df = test_dataset.to_pandas()
        
        # Separate features and labels
        X_train = train_df.drop('label', axis=1)
        y_train = train_df['label']
        X_val = val_df.drop('label', axis=1)
        y_val = val_df['label']
        
        # Initialize model based on type with config parameters
        if model_type == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=42,
                n_jobs=2  # Use 2 cores per job to avoid overloading
            )
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier(
                max_depth=config["max_depth"],
                random_state=42
            )
        elif model_type == "Logistic Regression":
            model = LogisticRegression(
                C=config["C"],
                max_iter=config["max_iter"],
                random_state=42,
                n_jobs=2  # Use 2 cores per job to avoid overloading
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Calculate metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # Report metrics to Ray Tune using different possible APIs
        try:
            from ray.air import session
            session.report({
                "auc": auc,
                "accuracy": accuracy,
                "f1_score": f1
            })
        except (ImportError, AttributeError):
            # Fallback for older Ray versions
            from ray import tune
            tune.report(auc=auc, accuracy=accuracy, f1_score=f1)
    
    # Use Ray Tune directly for maximum compatibility
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    
    # Create a simpler tuning approach
    analysis = tune.run(
        train_func,
        config=param_space,
        resources_per_trial={"cpu": 3},  # Allocate 3 CPUs per trial
        num_samples=8,  # Number of hyperparameter combinations to try
        scheduler=ASHAScheduler(
            max_t=10,
            grace_period=1,
            reduction_factor=2
        ),
        metric="auc",
        mode="max",
        name=f"{model_type}_{dataset_size}",
        local_dir="./ray_results",
        verbose=1
    )
    
    # Get best trial
    best_trial = analysis.get_best_trial("auc", "max", "last")
    best_config = best_trial.config
    metrics = best_trial.last_result
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Stop monitoring and get resource metrics
    monitor.stop_monitoring()
    
    # Add training time to metrics
    metrics['training_time'] = training_time
    
    # Log performance with resource metrics - this creates the comprehensive record
    performance_record = monitor.log_performance(model_type, dataset_size, metrics)
    
    print(f"Model {model_type} evaluation completed: AUC={metrics['auc']:.4f}, "
          f"Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    # Train final model with best hyperparameters
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    if model_type == "Random Forest":
        best_model = RandomForestClassifier(
            n_estimators=best_config["n_estimators"],
            max_depth=best_config["max_depth"],
            random_state=42,
            n_jobs=-1  # Use all available cores for final model
        )
    elif model_type == "Decision Tree":
        best_model = DecisionTreeClassifier(
            max_depth=best_config["max_depth"],
            random_state=42
        )
    elif model_type == "Logistic Regression":
        best_model = LogisticRegression(
            C=best_config["C"],
            max_iter=best_config["max_iter"],
            random_state=42,
            n_jobs=-1  # Use all available cores for final model
        )
    
    # Fit on entire training data
    X_train = train_dataset.to_pandas().drop('label', axis=1)
    y_train = train_dataset.to_pandas()['label']
    best_model.fit(X_train, y_train)
    
    return best_model, performance_record, best_config

# Step 6: Save and load models.
def save_model(model, feature_encoders, model_name, output_dir="models"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    encoders_path = os.path.join(output_dir, f"{model_name}_encoders.joblib")
    
    # Save model and encoders
    joblib.dump(model, model_path)
    joblib.dump(feature_encoders, encoders_path)
    
    print(f"Model saved to {model_path}")
    return model_path, encoders_path

def load_model(model_path, encoders_path):
    model = joblib.load(model_path)
    feature_encoders = joblib.load(encoders_path)
    return model, feature_encoders

# Step 7: Test prediction performance with different batch sizes.
def test_prediction_performance(model, feature_encoders, test_data, model_name, monitor):
    """Test prediction performance with different batch sizes"""
    batch_sizes = [1000, 5000]
    prediction_results = []
    
    for batch_size in batch_sizes:
        # Sample data for this batch size
        batch_data = test_data.sample(min(batch_size, len(test_data)))
        
        print(f"Testing {model_name} with batch size: {batch_size}")
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Encode categorical features
        batch_data_encoded = batch_data.copy()
        
        # Apply transformations
        try:
            batch_data_encoded['Primary Type_Index'] = feature_encoders['primary_type_encoder'].transform(
                batch_data['Primary Type'])
            batch_data_encoded['Location Description_Index'] = feature_encoders['location_desc_encoder'].transform(
                batch_data['Location Description'])
        except ValueError as e:
            # Handle potential unseen categories
            print(f"Warning: {e}. Falling back to safe encoding method.")
            # Apply a safer encoding approach for new categories
            for cat in batch_data['Primary Type'].unique():
                if cat not in feature_encoders['primary_type_encoder'].classes_:
                    print(f"Unseen category in Primary Type: {cat}")
            
            for cat in batch_data['Location Description'].unique():
                if cat not in feature_encoders['location_desc_encoder'].classes_:
                    print(f"Unseen category in Location Description: {cat}")
            
            # Filter to only include seen categories
            batch_data = batch_data[
                batch_data['Primary Type'].isin(feature_encoders['primary_type_encoder'].classes_) & 
                batch_data['Location Description'].isin(feature_encoders['location_desc_encoder'].classes_)
            ]
            
            # Try encoding again with filtered data
            batch_data_encoded = batch_data.copy()
            batch_data_encoded['Primary Type_Index'] = feature_encoders['primary_type_encoder'].transform(
                batch_data['Primary Type'])
            batch_data_encoded['Location Description_Index'] = feature_encoders['location_desc_encoder'].transform(
                batch_data['Location Description'])
        
        # Prepare feature columns
        feature_cols = ["Primary Type_Index", "Location Description_Index", "Domestic", 
                      "Beat", "District", "Ward", "Community Area", "Year", "Hour", "Day"]
        X_test = batch_data_encoded[feature_cols].values
        
        # Apply PCA transformation
        X_test_pca = feature_encoders['pca'].transform(X_test)
        
        # Time prediction
        start_time = time.time()
        predictions = model.predict(X_test_pca)
        prediction_time = time.time() - start_time
        
        row_count = len(batch_data)
        print(f"Processed {row_count} records in {prediction_time:.2f} seconds")
        
        # Get metrics before stopping monitoring
        resource_metrics = monitor.get_resource_metrics()
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Calculate records per second
        records_per_second = batch_size / prediction_time if prediction_time > 0 else 0
        
        # Create prediction result record
        result = {
            'model_name': model_name,
            'batch_size': batch_size,
            'prediction_time': prediction_time,
            'records_per_second': records_per_second,
            **resource_metrics
        }
        
        # Print detailed performance
        print(f"Performance metrics - Avg CPU: {resource_metrics['avg_cpu']:.1f}%, "
              f"Max CPU: {resource_metrics['max_cpu']:.1f}%, "
              f"Avg Memory: {resource_metrics['avg_memory']:.1f}%, "
              f"Records/sec: {records_per_second:.1f}")
        
        # Add results
        prediction_results.append(result)
    
    return prediction_results


# Function to save detailed prediction results with ground truth:
def save_detailed_prediction_results(model, feature_encoders, test_data, model_name, output_dir="results"):

    """Save detailed prediction results to CSV including predictions and ground truth"""

    print(f"Generating detailed prediction results for {model_name}...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Sample data for testing (use a reasonable size)
        test_sample = test_data.sample(min(10000, len(test_data)))
        
        # Encode categorical features
        test_sample_encoded = test_sample.copy()
        
        # Apply transformations safely
        try:
            test_sample_encoded['Primary Type_Index'] = feature_encoders['primary_type_encoder'].transform(
                test_sample['Primary Type'])
            test_sample_encoded['Location Description_Index'] = feature_encoders['location_desc_encoder'].transform(
                test_sample['Location Description'])
        except ValueError as e:
            # Handle potential unseen categories
            print(f"Warning: {e}. Filtering to only include seen categories.")
            
            # Filter to only include seen categories
            test_sample = test_sample[
                test_sample['Primary Type'].isin(feature_encoders['primary_type_encoder'].classes_) & 
                test_sample['Location Description'].isin(feature_encoders['location_desc_encoder'].classes_)
            ]
            
            # Try encoding again with filtered data
            test_sample_encoded = test_sample.copy()
            test_sample_encoded['Primary Type_Index'] = feature_encoders['primary_type_encoder'].transform(
                test_sample['Primary Type'])
            test_sample_encoded['Location Description_Index'] = feature_encoders['location_desc_encoder'].transform(
                test_sample['Location Description'])
        
        # Prepare feature columns
        feature_cols = ["Primary Type_Index", "Location Description_Index", "Domestic", 
                      "Beat", "District", "Ward", "Community Area", "Year", "Hour", "Day"]
        X_test = test_sample_encoded[feature_cols].values
        
        # Apply PCA transformation
        X_test_pca = feature_encoders['pca'].transform(X_test)
        
        # Make predictions
        start_time = time.time()
        predictions = model.predict(X_test_pca)
        probabilities = model.predict_proba(X_test_pca)[:, 1] if hasattr(model, "predict_proba") else predictions
        prediction_time = time.time() - start_time
        
        # Add predictions to test data
        result_df = test_sample.copy()
        result_df['predicted_label'] = predictions
        result_df['probability'] = probabilities
        result_df['true_label'] = result_df['Arrest']  # Ground truth value
        result_df['correct'] = (result_df['predicted_label'] == result_df['true_label']).astype(int)
        
        # Calculate metrics
        accuracy = (result_df['correct'].sum() / len(result_df)) * 100
        
        # Add model metadata
        result_df['model_name'] = model_name
        result_df['prediction_time_total'] = prediction_time
        result_df['prediction_time_per_record'] = prediction_time / len(result_df)
        result_df['accuracy'] = accuracy
        
        # Save the results
        output_path = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_predictions.csv")
        result_df.to_csv(output_path, index=False)
        print(f"Saved detailed prediction results to {output_path}")
        
        # Return summary metrics for further analysis
        return {
            'model_name': model_name,
            'records_processed': len(result_df),
            'prediction_time': prediction_time,
            'accuracy': accuracy,
            'records_per_second': len(result_df) / prediction_time if prediction_time > 0 else 0
        }
        
    except Exception as e:
        print(f"Error generating detailed prediction results for {model_name}: {e}")
        return None


def enhanced_prediction_analysis(best_models, feature_encoders, data):
    """Run enhanced prediction analysis with detailed results"""
    print("\nRunning enhanced prediction analysis...")
    
    # Create directories for results
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Store detailed results for each model
    detailed_results = []
    
    # Process each model
    for model_name, model in best_models.items():
        try:
            print(f"\nGenerating detailed prediction results for {model_name}...")
            
            # Generate and save detailed prediction results
            result_summary = save_detailed_prediction_results(
                model, feature_encoders, data, model_name
            )
            
            if result_summary:
                detailed_results.append(result_summary)
                
        except Exception as e:
            print(f"Error in detailed prediction analysis for {model_name}: {e}")
    
    # If we have results, save the summary and create plots
    if detailed_results:
        # Convert to DataFrame and save
        detailed_summary_df = pd.DataFrame(detailed_results)
        detailed_summary_df.to_csv('results/prediction_detailed_summary.csv', index=False)
        
        # Generate plots if we have model results dataframe
        if 'results_df' in globals() or 'results_df' in locals():
            # Create execution time validation plot
            plot_execution_time_validation(pd.DataFrame(detailed_results), results_df)
        
        # Create detailed testing performance plot
        plot_detailed_testing_performance()
    
    return detailed_results

# Step 7: Visualization Functions:
def plot_performance_metrics(results):
    
    # Check if required columns exist
    required_columns = ['model_name', 'dataset_size', 'training_time', 'accuracy', 'avg_cpu', 'avg_memory']
    missing_columns = [col for col in required_columns if col not in results.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in results DataFrame: {missing_columns}")
        # Add missing columns with default values
        for col in missing_columns:
            results[col] = 0
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training Time
    sns.barplot(data=results, x='model_name', y='training_time', hue='dataset_size', ax=axes[0,0])
    axes[0,0].set_title('Training Time by Model')
    axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
    
    # Model Performance
    sns.barplot(data=results, x='model_name', y='accuracy', hue='dataset_size', ax=axes[0,1])
    axes[0,1].set_title('Model Accuracy')
    axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
    
    # CPU Usage
    sns.barplot(data=results, x='model_name', y='avg_cpu', hue='dataset_size', ax=axes[1,0])
    axes[1,0].set_title('Average CPU Usage')
    axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
    
    # Memory Usage
    sns.barplot(data=results, x='model_name', y='avg_memory', hue='dataset_size', ax=axes[1,1])
    axes[1,1].set_title('Average Memory Usage')
    axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_performance_results.png')

def plot_prediction_performance(results_df):
    """Plot prediction performance metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prediction time by model and batch size
    sns.barplot(data=results_df, x='model_name', y='prediction_time', hue='batch_size', ax=axes[0,0])
    axes[0,0].set_title('Prediction Time by Model and Batch Size_Ray')
    axes[0,0].set_ylabel('Time (seconds)')
    axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
    
    # Throughput (predictions per second)
    results_df['throughput'] = results_df['batch_size'] / results_df['prediction_time']
    sns.barplot(data=results_df, x='model_name', y='throughput', hue='batch_size', ax=axes[0,1])
    axes[0,1].set_title('Throughput (predictions/second)_Ray')
    axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
    
    # CPU usage during prediction
    sns.barplot(data=results_df, x='model_name', y='avg_cpu', hue='batch_size', ax=axes[1,0])
    axes[1,0].set_title('CPU Usage During Prediction_Ray')
    axes[1,0].set_ylabel('CPU Usage (%)')
    axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
    
    # Memory usage during prediction
    sns.barplot(data=results_df, x='model_name', y='avg_memory', hue='batch_size', ax=axes[1,1])
    axes[1,1].set_title('Memory Usage During Prediction_Ray')
    axes[1,1].set_ylabel('Memory Usage (%)')
    axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('prediction_performance_results.png')


def plot_scalability_analysis(results_df):
    """
    Plot showing how training time scales with dataset size for each model
    """
    plt.figure(figsize=(10, 6))
    
    # Convert dataset_size to categorical for proper ordering
    size_order = ['1M', '4M', 'Full']
    results_df['dataset_size'] = pd.Categorical(results_df['dataset_size'], categories=size_order, ordered=True)
    
    # Create pivot table for line plotting
    pivot_data = results_df.pivot(index='dataset_size', columns='model_name', values='training_time')
    
    # Plot lines
    pivot_data.plot(marker='o', linewidth=2, ax=plt.gca())
    
    plt.title('Scalability Analysis: Training Time vs Dataset Size_Ray')
    plt.xlabel('Dataset Size')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Model Type')
    plt.tight_layout()
    
    plt.savefig('scalability_analysis.png')
    return plt.gcf()

def plot_resource_efficiency(results_df):
    """
    Scatter plot with bubble size showing accuracy vs training time vs memory usage
    """
    plt.figure(figsize=(10, 8))
    
    # Get unique models and dataset sizes for color mapping
    models = results_df['model_name'].unique()
    sizes = results_df['dataset_size'].unique()
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # Create scatter plot with bubbles
    for i, model in enumerate(models):
        model_data = results_df[results_df['model_name'] == model]
        
        # Scale memory for bubble size
        # Adding a minimum size to ensure bubbles are visible
        bubble_sizes = model_data['avg_memory'] * 10 + 100
        
        plt.scatter(
            model_data['accuracy'], 
            model_data['training_time'],
            s=bubble_sizes,
            c=[colors[i]],
            alpha=0.7,
            label=model
        )
    
    # Add labels for each point
    for _, row in results_df.iterrows():
        plt.annotate(
            row['dataset_size'],
            (row['accuracy'], row['training_time']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.title('Resource Efficiency: Accuracy vs Training Time vs Memory Usage_Ray')
    plt.xlabel('Accuracy')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(title='Model Type')
    
    # Add bubble size legend
    memory_levels = [10, 30, 50, 70]
    for mem in memory_levels:
        plt.scatter([], [], s=mem*10+100, c='gray', alpha=0.5, label=f'{mem}% Memory')
    
    plt.legend(title='Model Type & Memory Usage_Ray', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    plt.savefig('resource_efficiency.png')
    return plt.gcf()

def plot_training_prediction_tradeoff(results_df, pred_results_df):
    """
    Plot showing training time vs prediction throughput
    """
    plt.figure(figsize=(10, 6))
    
    # Merge results dataframes to get both training and prediction metrics
    # We'll use the largest batch size for prediction metrics
    max_batch_pred_df = pred_results_df[pred_results_df['batch_size'] == pred_results_df['batch_size'].max()]
    
    # Keep only full dataset results for simplicity
    full_results_df = results_df[results_df['dataset_size'] == 'Full']
    
    # Create plot data
    models = full_results_df['model_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        train_data = full_results_df[full_results_df['model_name'] == model]
        pred_data = max_batch_pred_df[max_batch_pred_df['model_name'] == model]
        
        if not train_data.empty and not pred_data.empty:
            plt.scatter(
                train_data['training_time'].values[0],
                pred_data['records_per_second'].values[0],
                c=[colors[i]],
                s=150,
                label=model
            )
    
    plt.title('Training-Prediction Tradeoff_Ray')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Prediction Throughput (records/second)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Model')
    plt.tight_layout()
    
    plt.savefig('training_prediction_tradeoff.png')
    return plt.gcf()

def plot_resource_scaling(results_df):
    """
    Plot showing how CPU and memory usage scales with dataset size during training
    for each model type
    """
    plt.figure(figsize=(12, 8))
    
    # Create a figure with subplots - one for CPU, one for memory
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Convert dataset_size to categorical for proper ordering if not already
    if not pd.api.types.is_categorical_dtype(results_df['dataset_size']):
        size_order = ['1M', '4M', 'Full']
        results_df['dataset_size'] = pd.Categorical(
            results_df['dataset_size'], 
            categories=size_order, 
            ordered=True
        )
    
    # Get unique models for color mapping
    models = results_df['model_name'].unique()
    
    # Plot CPU usage by dataset size for each model
    for model in models:
        model_data = results_df[results_df['model_name'] == model]
        model_data = model_data.sort_values('dataset_size')
        
        ax1.plot(model_data['dataset_size'], model_data['avg_cpu'], 
                marker='o', linewidth=2, label=model)
        
        # Add value labels
        for i, row in model_data.iterrows():
            ax1.annotate(f"{row['avg_cpu']:.1f}%", 
                        (row['dataset_size'], row['avg_cpu']),
                        textcoords="offset points", 
                        xytext=(0,5), 
                        ha='center')
    
    ax1.set_title('Average CPU Usage During Training by Dataset Size_Ray')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(title='Model Type')
    
    # Plot Memory usage by dataset size for each model
    for model in models:
        model_data = results_df[results_df['model_name'] == model]
        model_data = model_data.sort_values('dataset_size')
        
        ax2.plot(model_data['dataset_size'], model_data['avg_memory'], 
                marker='s', linewidth=2, label=model)
        
        # Add value labels
        for i, row in model_data.iterrows():
            ax2.annotate(f"{row['avg_memory']:.1f}%", 
                        (row['dataset_size'], row['avg_memory']),
                        textcoords="offset points", 
                        xytext=(0,5), 
                        ha='center')
    
    ax2.set_title('Average Memory Usage During Training by Dataset Size_Ray')
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Memory Usage (%)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(title='Model Type')
    
    plt.tight_layout()
    plt.savefig('resource_scaling_by_dataset.png')
    return fig

def plot_batch_size_optimization(pred_results_df):
    """
    Plot showing relationship between batch size, throughput, and memory usage
    """
    plt.figure(figsize=(10, 6))
    
    # Create subplot with shared x-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Get unique models
    models = pred_results_df['model_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # Plot throughput lines
    for i, model in enumerate(models):
        model_data = pred_results_df[pred_results_df['model_name'] == model]
        # Sort by batch size to ensure correct line
        model_data = model_data.sort_values('batch_size')
        
        ax1.plot(model_data['batch_size'], model_data['records_per_second'], 
                marker='o', linewidth=2, color=colors[i], label=model)
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (records/second)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create second y-axis for memory usage
    ax2 = ax1.twinx()
    
    # Plot memory usage as dashed lines
    for i, model in enumerate(models):
        model_data = pred_results_df[pred_results_df['model_name'] == model]
        # Sort by batch size to ensure correct line
        model_data = model_data.sort_values('batch_size')
        
        ax2.plot(model_data['batch_size'], model_data['avg_memory'], 
                marker='s', linestyle='--', color=colors[i])
    
    ax2.set_ylabel('Memory Usage (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Batch Size Optimization: Throughput vs Memory Usage_Ray')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(title='Model')
    plt.tight_layout()
    
    plt.savefig('batch_size_optimization.png')
    return fig

def plot_parallel_scaling_efficiency():
    """
    Placeholder function for parallel scaling efficiency plot
    """
    plt.figure(figsize=(10, 6))
    
    # Simulate ideal scaling
    cores = np.array([1, 2, 4, 8, 16])
    ideal_speedup = cores
    
    # Simulate realistic scaling with diminishing returns
    # Using Amdahl's Law with different parallel portions
    def amdahl_speedup(cores, parallel_portion):
        return 1 / ((1 - parallel_portion) + parallel_portion / cores)
    
    # Different models have different parallel efficiency
    speedup_rf = amdahl_speedup(cores, 0.95)  # Random Forest - highly parallel
    speedup_lr = amdahl_speedup(cores, 0.85)  # Logistic Regression - less parallel
    speedup_dt = amdahl_speedup(cores, 0.80)  # Decision Tree - least parallel
    
    plt.plot(cores, ideal_speedup, 'k--', label='Ideal Linear Speedup')
    plt.plot(cores, speedup_rf, 'b-o', label='Random Forest')
    plt.plot(cores, speedup_lr, 'r-s', label='Logistic Regression')
    plt.plot(cores, speedup_dt, 'g-^', label='Decision Tree')
    
    plt.title('Parallel Scaling Efficiency_Ray')
    plt.xlabel('Number of Executor Cores')
    plt.ylabel('Speedup Factor')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(cores)
    plt.tight_layout()
    
    plt.savefig('parallel_scaling_efficiency.png')
    return plt.gcf()

# Plot execution time validation function
def plot_execution_time_validation(pred_results_df, model_results_df):
    """Plot execution time validation comparing different aspects of model execution time"""
    print("Generating execution time validation plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Training Time vs Prediction Time
    ax1 = axes[0, 0]
    
    # Get training times from model_results_df - focusing on full dataset
    train_times = model_results_df[model_results_df['dataset_size'] == 'Full'][['model_name', 'training_time']]
    
    # Get prediction times - use mean across different batch sizes
    pred_times = pred_results_df.groupby('model_name')['prediction_time'].mean().reset_index()
    
    # Merge the data
    time_comparison = pd.merge(train_times, pred_times, on='model_name')
    
    # Create a grouped bar chart
    models = time_comparison['model_name']
    x = np.arange(len(models))
    bar_width = 0.35
    
    ax1.bar(x - bar_width/2, time_comparison['training_time'], bar_width, label='Training Time', color='skyblue')
    ax1.bar(x + bar_width/2, time_comparison['prediction_time'], bar_width, label='Prediction Time', color='orange')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training Time vs Prediction Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Log scale comparison to show difference magnitudes
    ax2 = axes[0, 1]
    
    ax2.bar(x - bar_width/2, time_comparison['training_time'], bar_width, label='Training Time', color='skyblue')
    ax2.bar(x + bar_width/2, time_comparison['prediction_time'], bar_width, label='Prediction Time', color='orange')
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Time (seconds) - Log Scale')
    ax2.set_title('Training vs Prediction Time (Log Scale)')
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Records per second by batch size
    ax3 = axes[1, 0]
    
    # Pivot the data to have models as rows and batch sizes as columns
    throughput_pivot = pred_results_df.pivot(index='model_name', columns='batch_size', values='records_per_second')
    throughput_pivot.plot(kind='bar', ax=ax3)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Records per Second')
    ax3.set_title('Prediction Throughput by Batch Size')
    ax3.legend(title='Batch Size')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Time Efficiency Ratio (records processed / time taken)
    ax4 = axes[1, 1]
    
    # Calculate efficiency as records_per_second / training_time (higher is better)
    # This shows which models give the best performance return on training investment
    efficiency_data = []
    
    for model in models:
        # Get average records per second for this model
        avg_rps = pred_results_df[pred_results_df['model_name'] == model]['records_per_second'].mean()
        # Get training time
        train_time = time_comparison[time_comparison['model_name'] == model]['training_time'].values[0]
        
        # Calculate efficiency (higher is better)
        # Using log scale to normalize the values since training times can be much larger
        efficiency = np.log10(avg_rps) / np.log10(train_time) if train_time > 0 else 0
        
        efficiency_data.append({
            'model_name': model,
            'efficiency': efficiency
        })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    
    # Plot efficiency
    sns.barplot(data=efficiency_df, x='model_name', y='efficiency', ax=ax4, palette='viridis')
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Log-Efficiency Ratio')
    ax4.set_title('Time Efficiency: Throughput/Training Time (Log Scale)')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(efficiency_df['efficiency']):
        ax4.text(i, v + 0.05, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('execution_time_validation.png')
    print("Execution time validation plot saved as 'execution_time_validation.png'")
    return fig

# Plot detailed model testing performance
def plot_detailed_testing_performance(detailed_results_dir="results"):
    """Plot detailed testing performance metrics from saved prediction CSVs"""
    print("Generating detailed testing performance plots...")
    
    # Check if results directory exists
    if not os.path.exists(detailed_results_dir):
        print(f"Results directory '{detailed_results_dir}' not found.")
        return None
    
    # Get list of prediction CSV files
    prediction_files = [f for f in os.listdir(detailed_results_dir) if f.endswith('_predictions.csv')]
    
    if not prediction_files:
        print(f"No prediction CSV files found in {detailed_results_dir}")
        return None
    
    # Load all prediction files into a single DataFrame
    all_results = []
    for file in prediction_files:
        try:
            file_path = os.path.join(detailed_results_dir, file)
            df = pd.read_csv(file_path)
            all_results.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_results:
        print("No valid prediction results could be loaded.")
        return None
        
    # Combine all results
    combined_results = pd.concat(all_results)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Confusion Matrix Heat Map for each model
    ax1 = axes[0, 0]
    models = combined_results['model_name'].unique()
    
    # Create a confusion matrix for each model and stack them together
    confusion_data = []
    for model in models:
        model_data = combined_results[combined_results['model_name'] == model]
        
        # Count true positives, false positives, etc.
        tp = ((model_data['true_label'] == 1) & (model_data['predicted_label'] == 1)).sum()
        fp = ((model_data['true_label'] == 0) & (model_data['predicted_label'] == 1)).sum()
        tn = ((model_data['true_label'] == 0) & (model_data['predicted_label'] == 0)).sum()
        fn = ((model_data['true_label'] == 1) & (model_data['predicted_label'] == 0)).sum()
        
        confusion_data.append({
            'model': model,
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        })
    
    confusion_df = pd.DataFrame(confusion_data)
    
    # Plot confusion matrix components as a heatmap
    confusion_plot = confusion_df[['model', 'true_positive', 'false_positive', 'true_negative', 'false_negative']]
    confusion_plot = confusion_plot.set_index('model')
    
    sns.heatmap(confusion_plot, annot=True, fmt='g', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix Components by Model')
    
    # 2. Performance Metrics Bar Chart
    ax2 = axes[0, 1]
    
    # Plot precision, recall, F1 score
    metrics_df = confusion_df.melt(
        id_vars=['model'], 
        value_vars=['precision', 'recall', 'f1_score'],
        var_name='metric', 
        value_name='value'
    )
    
    sns.barplot(data=metrics_df, x='model', y='value', hue='metric', ax=ax2)
    ax2.set_title('Performance Metrics by Model')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Prediction Confidence Distribution
    ax3 = axes[1, 0]
    
    for model in models:
        model_data = combined_results[combined_results['model_name'] == model]
        # Only plot if probability data exists
        if 'probability' in model_data.columns:
            sns.kdeplot(model_data['probability'], ax=ax3, label=model)
    
    ax3.set_title('Prediction Probability Distribution')
    ax3.set_xlabel('Prediction Probability')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Error Analysis by Feature
    ax4 = axes[1, 1]
    
    # Analyze error rates by one feature, e.g., "Primary Type"
    if 'Primary Type' in combined_results.columns:
        # Get top 10 most common primary types
        top_types = combined_results['Primary Type'].value_counts().head(10).index.tolist()
        error_data = []
        
        for model in models:
            model_data = combined_results[combined_results['model_name'] == model]
            
            for ptype in top_types:
                type_data = model_data[model_data['Primary Type'] == ptype]
                if len(type_data) > 0:
                    error_rate = 1 - (type_data['correct'].sum() / len(type_data))
                    error_data.append({
                        'model': model,
                        'primary_type': ptype,
                        'error_rate': error_rate,
                        'count': len(type_data)
                    })
        
        error_df = pd.DataFrame(error_data)
        
        # Create bubble chart - size of bubble represents count
        for model in models:
            model_error = error_df[error_df['model'] == model]
            ax4.scatter(
                model_error['primary_type'], 
                model_error['error_rate'],
                s=model_error['count'] / 10,  # Scale down the count for better bubble sizes
                alpha=0.7,
                label=model
            )
        
        ax4.set_title('Error Rate by Primary Crime Type')
        ax4.set_xlabel('Primary Type')
        ax4.set_ylabel('Error Rate')
        ax4.set_ylim(0, 1)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend()
        plt.xticks(rotation=90)
    else:
        ax4.text(0.5, 0.5, 'Primary Type data not available for error analysis', 
                 ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('detailed_testing_performance.png')
    print("Detailed testing performance plot saved as 'detailed_testing_performance.png'")
    return fig


# Step 8: Main Execution
def main():
    try:
        print("Starting the Ray ML Performance Analysis...")
        
        # Initialize Ray
        print("Initializing Ray...")
        initialize_ray(num_cpus=14, object_store_memory=10 * 10**9)  # 10GB memory
        
        # Initialize performance monitor
        monitor = RayPerformanceMonitor()
        
        # Load and process data
        print("Loading and processing data...")
        file_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/computeinw/code/Users/bra_er-rahmaouy/Crimes_-_2001_to_Present_20250127.csv"
        data = load_and_preprocess_data(file_path)
        
        # Print data summary
        record_count = len(data)
        print(f"Data loaded successfully. Total records: {record_count}")
        
        # Process features
        print("Processing features...")
        processed_data, feature_encoders = preprocess_features(data)
        
        # Create different dataset sizes with safety checks
        total_size = len(processed_data)
        sizes = {
            "1M": min(1_000_000, total_size),
            "4M": min(4_000_000, total_size),
            "Full": total_size
        }
        all_results = []
        best_models = {}
        best_configs = {}
        
        # Define models and their parameter spaces - simplified for debugging
        models = {
            "Random Forest": {
                "n_estimators": tune.choice([50]),
                "max_depth": tune.choice([5])
            },
            "Decision Tree": {
                "max_depth": tune.choice([5])
            },
            "Logistic Regression": {
                "C": tune.choice([0.01]),
                "max_iter": tune.choice([100])
            }
        }
        
        # Train and evaluate models for each dataset size
        for size_name, size_value in sizes.items():
            print(f"\nProcessing dataset size: {size_name}")
            data_sample = processed_data.sample(min(size_value, len(processed_data)))
            
            for model_name, param_space in models.items():
                try:
                    print(f"\nTraining {model_name}...")
                    best_model, metrics, best_config = train_model_with_ray(
                        data_sample, model_name, param_space, size_name, monitor
                    )
                    all_results.append(metrics)
                    
                    # Save the best model for each model type
                    if size_name == "Full":
                        best_models[model_name] = best_model
                        best_configs[model_name] = best_config
                except Exception as e:
                    print(f"Error training {model_name} on {size_name} dataset: {e}")
                    # Add default metrics to ensure the program continues
                    all_results.append({
                        'model_name': model_name,
                        'dataset_size': size_name,
                        'training_time': 0,
                        'accuracy': 0,
                        'auc': 0,
                        'f1_score': 0,
                        'avg_cpu': 0,
                        'max_cpu': 0,
                        'avg_memory': 0,
                        'max_memory': 0
                    })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Print column names for debugging
        print(f"Available columns in results_df: {results_df.columns.tolist()}")
        print(f"Sample of results data:\n{results_df.head()}")
        
        # Save results
        results_df.to_csv('model_performance_results.csv', index=False)
        
        # Generate performance visualizations
        print("Generating model performance visualizations...")
        plot_performance_metrics(results_df)
        plot_scalability_analysis(results_df)
        plot_resource_efficiency(results_df)
        plot_resource_scaling(results_df)
        
        
        # Test prediction performance for each model
        prediction_results = []
        for model_name, model in best_models.items():
            try:
                print(f"\nTesting prediction performance for {model_name}...")
                
                # Save the model
                model_path, encoders_path = save_model(
                    model, 
                    feature_encoders, 
                    f"{model_name.replace(' ', '_').lower()}"
                )
                
                # Test prediction performance
                model_results = test_prediction_performance(
                    model, feature_encoders, data, model_name, monitor
                )
                prediction_results.extend(model_results)
            except Exception as e:
                print(f"Error testing prediction performance for {model_name}: {e}")
        
        # Convert prediction results to DataFrame and save
        if prediction_results:
            pred_results_df = pd.DataFrame(prediction_results)
            print(f"Available columns in pred_results_df: {pred_results_df.columns.tolist()}")
            print(f"Sample of prediction results:\n{pred_results_df.head()}")
            pred_results_df.to_csv('prediction_performance_results.csv', index=False)
            
            # Generate prediction visualizations
            print("Generating prediction performance visualizations...")
            plot_prediction_performance(pred_results_df)
            plot_batch_size_optimization(pred_results_df)
            
            # Plot combined visualizations
            plot_training_prediction_tradeoff(results_df, pred_results_df)
        
        # Generate parallel scaling efficiency simulation
        print("Generating parallel scaling efficiency simulation...")
        plot_parallel_scaling_efficiency()


        # Run enhanced prediction analysis with detailed CSV output
        print("\nRunning enhanced prediction analysis...")
        detailed_results = enhanced_prediction_analysis(best_models, feature_encoders, data)
        
        # Generate execution time validation plots if we have both results
        if 'pred_results_df' in locals() and detailed_results:
            print("Generating execution time validation plots...")
            detailed_summary_df = pd.DataFrame(detailed_results)
            plot_execution_time_validation(detailed_summary_df, results_df)
        
        # Generate detailed testing performance plots
        print("Generating detailed testing performance plots...")
        plot_detailed_testing_performance()

        # Example of loading and using a saved model for predictions
        print("\nExample of loading a saved model for predictions:")
        try:
            model_path = "models/random_forest.joblib"
            encoders_path = "models/random_forest_encoders.joblib"
            
            if os.path.exists(model_path) and os.path.exists(encoders_path):
                loaded_model, loaded_encoders = load_model(model_path, encoders_path)
                
                # Create a small sample for demonstration
                sample_data = data.sample(10)
                
                # Process features for prediction
                sample_data_encoded = sample_data.copy()
                sample_data_encoded['Primary Type_Index'] = loaded_encoders['primary_type_encoder'].transform(
                    sample_data['Primary Type'])
                sample_data_encoded['Location Description_Index'] = loaded_encoders['location_desc_encoder'].transform(
                    sample_data['Location Description'])
                
                # Prepare feature columns
                feature_cols = ["Primary Type_Index", "Location Description_Index", "Domestic", 
                            "Beat", "District", "Ward", "Community Area", "Year", "Hour", "Day"]
                X_sample = sample_data_encoded[feature_cols].values
                
                # Apply PCA transformation
                X_sample_pca = loaded_encoders['pca'].transform(X_sample)
                
                # Make predictions
                print("Making predictions with the loaded model...")
                predictions = loaded_model.predict(X_sample_pca)
                probabilities = loaded_model.predict_proba(X_sample_pca)[:, 1] if hasattr(loaded_model, "predict_proba") else predictions
                
                # Add predictions to sample data
                sample_data['prediction'] = predictions
                sample_data['probability'] = probabilities
                
                # Show results
                print("\nPrediction Results:")
                print(sample_data[["Primary Type", "Location Description", "Year", "Hour", "Day", 
                                "prediction", "probability"]].head(20))
            else:
                print(f"Model files not found: {model_path} or {encoders_path}")
        except Exception as e:
            print(f"Error loading model or making predictions: {e}")
        
        # Save results to Azure ML 
        try:
            from azureml.core import Run
            run = Run.get_context()
            if os.path.exists('model_performance_results.csv'):
                run.upload_file("outputs/model_performance_results.csv", "model_performance_results.csv")
            if os.path.exists('prediction_performance_results.csv'):
                run.upload_file("outputs/prediction_performance_results.csv", "prediction_performance_results.csv")

            if os.path.exists('results/prediction_detailed_summary.csv'):  # NEW: Upload detailed results
                run.upload_file("outputs/prediction_detailed_summary.csv", "results/prediction_detailed_summary.csv")   
            
            # Upload visualization images 
            for img_file in ['model_performance_results.png', 'prediction_performance_results.png',
                           'scalability_analysis.png', 'resource_efficiency.png',
                           'training_prediction_tradeoff.png', 'resource_scaling_by_dataset.png',
                           'batch_size_optimization.png', 'parallel_scaling_efficiency.png',
                           'execution_time_validation.png', 'detailed_testing_performance.png']:
                if os.path.exists(img_file):
                    run.upload_file(f"outputs/{img_file}", img_file)
                    print(f"Uploaded {img_file} to Azure ML outputs")
        except Exception as e:
            print(f"Not running in AzureML or error uploading files: {e}")
        
        print("Performance analysis completed successfully!")

    finally:
        # Make sure monitoring is stopped
        if 'monitor' in locals():
            try:
                monitor.stop_monitoring()
            except Exception as e:
                print(f"Error stopping monitoring: {e}")
        
        # Shutdown Ray
        try:
            ray.shutdown()
            print("Ray shutdown completed")
        except Exception as e:
            print(f"Error shutting down Ray: {e}")


if __name__ == "__main__":
    main()