# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import psutil
import time
import pandas as pd
import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, countDistinct, to_timestamp, hour, dayofmonth
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from concurrent.futures import ThreadPoolExecutor
from pyspark.ml import Pipeline, PipelineModel

# Step 1: Create Performance Monitoring Class:
class SparkPerformanceMonitor:
    def __init__(self, spark):
        self.spark = spark
        self.metrics = []
        self._monitoring = False
        self.system_metrics = []
        
    def start_monitoring(self):
        self._monitoring = True
        self.system_metrics = []  # Reset metrics on each start
        
        # Create a separate function for metric collection
        def collect_metrics():
            print("Monitoring thread started")  # Debug message
            start_time = time.time()
            while self._monitoring:
                try:
                    # Only collect system metrics, not Spark executor metrics
                    metrics = {
                        'timestamp': datetime.now(),
                        'cpu_percent': psutil.cpu_percent(interval=0.5),
                        'memory_percent': psutil.virtual_memory().percent,
                    }
                    self.system_metrics.append(metrics)
                    
                    # Debug output every 10 data points
                    if len(self.system_metrics) % 10 == 0:
                        print(f"Collected {len(self.system_metrics)} metrics over {time.time() - start_time:.1f} seconds")
                except Exception as e:
                    print(f"Error collecting metrics: {e}")
                
                # Small sleep to prevent CPU overload but still collect metrics
                time.sleep(0.5)
        
        # Use a ThreadPoolExecutor with a single worker
        self.monitor_thread = ThreadPoolExecutor(max_workers=1)
        self.monitor_future = self.monitor_thread.submit(collect_metrics)
        
        # Small delay to ensure thread starts before continuing
        time.sleep(1)
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

    def log_performance(self, model_name, dataset_size, metrics):
        # Calculate system metrics if available
        if self.system_metrics:
            avg_cpu = np.mean([m['cpu_percent'] for m in self.system_metrics])
            max_cpu = max([m['cpu_percent'] for m in self.system_metrics])
            avg_memory = np.mean([m['memory_percent'] for m in self.system_metrics])
            max_memory = max([m['memory_percent'] for m in self.system_metrics])
            
            # Print debug information
            print(f"Performance metrics - Model: {model_name}, Dataset: {dataset_size}")
            print(f"CPU: avg={avg_cpu:.2f}%, max={max_cpu:.2f}%")
            print(f"Memory: avg={avg_memory:.2f}%, max={max_memory:.2f}%")
        else:
            # Provide default values if no metrics collected
            avg_cpu = max_cpu = avg_memory = max_memory = 0
            print("WARNING: No system metrics available for performance logging")
        
        # Add the performance metrics to the metrics dictionary
        metrics_with_resources = {
            'timestamp': datetime.now(),
            'model_name': model_name,
            'dataset_size': dataset_size,
            'avg_cpu': float(avg_cpu),  # Convert to float to ensure numeric type
            'max_cpu': float(max_cpu),
            'avg_memory': float(avg_memory),
            'max_memory': float(max_memory),
            **metrics
        }
        
        # Append to metrics list
        self.metrics.append(metrics_with_resources)
        return metrics_with_resources  # Return the metrics for immediate use

# Step 2: Initialize Spark Session with Performance Configuration
def create_spark_session(app_name, executor_cores, executor_memory, executor_instances):
    # Check if we're running in AzureML environment
    in_azureml = os.environ.get('AZUREML_EXPERIMENT_ID') is not None
    
    # Create builder with basic configs
    builder = SparkSession.builder.appName(app_name)
    
    # Apply different configurations based on environment
    if in_azureml:
        # In AzureML, use local mode with simpler configuration
        builder = (builder
            .master("local[*]")  # Use all available cores
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", "200"))
    else:
        # For a distributed Spark cluster
        builder = (builder
            .config("spark.executor.cores", executor_cores)
            .config("spark.executor.memory", executor_memory)
            .config("spark.executor.instances", executor_instances)
            .config("spark.driver.memory", "8g")
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.5")
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.shuffle.spill", "true"))
    
    # Create the session
    spark = builder.getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


# Step 3: Data Loading and Preprocessing Function
def load_and_preprocess_data(spark, file_path):
    # Load Dataset
    data = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Select Relevant Columns
    selected_columns = ["ID", "Case Number", "Primary Type", "Location Description", 
                       "Domestic", "Beat", "District", "Ward", "Community Area", 
                       "Year", "Arrest", "Date"]
    data = data.select(selected_columns)
    
    # Data preprocessing:

    # Drop duplicates valus.
    data = data.dropDuplicates(subset=['ID', 'Case Number'])

    # convert date to timestamp.
    data = data.withColumn("Date", to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a"))

    # Extract hour and day from date.
    data = data.withColumn("Hour", hour(col("Date")))
    data = data.withColumn("Day", dayofmonth(col("Date")))

    # hindling missing valus.
    data = data.dropna()
    
    # Convert boolean to integer
    data = data.withColumn("Domestic", when(col("Domestic") == True, 1).otherwise(0))
    
    return data

# Step 4: Feature Processing Pipeline (for consistent preprocessing in training and prediction)
def create_feature_pipeline(input_data):
    # Create indexers for categorical features
    categorical_cols = ["Primary Type", "Location Description"]
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index", handleInvalid="keep") 
                for col in categorical_cols]
    
    # Feature columns for vector assembly
    feature_cols = ["Primary Type_Index", "Location Description_Index", "Domestic", 
                   "Beat", "District", "Ward", "Community Area", "Year", "Hour", "Day"]
    
    # Create feature processing stages
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features", handleInvalid="keep")
    scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withStd=True, withMean=True)
    pca = PCA(k=5, inputCol="features", outputCol="pca_features")
    
    # Create and fit the pipeline
    feature_pipeline = Pipeline(stages=indexers + [assembler, scaler, pca])
    
    return feature_pipeline

# Step 5: Model Training and Evaluation Function
def train_and_evaluate_model(model, train_data, test_data, model_name, dataset_size, 
                           param_grid, monitor):
    # Start monitoring and give it time to initialize
    monitor.start_monitoring()
    
    # Debug output
    print(f"Starting training for {model_name} on {dataset_size} dataset")
    start_time = time.time()

    # Train model with cross validation
    crossval = CrossValidator(estimator=model,
                            estimatorParamMaps=param_grid,
                            evaluator=BinaryClassificationEvaluator(),
                            numFolds=3,
                            parallelism=4)
    
    model_trained = crossval.fit(train_data)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    start_time = time.time()
    predictions = model_trained.transform(test_data)

    # Force evaluation with an action
    predictions.cache()
    row_count = predictions.count()
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", 
                                                 rawPredictionCol="rawPrediction")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", 
                                                     metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", 
                                                    metricName="f1")
    
    # Calculate performance metrics
    auc = evaluator_auc.evaluate(predictions)
    accuracy = evaluator_acc.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)
    
    # Create metrics dictionary
    metrics = {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'f1_score': float(f1_score),
        'training_time': float(training_time),
        'prediction_time': float(prediction_time),
    }
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Log performance (system metrics now handled in log_performance)
    complete_metrics = monitor.log_performance(model_name, dataset_size, metrics)
    
    print(f"Model {model_name} evaluation completed: AUC={auc:.4f}, Accuracy={accuracy:.4f}, f1_score={f1_score:.4f}")
    return model_trained.bestModel, complete_metrics


# Step 6: Create a complete model pipeline for deployment
def create_model_pipeline(feature_pipeline_model, model):
    """Create a complete model pipeline for deployment"""
    # Create a full pipeline containing feature processing and model
    return PipelineModel(stages=feature_pipeline_model.stages + [model])

# Step 7: Save and load models
def save_model(model, model_name, output_dir="models"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, model_name)
    model.write().overwrite().save(model_path)
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_path):
    return PipelineModel.load(model_path)

# Step 8: Test prediction performance with different batch sizes
def test_prediction_performance(spark, model, test_data, model_name, monitor):
    """Test prediction performance with different batch sizes"""
    batch_sizes = [1000, 5000]
    prediction_results = []
    
    for batch_size in batch_sizes:
        # Sample data for this batch size
        batch_data = test_data.limit(batch_size)
        
        print(f"Testing {model_name} with batch size: {batch_size}")
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Time prediction
        start_time = time.time()
        predictions = model.transform(batch_data)
        # Force evaluation with an action
        predictions.cache()
        row_count = predictions.count()
        prediction_time = time.time() - start_time
        
        print(f"Processed {row_count} records in {prediction_time:.2f} seconds")
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Calculate system metrics if available
        if monitor.system_metrics:
            avg_cpu = np.mean([m['cpu_percent'] for m in monitor.system_metrics])
            max_cpu = max([m['cpu_percent'] for m in monitor.system_metrics])
            avg_memory = np.mean([m['memory_percent'] for m in monitor.system_metrics])
            max_memory = max([m['memory_percent'] for m in monitor.system_metrics])
            print(f"Performance metrics - Avg CPU: {avg_cpu:.1f}%, Max CPU: {max_cpu:.1f}%, " 
                  f"Avg Memory: {avg_memory:.1f}%, Records/sec: {batch_size/prediction_time:.1f}")
        else:
            # Provide default values if no metrics collected
            avg_cpu = max_cpu = avg_memory = max_memory = 0
            print("WARNING: No system metrics available for performance analysis")
        
        # Add results
        prediction_results.append({
            'model_name': model_name,
            'batch_size': batch_size,
            'prediction_time': prediction_time,
            'records_per_second': batch_size / prediction_time,
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'avg_memory': avg_memory,
            'max_memory': max_memory
        })
    
    return prediction_results

# Function to save detailed prediction results with ground truth:
def save_detailed_prediction_results(spark, model, test_data, model_name, output_dir="results"):
    """
    Save detailed prediction results to CSV including predictions and ground truth.
    Adapted for PySpark models.
    
    Args:
        spark: SparkSession object
        model: PySpark Pipeline model
        test_data: PySpark DataFrame with test data
        model_name: Name of the model for identification
        output_dir: Directory to save the results
    
    Returns:
        Dictionary with performance metrics
    """
    import os
    import time
    import pandas as pd
    from pyspark.sql.functions import col, when

    print(f"Generating detailed prediction results for {model_name}...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Sample data for testing (use a reasonable size)
        test_sample = test_data.limit(10000)
        
        # Make predictions
        start_time = time.time()
        predictions = model.transform(test_sample)
        
        # Force evaluation with an action
        predictions.cache()
        row_count = predictions.count()
        prediction_time = time.time() - start_time
        
        # Add a column to indicate if prediction was correct
        predictions = predictions.withColumn(
            "correct", 
            when(col("prediction") == col("label"), 1).otherwise(0)
        )
        
        # Select relevant columns for output
        result_df = predictions.select(
            "Primary Type", "Location Description", "Beat", "District", 
            "Ward", "Community Area", "Year", "Hour", "Day",
            "label", "prediction", "probability", "correct"
        )
        
        # Calculate accuracy
        accuracy = predictions.select("correct").agg({"correct": "mean"}).collect()[0][0] * 100
        
        # Convert to pandas for easier CSV saving
        pandas_df = result_df.toPandas()
        
        # Add model metadata
        pandas_df['model_name'] = model_name
        pandas_df['prediction_time_total'] = prediction_time
        pandas_df['prediction_time_per_record'] = prediction_time / len(pandas_df) if len(pandas_df) > 0 else 0
        pandas_df['accuracy'] = accuracy
        
        # Save the results
        output_path = os.path.join(output_dir, f"{model_name.replace(' ', '_').lower()}_predictions.csv")
        pandas_df.to_csv(output_path, index=False)
        print(f"Saved detailed prediction results to {output_path}")
        
        # Return summary metrics for further analysis
        return {
            'model_name': model_name,
            'records_processed': row_count,
            'prediction_time': prediction_time,
            'accuracy': float(accuracy),
            'records_per_second': row_count / prediction_time if prediction_time > 0 else 0
        }
        
    except Exception as e:
        print(f"Error generating detailed prediction results for {model_name}: {e}")
        return None

# Step 9: Visualization Functions:
def plot_performance_metrics(results):
    # Convert to pandas DataFrame 
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)
    
    # Check if required columns exist
    required_columns = ['model_name', 'dataset_size', 'training_time', 'accuracy', 'avg_cpu', 'avg_memory']
    missing_columns = [col for col in required_columns if col not in results.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in results DataFrame: {missing_columns}")
        # Add missing columns with default values
        for col in missing_columns:
            results[col] = 0
    
    # Print the data before plotting to debug
    print("Data to be plotted:")
    print(results[['model_name', 'dataset_size', 'training_time', 'accuracy', 'avg_cpu', 'avg_memory']].head())
    
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

def plot_prediction_performance(results_df, detailed_results=None):
    """Plot prediction performance metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prediction time by model and batch size
    sns.barplot(data=results_df, x='model_name', y='prediction_time', hue='batch_size', ax=axes[0,0])
    axes[0,0].set_title('Prediction Time by Model and Batch Size')
    axes[0,0].set_ylabel('Time (seconds)')
    axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
    
    # Throughput (predictions per second)
    results_df['throughput'] = results_df['batch_size'] / results_df['prediction_time']
    sns.barplot(data=results_df, x='model_name', y='throughput', hue='batch_size', ax=axes[0,1])
    axes[0,1].set_title('Throughput (predictions/second)')
    axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
    
    # CPU usage during prediction
    sns.barplot(data=results_df, x='model_name', y='avg_cpu', hue='batch_size', ax=axes[1,0])
    axes[1,0].set_title('CPU Usage During Prediction')
    axes[1,0].set_ylabel('CPU Usage (%)')
    axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
    
    # Memory usage during prediction
    sns.barplot(data=results_df, x='model_name', y='avg_memory', hue='batch_size', ax=axes[1,1])
    axes[1,1].set_title('Memory Usage During Prediction')
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
    
    plt.title('Scalability Analysis: Training Time vs Dataset Size')
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
        
        # Scale memory for bubble size (adjust scaling factor as needed)
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
    
    plt.title('Resource Efficiency: Accuracy vs Training Time vs Memory Usage')
    plt.xlabel('Accuracy')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(title='Model Type')
    
    # Add bubble size legend
    memory_levels = [10, 30, 50, 70]
    for mem in memory_levels:
        plt.scatter([], [], s=mem*10+100, c='gray', alpha=0.5, label=f'{mem}% Memory')
    
    plt.legend(title='Model Type & Memory Usage', loc='upper left', bbox_to_anchor=(1, 1))
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
    
    plt.title('Training-Prediction Tradeoff')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Prediction Throughput (records/second)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Model')
    plt.tight_layout()
    
    plt.savefig('training_prediction_tradeoff.png')
    return plt.gcf()

def plot_resource_utilization_timeline(monitor_metrics):
    """
    Line chart showing CPU and memory usage over time during training
    """
    if not monitor_metrics:
        print("No timeline metrics available for plotting")
        return None
    
    # Convert list of dictionaries to DataFrame
    metrics_df = pd.DataFrame(monitor_metrics)
    
    # Convert timestamp to seconds elapsed from start
    start_time = metrics_df['timestamp'].min()
    metrics_df['seconds_elapsed'] = (metrics_df['timestamp'] - start_time).dt.total_seconds()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot CPU usage
    plt.plot(metrics_df['seconds_elapsed'], metrics_df['cpu_percent'], 'b-', 
             linewidth=2, label='CPU Usage (%)')
    
    # Plot Memory usage
    plt.plot(metrics_df['seconds_elapsed'], metrics_df['memory_percent'], 'r-', 
             linewidth=2, label='Memory Usage (%)')
    
    plt.title('Resource Utilization Timeline')
    plt.xlabel('Time Elapsed (seconds)')
    plt.ylabel('Resource Usage (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('resource_utilization_timeline.png')
    return plt.gcf()

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
    
    plt.title('Batch Size Optimization: Throughput vs Memory Usage')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(title='Model')
    plt.tight_layout()
    
    plt.savefig('batch_size_optimization.png')
    return fig

def plot_parallel_scaling_efficiency():
    """
    Placeholder function for parallel scaling efficiency plot
    
    Note: This would require additional experiments with different numbers of cores/executors,
    which isn't in our current data. This function creates a simulated plot.
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
    
    plt.title('Parallel Scaling Efficiency')
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
        tp = ((model_data['label'] == 1) & (model_data['prediction'] == 1)).sum()
        fp = ((model_data['label'] == 0) & (model_data['prediction'] == 1)).sum()
        tn = ((model_data['label'] == 0) & (model_data['prediction'] == 0)).sum()
        fn = ((model_data['label'] == 1) & (model_data['prediction'] == 0)).sum()
        
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
            # Extract the probability value - PySpark probability is often stored as an array
            # We need to extract the probability of class 1
            if isinstance(model_data['probability'].iloc[0], str) and '[' in model_data['probability'].iloc[0]:
                # If stored as string representation of array
                model_data['prob_class1'] = model_data['probability'].apply(
                    lambda x: float(x.strip('[]').split(',')[1]) if isinstance(x, str) else 0.5
                )
                sns.kdeplot(model_data['prob_class1'], ax=ax3, label=model)
            else:
                # If already a numeric value
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


# Step 10: Main Execution
def main():
    print("Starting the Spark ML Performance Analysis...")
    
    # Initialize Spark with performance configuration
    print("Initializing Spark session...")
    spark = create_spark_session("ChicagoCrimesML", 
                               executor_cores=4,
                               executor_memory="8g",
                               executor_instances=2)
    
    # Initialize performance monitor
    monitor = SparkPerformanceMonitor(spark)
    
    # Load and process data
    print("Loading and processing data...")
    file_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/computeinw/code/Users/bra_er-rahmaouy/Crimes_-_2001_to_Present_20250127.csv"
    data = load_and_preprocess_data(spark, file_path)
    
    # Print data summary
    record_count = data.count()
    print(f"Data loaded successfully. Total records: {record_count}")
    
    # Add label column
    data = data.withColumn("label", when(col("Arrest") == True, 1).otherwise(0))
    
    # Create feature pipeline
    print("Creating and applying feature pipeline...")
    feature_pipeline = create_feature_pipeline(data)
    
    # Apply the feature pipeline
    pipeline_model = feature_pipeline.fit(data)
    data_with_features = pipeline_model.transform(data)
    
    # Create different dataset sizes
    sizes = {"1M": 1_000_000, "4M": 4_000_000, "Full": data_with_features.count()}
    all_results = []
    best_models = {}
    
    # Define models and their parameter grids
    models = {
        "Random Forest": (
            RandomForestClassifier(featuresCol="pca_features", labelCol="label"),
            ParamGridBuilder()
                .addGrid(RandomForestClassifier.maxDepth, [5])
                .addGrid(RandomForestClassifier.numTrees, [50])
                .build()
        ),
        "Decision Tree": (
            DecisionTreeClassifier(featuresCol="pca_features", labelCol="label"),
            ParamGridBuilder()
                .addGrid(DecisionTreeClassifier.maxDepth, [5])
                .build()
        ),
        "Logistic Regression": (
            LogisticRegression(featuresCol="pca_features", labelCol="label"),
            ParamGridBuilder()
                .addGrid(LogisticRegression.regParam, [0.01])
                .addGrid(LogisticRegression.maxIter, [100])
                .build()
        )
    }
    
    # Train and evaluate models for each dataset size
    for size_name, size_value in sizes.items():
        print(f"\nProcessing dataset size: {size_name}")
        data_sample = data_with_features.limit(size_value)
        train_data, test_data = data_sample.randomSplit([0.8, 0.2], seed=42)
        
        for model_name, (model, param_grid) in models.items():
            print(f"\nTraining {model_name}...")
            best_model, complete_metrics = train_and_evaluate_model(
                model, train_data, test_data, model_name, size_name, param_grid, monitor
            )
            all_results.append(complete_metrics)
            
            # Save the best model for each model type
            if size_name == "Full":
                best_models[model_name] = best_model
    
    # Convert results to DataFrame, use pandas directly for better control
    results_df = pd.DataFrame(all_results)
    
    # Print column names and first few rows for debugging
    print(f"Available columns in results_df: {results_df.columns.tolist()}")
    print("Sample data in results_df:")
    print(results_df.head())
    
    # Plot performance metrics
    plot_performance_metrics(results_df)
    
    # Save results
    results_df.to_csv('model_performance_results.csv', index=False)
    
    # Test prediction performance for each model
    prediction_results = []
    detailed_results = []  # List to store detailed prediction results
    
    for model_name, model in best_models.items():
        print(f"\nTesting prediction performance for {model_name}...")
        # Create a complete pipeline for this model
        model_pipeline = create_model_pipeline(pipeline_model, model)
        
        # Save the model pipeline
        save_model(model_pipeline, f"{model_name.replace(' ', '_').lower()}_pipeline")
        
        # Test prediction performance
        model_results = test_prediction_performance(
            spark, model_pipeline, data, model_name, monitor
        )
        prediction_results.extend(model_results)
        
        # Generate detailed prediction results
        print(f"\nGenerating detailed prediction results for {model_name}...")
        detailed_result = save_detailed_prediction_results(
            spark, model_pipeline, data, model_name
        )
        if detailed_result:
            detailed_results.append(detailed_result)
    
    # Ensure all prediction results have the required fields
    for result in prediction_results:
        for field in ['model_name', 'batch_size', 'prediction_time', 'avg_cpu', 'avg_memory']:
            if field not in result:
                result[field] = 0.0
    
    # Visualize prediction performance
    pred_results_df = pd.DataFrame(prediction_results)
    print(f"Available columns in pred_results_df: {pred_results_df.columns.tolist()}")
    plot_prediction_performance(pred_results_df)
    pred_results_df.to_csv('prediction_performance_results.csv', index=False)
    
    # Generate the new enhanced visualizations
    enhanced_visualizations(results_df, pred_results_df, monitor)
    
    # Example of loading and using a saved model for predictions
    print("\nExample of loading a saved model for predictions:")
    model_path = f"models/random_forest_pipeline"
    loaded_model = load_model(model_path)
    
    # Create a small sample for demonstration
    sample_data = data.limit(10)
    
    # Make predictions
    print("Making predictions with the loaded model...")
    predictions = loaded_model.transform(sample_data)
    
    # Show results
    print("\nPrediction Results:")
    predictions.select("Primary Type", "Location Description", "Year", "Hour", "Day", 
                      "prediction", "probability").show()
    
    # Save plots and results - upload all output files to AzureML
    print("\nUploading results to AzureML...")
    from azureml.core import Run
    run = Run.get_context()
    
    # Upload original plots
    run.upload_file("outputs/model_performance_results.png", "model_performance_results.png")
    run.upload_file("outputs/prediction_performance_results.png", "prediction_performance_results.png")
    run.upload_file("outputs/scalability_analysis.png", "scalability_analysis.png")
    run.upload_file("outputs/resource_efficiency.png", "resource_efficiency.png")
    run.upload_file("outputs/training_prediction_tradeoff.png", "training_prediction_tradeoff.png")
    run.upload_file("outputs/resource_utilization_timeline.png", "resource_utilization_timeline.png")
    run.upload_file("outputs/parallel_scaling_efficiency.png", "parallel_scaling_efficiency.png")
    run.upload_file("outputs/batch_size_optimization.png", "batch_size_optimization.png")
    run.upload_file("outputs/detailed_testing_performance.png", "detailed_testing_performance.png")
    run.upload_file("outputs/execution_time_validation.png", "execution_time_validation.png")
    
    # Upload CSV files
    run.upload_file("outputs/model_performance_results.csv", "model_performance_results.csv")
    run.upload_file("outputs/prediction_performance_results.csv", "prediction_performance_results.csv")
    
    # Upload detailed prediction results files
    for model_name in best_models.keys():
        file_name = f"{model_name.replace(' ', '_').lower()}_predictions.csv"
        file_path = os.path.join("results", file_name)
        if os.path.exists(file_path):
            run.upload_file(f"outputs/{file_name}", file_path)
            print(f"Uploaded detailed predictions for {model_name}")
    
    # Upload the summary of detailed results if available
    if detailed_results:
        run.upload_file("outputs/detailed_prediction_summary.csv", "detailed_prediction_summary.csv")
        print("Uploaded detailed prediction summary")
    
    print("All results and visualizations uploaded successfully")

    # Stop Spark session
    spark.stop()

def enhanced_visualizations(results_df, pred_results_df, monitor):
    """Wrapper function to generate all the new visualizations"""
    print("Generating enhanced visualizations...")
    
    # Scalability Analysis
    plot_scalability_analysis(results_df)
    
    # Resource Efficiency
    plot_resource_efficiency(results_df)
    
    # Training-Prediction Tradeoff
    plot_training_prediction_tradeoff(results_df, pred_results_df)
    
    # Resource Utilization Timeline
    # Use the last collected metrics as an example
    if hasattr(monitor, 'system_metrics') and monitor.system_metrics:
        plot_resource_utilization_timeline(monitor.system_metrics)
    else:
        print("No system metrics available for timeline visualization")
    
    # Parallel Scaling Efficiency (simulated)
    plot_parallel_scaling_efficiency()
    
    # Batch Size Optimization
    plot_batch_size_optimization(pred_results_df)

    # Execution time validation function
    plot_execution_time_validation(pred_results_df, model_results_df)

    # Detailed model testing performance
    plot_detailed_testing_performance(detailed_results_dir="results")
    
    print("Enhanced visualizations completed!")

if __name__ == "__main__":
    main()