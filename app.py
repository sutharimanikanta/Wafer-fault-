# app.py - Flask Application for Wafer Fault Detection
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import os
import io
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
import json
from file_operations import File_Operation

# Import your existing File_Operation class
from file_operations import File_Operation

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("logs", exist_ok=True)


class WaferFaultDetectionSystem:
    """
    Advanced Wafer Fault Detection System
    Integrates with your existing File_Operation class for model management
    """

    def __init__(self):
        self.file_operations = None
        self.models = {}
        self.scalers = {}
        self.clusterer = None
        self.expected_sensors = 590
        self.initialize_system()

    def initialize_system(self):
        """Initialize the detection system with logging"""
        try:
            # Create log file for this session
            log_file = open(
                f'logs/prediction_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                "w",
            )

            # Initialize file operations with logging
            self.file_operations = File_Operation(log_file, logger)
            # *************************************************************

            # Load available models
            self.load_available_models()

            logger.info("Wafer Fault Detection System initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing system: {str(e)}")
            raise

    def load_available_models(self):
        """Load all available models from the models directory"""
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                logger.warning(
                    "Models directory not found. Creating empty models directory."
                )
                os.makedirs(models_dir)
                return

            # Get list of model files
            model_files = [
                f
                for f in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, f))
            ]

            for model_name in model_files:
                try:
                    # Try to load each model using your File_Operation class
                    model = self.file_operations.load_model(model_name)
                    self.models[model_name] = model
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Could not load model {model_name}: {str(e)}")

            if not self.models:
                logger.warning("No models loaded. System will use simulation mode.")
                self.load_simulation_models()

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.load_simulation_models()

    def load_simulation_models(self):
        """Load simulation models for demo purposes"""
        logger.info("Loading simulation models for demonstration")

        # Create mock models for different clusters
        for i in range(3):  # 3 clusters
            cluster_name = f"cluster_{i}"
            # Simulate a trained model (you'll replace this with actual model loading)
            mock_model = {
                "type": "simulation",
                "cluster": i,
                "weights": np.random.randn(self.expected_sensors) * 0.01,
                "bias": np.random.randn() * 0.1,
                "threshold": 0.5 + np.random.randn() * 0.1,
            }
            self.models[cluster_name] = mock_model

        # Create mock scaler
        self.scalers["standard_scaler"] = {
            "mean": np.random.randn(self.expected_sensors) * 100,
            "std": np.random.randn(self.expected_sensors) * 20 + 30,
        }

    def find_optimal_cluster(self, sensor_data):
        """Find the optimal cluster for the given sensor data"""
        try:
            if len(sensor_data) != self.expected_sensors:
                logger.warning(
                    f"Expected {self.expected_sensors} sensors, got {len(sensor_data)}"
                )
                # Pad or truncate data
                if len(sensor_data) < self.expected_sensors:
                    sensor_data = np.pad(
                        sensor_data,
                        (0, self.expected_sensors - len(sensor_data)),
                        "constant",
                    )
                else:
                    sensor_data = sensor_data[: self.expected_sensors]

            # Use your existing find_correct_model_file logic
            # For simulation, we'll use a simple clustering approach
            cluster_distances = []

            for cluster_name, model in self.models.items():
                if model.get("type") == "simulation":
                    # Simple distance-based clustering
                    distance = (
                        np.random.random()
                    )  # Replace with actual clustering logic
                    cluster_distances.append((cluster_name, distance))

            if cluster_distances:
                # Return the cluster with minimum distance
                best_cluster = min(cluster_distances, key=lambda x: x[1])[0]
                return best_cluster

            return list(self.models.keys())[0] if self.models else "cluster_0"

        except Exception as e:
            logger.error(f"Error finding optimal cluster: {str(e)}")
            return "cluster_0"

    def preprocess_data(self, sensor_data):
        """Preprocess sensor data using loaded scalers"""
        try:
            if "standard_scaler" in self.scalers:
                scaler = self.scalers["standard_scaler"]
                normalized_data = (
                    sensor_data - scaler["mean"][: len(sensor_data)]
                ) / scaler["std"][: len(sensor_data)]
                return normalized_data
            else:
                # Simple standardization if no scaler available
                return (sensor_data - np.mean(sensor_data)) / (
                    np.std(sensor_data) + 1e-8
                )
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return sensor_data

    def predict_wafer_quality(self, sensor_data, wafer_id=None):
        """
        Predict wafer quality using the loaded models
        Returns prediction results with confidence scores
        """
        try:
            # Convert to numpy array
            sensor_data = np.array(sensor_data, dtype=float)

            # Find optimal cluster
            cluster_name = self.find_optimal_cluster(sensor_data)

            # Get the appropriate model
            model = self.models.get(cluster_name)
            if not model:
                raise ValueError(f"Model for cluster {cluster_name} not found")

            # Preprocess data
            processed_data = self.preprocess_data(sensor_data)

            # Make prediction
            if model.get("type") == "simulation":
                # Simulation prediction
                score = (
                    np.dot(processed_data[: len(model["weights"])], model["weights"])
                    + model["bias"]
                )
                probability = 1 / (1 + np.exp(-score))  # Sigmoid
                prediction = "Faulty" if probability > model["threshold"] else "Good"
                confidence = abs(probability - 0.5) * 2
            else:
                # Use actual model prediction (replace with your model's predict method)
                prediction_proba = model.predict_proba([processed_data])[0]
                probability = prediction_proba[1]  # Assuming binary classification
                prediction = "Faulty" if probability > 0.5 else "Good"
                confidence = max(prediction_proba)

            # Calculate additional metrics
            anomaly_score = abs(np.mean(processed_data))
            quality_score = (
                (1 - probability) * 100 if prediction == "Good" else probability * 100
            )

            result = {
                "wafer_id": wafer_id or f"wafer_{datetime.now().strftime('%H%M%S')}",
                "prediction": prediction,
                "probability": float(probability),
                "confidence": float(min(0.99, max(0.6, confidence))),
                "cluster": cluster_name,
                "anomaly_score": float(anomaly_score),
                "quality_score": float(quality_score),
                "sensor_count": len(sensor_data),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Prediction completed for {result['wafer_id']}: {prediction}")
            return result

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise


# Initialize the detection system
detection_system = WaferFaultDetectionSystem()


@app.route("/")
def index():
    """Main application page"""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict_single():
    """API endpoint for single wafer prediction"""
    try:
        data = request.json
        sensor_data = data.get("sensor_data", [])
        wafer_id = data.get("wafer_id")

        if not sensor_data:
            return jsonify({"error": "No sensor data provided"}), 400

        if len(sensor_data) < 100:
            return (
                jsonify(
                    {
                        "error": f"Insufficient sensor data. Expected at least 100 sensors, got {len(sensor_data)}"
                    }
                ),
                400,
            )

        result = detection_system.predict_wafer_quality(sensor_data, wafer_id)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_batch", methods=["POST"])
def predict_batch():
    """API endpoint for batch wafer prediction"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "File must be CSV format"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Read and process CSV
        df = pd.read_csv(filepath)

        # Validate CSV structure
        sensor_columns = [col for col in df.columns if col.startswith("Sensor-")]
        if len(sensor_columns) < 100:
            return (
                jsonify(
                    {
                        "error": f"CSV must contain at least 100 sensor columns, found {len(sensor_columns)}"
                    }
                ),
                400,
            )

        # Process each row
        results = []
        for idx, row in df.iterrows():
            try:
                sensor_data = [row[col] for col in sensor_columns if pd.notna(row[col])]
                wafer_id = row.get("Wafer", f"wafer_{idx+1}")

                result = detection_system.predict_wafer_quality(sensor_data, wafer_id)
                results.append(result)

            except Exception as e:
                logger.warning(f"Error processing row {idx}: {str(e)}")
                continue

        # Clean up uploaded file
        os.remove(filepath)

        # Calculate batch statistics
        total_wafers = len(results)
        faulty_count = sum(1 for r in results if r["prediction"] == "Faulty")
        good_count = total_wafers - faulty_count
        avg_confidence = np.mean([r["confidence"] for r in results]) if results else 0

        batch_stats = {
            "total_wafers": total_wafers,
            "good_wafers": good_count,
            "faulty_wafers": faulty_count,
            "fault_rate": (
                (faulty_count / total_wafers * 100) if total_wafers > 0 else 0
            ),
            "average_confidence": float(avg_confidence),
            "processed_at": datetime.now().isoformat(),
        }

        return jsonify({"results": results, "batch_statistics": batch_stats})

    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/system_status")
def system_status():
    """Get system status and loaded models"""
    try:
        status = {
            "status": "operational",
            "models_loaded": len(detection_system.models),
            "model_names": list(detection_system.models.keys()),
            "expected_sensors": detection_system.expected_sensors,
            "system_initialized": detection_system.file_operations is not None,
            "timestamp": datetime.now().isoformat(),
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate_sample")
def generate_sample():
    """Generate sample CSV file for testing"""
    try:
        # Create sample data
        num_samples = 10
        sensor_cols = [f"Sensor-{i}" for i in range(1, 591)]

        # Generate realistic sensor data
        data = []
        for i in range(num_samples):
            row = {"Wafer": f"Wafer-{i+1}"}
            for sensor in sensor_cols:
                # Generate realistic semiconductor sensor values
                base_value = np.random.normal(100, 50)
                noise = np.random.normal(0, 5)
                row[sensor] = round(base_value + noise, 4)
            data.append(row)

        df = pd.DataFrame(data)

        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Convert to BytesIO for file download
        csv_data = io.BytesIO()
        csv_data.write(output.getvalue().encode("utf-8"))
        csv_data.seek(0)

        return send_file(
            csv_data,
            mimetype="text/csv",
            as_attachment=True,
            download_name="sample_wafer_data.csv",
        )

    except Exception as e:
        logger.error(f"Error generating sample: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 50MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
