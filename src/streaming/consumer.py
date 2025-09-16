from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer
from pyflink.common.serialization import JsonRowDeserializationSchema
from pyflink.common import Row
import pickle
import numpy as np
import json
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntrusionDetectionModel:
    def __init__(self, model_path: str):
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Modelo carregado com sucesso: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def predict(self, features: list) -> dict:
        try:
            X = np.array(features).reshape(1, -1)
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
            
            return {
                "prediction": int(y_pred[0]),
                "probability": float(y_prob[0][y_pred[0]]) if y_prob is not None else None,
                "features": features
            }
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {"error": str(e), "features": features}

def setup_environment():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)  # Aumentar paralelismo para melhor performance
    return env

def create_kafka_consumer():
    return FlinkKafkaConsumer(
        topics="network_traffic",
        deserialization_schema=JsonRowDeserializationSchema(),
        properties={
            "bootstrap.servers": "localhost:9092",
            "group.id": "flink-intrusion-detection",
            "auto.offset.reset": "latest"
        }
    )

def main():
    try:
        env = setup_environment()
        
        model = IntrusionDetectionModel("models/xgboost_model.pkl")
        
        consumer = create_kafka_consumer()
        
        ds = env.add_source(consumer)
        
        def process_record(record: Row):
            try:
                features = record.as_dict().get("features", [])
                if not features:
                    return json.dumps({"error": "No features provided"})
                
                result = model.predict(features)
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e), "raw_data": str(record)})
        
        #Pipeline
        ds.map(process_record) \
          .name("intrusion_detection_processor") \
          .print()
        
        logger.info("Iniciando job de detecção de intrusão...")
        env.execute("Real-time Intrusion Detection Stream Job")
        
    except Exception as e:
        logger.error(f"Erro fatal no consumer: {e}")
        raise

if __name__ == "__main__":
    main()