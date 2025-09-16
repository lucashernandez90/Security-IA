from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema, JsonRowSerializationSchema
from pyflink.common.typeinfo import Types
from pyflink.common import Row
import json
import joblib
import numpy as np
import logging
from typing import Dict, Any

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamIntrusionDetector:
    def __init__(self, model_path: str):
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Modelo {model_path} carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            features = data.get("features", [])
            if not features:
                return {"error": "No features provided", "raw": data}
            
            features_array = np.array(features).reshape(1, -1)
            prediction = self.model.predict(features_array)[0]
            probability = self.model.predict_proba(features_array)[0] if hasattr(self.model, 'predict_proba') else None
            
            result = {
                "prediction": int(prediction),
                "probability": float(probability[prediction]) if probability is not None else None,
                "features": features,
                "timestamp": data.get("timestamp", np.datetime64('now').astype(str))
            }
            
            # Adicionar metadados do modelo se disponível
            if hasattr(self.model, 'classes_'):
                result["class_labels"] = self.model.classes_.tolist()
                
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {"error": str(e), "raw": data}

def setup_environment():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(3)  # Paralelismo otimizado
    env.get_config().set_auto_watermark_interval(1000)
    return env

def create_kafka_consumer():
    return FlinkKafkaConsumer(
        topics='network-traffic',
        deserialization_schema=SimpleStringSchema(),
        properties={
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'flink-ids-consumer',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': 'true'
        }
    )

def create_kafka_producer():
    return FlinkKafkaProducer(
        topic='ids-predictions',
        serialization_schema=SimpleStringSchema(),
        producer_config={
            'bootstrap.servers': 'localhost:9092',
            'acks': '1',
            'linger.ms': '5',
            'compression.type': 'lz4'
        }
    )

def process_stream(detector: StreamIntrusionDetector, line: str) -> str:
    try:
        data = json.loads(line)
        result = detector.predict(data)
        return json.dumps(result)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON", "raw": line})
    except Exception as e:
        return json.dumps({"error": f"Processing error: {str(e)}", "raw": line})

def main():
    try:
        #Configuração
        env = setup_environment()
        
        # Carregar detector
        detector = StreamIntrusionDetector("models/xgb_model.pkl")
        
        #Conectores Kafka
        kafka_consumer = create_kafka_consumer()
        kafka_producer = create_kafka_producer()
        
        #Pipeline
        ds = env.add_source(kafka_consumer) \
               .name("kafka_source") \
               .map(lambda x: process_stream(detector, x), 
                    output_type=Types.STRING()) \
               .name("prediction_processor")

        ds.add_sink(kafka_producer) \
          .name("kafka_sink")
        
        ds.map(lambda x: logger.info(f"Processed: {x}")) \
          .name("metrics_logger")
        
        logger.info("Iniciando Flink IDS Streaming Job...")
        env.execute("Flink IDS Streaming Job - Enhanced")
        
    except Exception as e:
        logger.error(f"Falha na execução do job: {e}")
        raise

if __name__ == "__main__":
    main()