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
import os

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamIntrusionDetector:
    def __init__(self, model_path: str, expected_features: int = 67):
        try:
            self.model = joblib.load(model_path)
            self.expected_features = expected_features
            self.class_names = self._get_class_names()
            logger.info(f"Modelo {model_path} carregado com sucesso")
            logger.info(f"Esperando {expected_features} features")
            if self.class_names:
                logger.info(f"Classes do modelo: {self.class_names}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def _get_class_names(self):
        """Tenta obter nomes das classes do modelo"""
        try:
            if hasattr(self.model, 'classes_'):
                return self.model.classes_.tolist()
            # Verificar se existe label_encoder salvo
            encoder_path = "models/label_encoder.pkl"
            if os.path.exists(encoder_path):
                encoder = joblib.load(encoder_path)
                return encoder.classes_.tolist()
            return None
        except:
            return None

    def validate_features(self, features: list) -> bool:
        """Valida se as features têm o formato correto"""
        if not isinstance(features, list):
            return False
        if len(features) != self.expected_features:
            logger.warning(f"Features recebidas: {len(features)}, esperadas: {self.expected_features}")
            return False
        return True

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            features = data.get("features", [])
            
            # Validar features
            if not self.validate_features(features):
                return {
                    "error": f"Invalid features format. Expected {self.expected_features}, got {len(features)}",
                    "raw_data": data
                }
            
            # Converter para numpy array
            features_array = np.array(features, dtype=np.float32).reshape(1, -1)
            
            # Fazer predição
            prediction = self.model.predict(features_array)[0]
            
            # Obter probabilidades se disponível
            probability = None
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(features_array)[0]
                confidence = float(probability[prediction])
            else:
                confidence = 1.0  # Valor padrão para modelos sem probabilidade
            
            # Mapear nome da classe se disponível
            prediction_label = str(prediction)
            if self.class_names and 0 <= prediction < len(self.class_names):
                prediction_label = self.class_names[prediction]
            
            result = {
                "id": data.get("id", "unknown"),
                "prediction": int(prediction),
                "prediction_label": prediction_label,
                "confidence": confidence,
                "timestamp": data.get("timestamp", ""),
                "source_ip": data.get("source_ip", ""),
                "destination_ip": data.get("destination_ip", ""),
                "protocol": data.get("protocol", ""),
                "processing_time": np.datetime64('now').astype(str),
                "features_received": len(features)
            }
            
            # Adicionar todas as probabilidades se disponível
            if probability is not None and self.class_names:
                result["probabilities"] = {
                    str(self.class_names[i] if self.class_names else i): float(prob)
                    for i, prob in enumerate(probability)
                }
                
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {
                "error": str(e),
                "raw_data": data,
                "processing_time": np.datetime64('now').astype(str)
            }

def setup_environment():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)  # Reduzido para desenvolvimento
    env.get_config().set_auto_watermark_interval(1000)
    
    # Adicionar dependências do Python (opcional)
    env.get_config().set_python_executable("python")
    return env

def create_kafka_consumer(bootstrap_servers: str = "localhost:9092"):
    return FlinkKafkaConsumer(
        topics='network-traffic',
        deserialization_schema=SimpleStringSchema(),
        properties={
            'bootstrap.servers': bootstrap_servers,
            'group.id': 'flink-ids-consumer',
            'auto.offset.reset': 'latest',  # Mudado para latest
            'enable.auto.commit': 'true',
            'session.timeout.ms': '30000'
        }
    )

def create_kafka_producer(bootstrap_servers: str = "localhost:9092"):
    return FlinkKafkaProducer(
        topic='ids-predictions',
        serialization_schema=SimpleStringSchema(),
        producer_config={
            'bootstrap.servers': bootstrap_servers,
            'acks': '1',
            'linger.ms': '5',
            'compression.type': 'lz4',
            'batch.size': '16384'
        }
    )

def process_stream(detector: StreamIntrusionDetector, line: str) -> str:
    try:
        data = json.loads(line)
        result = detector.predict(data)
        
        # Log apenas para debug (comentar em produção)
        if "error" in result:
            logger.warning(f"Erro no processamento: {result.get('error')}")
        else:
            logger.debug(f"Predição: {result.get('prediction_label')} - Confiança: {result.get('confidence'):.3f}")
            
        return json.dumps(result)
        
    except json.JSONDecodeError:
        error_msg = json.dumps({
            "error": "Invalid JSON format",
            "raw_message": line[:200],  # Log parcial para debug
            "processing_time": np.datetime64('now').astype(str)
        })
        logger.warning("JSON inválido recebido")
        return error_msg
        
    except Exception as e:
        error_msg = json.dumps({
            "error": f"Processing error: {str(e)}",
            "raw_message": line[:200],
            "processing_time": np.datetime64('now').astype(str)
        })
        logger.error(f"Erro no processamento: {e}")
        return error_msg

def main():
    try:
        # Configuração
        env = setup_environment()
        
        # Carregar detector - AJUSTAR CAMINHO CONFORME SEU MODELO
        model_path = "models/best_model_xgb.pkl"  # ou "models/best_model_lgbm.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
            
        detector = StreamIntrusionDetector(model_path, expected_features=67)
        
        # Conectores Kafka
        bootstrap_servers = "localhost:9092"
        kafka_consumer = create_kafka_consumer(bootstrap_servers)
        kafka_producer = create_kafka_producer(bootstrap_servers)
        
        # Pipeline de processamento
        ds = env.add_source(kafka_consumer) \
               .name("kafka_source") \
               .map(lambda x: process_stream(detector, x), 
                    output_type=Types.STRING()) \
               .name("prediction_processor")

        # Enviar para Kafka
        ds.add_sink(kafka_producer) \
          .name("kafka_sink")
        
        # Log para monitoramento (opcional)
        ds.map(lambda x: logger.info(f"Processed message")) \
          .name("metrics_logger") \
          .set_parallelism(1)  # Apenas uma instância para logging
        
        logger.info("Iniciando Flink IDS Streaming Job...")
        logger.info(f"Modelo: {model_path}")
        logger.info(f"Bootstrap servers: {bootstrap_servers}")
        
        env.execute("Network Intrusion Detection Stream Processing")
        
    except Exception as e:
        logger.error(f"Falha na execução do job: {e}")
        raise

if __name__ == "__main__":
    main()