from kafka import KafkaProducer
import json
import time
import numpy as np
from datetime import datetime
import logging
import argparse

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkTrafficProducer:
    def __init__(self, bootstrap_servers: str):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=1,
            compression_type='gzip',
            linger_ms=5
        )
        logger.info(f"Producer conectado ao Kafka: {bootstrap_servers}")

    def generate_sample_features(self, num_features: int = 78) -> list:
        """Gera features de exemplo baseadas no CICIDS2017"""
        # Valores típicos do dataset
        return [
            float(np.random.uniform(0, 1000)),  # Duração
            float(np.random.uniform(0, 10000)), # Protocolo
            float(np.random.uniform(0, 1000)),  # Packets
            float(np.random.uniform(0, 1000000)), # Bytes
            *[float(np.random.uniform(0, 1)) for _ in range(num_features - 4)]
        ]

    def send_message(self, topic: str, message: dict):
        try:
            future = self.producer.send(topic, value=message)
            future.get(timeout=10) 
            logger.debug(f"Mensagem enviada: {message['id']}")
        except Exception as e:
            logger.error(f"Erro ao enviar mensagem: {e}")

    def close(self):
        self.producer.flush()
        self.producer.close()
        logger.info("Producer fechado")

def main():
    parser = argparse.ArgumentParser(description='Kafka Producer para tráfego de rede')
    parser.add_argument('--bootstrap-servers', default='localhost:9092')
    parser.add_argument('--topic', default='network-traffic')
    parser.add_argument('--num-messages', type=int, default=100)
    parser.add_argument('--delay', type=float, default=0.1)
    args = parser.parse_args()

    producer = NetworkTrafficProducer(args.bootstrap_servers)

    try:
        for i in range(args.num_messages):
            message = {
                "id": i,
                "timestamp": datetime.utcnow().isoformat(),
                "features": producer.generate_sample_features(),
                "source_ip": f"192.168.1.{np.random.randint(1, 255)}",
                "destination_ip": f"10.0.0.{np.random.randint(1, 255)}",
                "protocol": np.random.choice(["TCP", "UDP", "ICMP"])
            }
            
            producer.send_message(args.topic, message)
            
            if i % 10 == 0:
                logger.info(f"Enviadas {i} mensagens")
            
            time.sleep(args.delay)
            
    except KeyboardInterrupt:
        logger.info("Interrupção recebida")
    except Exception as e:
        logger.error(f"Erro no producer: {e}")
    finally:
        producer.close()

if __name__ == "__main__":
    main()