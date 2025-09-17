from kafka import KafkaProducer
import json
import time
import numpy as np
from datetime import datetime
import logging
import argparse
import os

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkTrafficProducer:
    def __init__(self, bootstrap_servers: str):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=1,
            compression_type='gzip',
            linger_ms=5,
            retries=3
        )
        self.messages_sent = 0
        self.errors = 0
        logger.info(f"Producer conectado ao Kafka: {bootstrap_servers}")

    def generate_sample_features(self, num_features: int = 67) -> list:
        """Gera features de exemplo baseadas no CICIDS2017 com valores realistas"""
        # Valores típicos do dataset CICIDS2017
        features = [
            float(np.random.uniform(0, 1000)),       # Flow Duration
            float(np.random.uniform(0, 65535)),      # Destination Port
            float(np.random.uniform(0, 1000)),       # Total Fwd Packets
            float(np.random.uniform(0, 1000)),       # Total Backward Packets
            float(np.random.uniform(0, 1000000)),    # Total Length of Fwd Packets
            float(np.random.uniform(0, 1000000)),    # Total Length of Bwd Packets
            float(np.random.uniform(0, 1500)),       # Fwd Packet Length Max
            float(np.random.uniform(0, 1500)),       # Fwd Packet Length Min
            float(np.random.uniform(0, 1500)),       # Fwd Packet Length Mean
            float(np.random.uniform(0, 500)),        # Fwd Packet Length Std
            float(np.random.uniform(0, 1500)),       # Bwd Packet Length Max
            float(np.random.uniform(0, 1500)),       # Bwd Packet Length Min
            float(np.random.uniform(0, 1500)),       # Bwd Packet Length Mean
            float(np.random.uniform(0, 500)),        # Bwd Packet Length Std
            float(np.random.uniform(0, 1e9)),        # Flow Bytes/s
            float(np.random.uniform(0, 1e6)),        # Flow Packets/s
            float(np.random.uniform(0, 10000)),      # Flow IAT Mean
            float(np.random.uniform(0, 10000)),      # Flow IAT Std
            float(np.random.uniform(0, 100000)),     # Flow IAT Max
            float(np.random.uniform(0, 1000)),       # Flow IAT Min
        ]
        
        # Completar com valores aleatórios para as demais features
        while len(features) < num_features:
            # Gerar valores com distribuições mais realistas
            if np.random.random() < 0.3:  # 30% chance de valor zero
                features.append(0.0)
            else:
                features.append(float(np.random.uniform(0, 1)))
        
        return features[:num_features]

    def generate_from_real_data(self, data_path: str = None):
        """Gera features a partir de dados reais se disponível"""
        if data_path and os.path.exists(data_path):
            try:
                # Carregar amostra real para maior realismo
                real_data = np.load(data_path)
                if len(real_data) > 0:
                    sample = real_data[np.random.randint(0, len(real_data))]
                    return sample.tolist()
            except Exception as e:
                logger.warning(f"Não foi possível carregar dados reais: {e}")
        
        # Fallback para geração sintética
        return self.generate_sample_features()

    def send_message(self, topic: str, message: dict):
        try:
            future = self.producer.send(topic, value=message)
            future.get(timeout=10) 
            self.messages_sent += 1
            if self.messages_sent % 100 == 0:
                logger.info(f"Mensagens enviadas: {self.messages_sent}")
            return True
        except Exception as e:
            self.errors += 1
            logger.error(f"Erro ao enviar mensagem {message.get('id', 'N/A')}: {e}")
            return False

    def close(self):
        self.producer.flush()
        self.producer.close()
        logger.info(f"Producer fechado. Estatísticas: {self.messages_sent} mensagens enviadas, {self.errors} erros")

def main():
    parser = argparse.ArgumentParser(description='Kafka Producer para tráfego de rede')
    parser.add_argument('--bootstrap-servers', default='localhost:9092', help='Endereços dos brokers Kafka')
    parser.add_argument('--topic', default='network-traffic', help='Tópico Kafka')
    parser.add_argument('--num-messages', type=int, default=100, help='Número total de mensagens')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay entre mensagens em segundos')
    parser.add_argument('--real-data', type=str, help='Caminho para arquivo .npy com dados reais')
    parser.add_argument('--batch-size', type=int, default=10, help='Tamanho do lote para logging')
    args = parser.parse_args()

    producer = NetworkTrafficProducer(args.bootstrap_servers)

    try:
        start_time = time.time()
        
        for i in range(args.num_messages):
            # Gerar features (com dados reais se disponível)
            if args.real_data:
                features = producer.generate_from_real_data(args.real_data)
            else:
                features = producer.generate_sample_features()
            
            message = {
                "id": i,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "features": features,
                "source_ip": f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                "destination_ip": f"10.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}",
                "destination_port": int(np.random.uniform(0, 65535)),
                "protocol": np.random.choice(["TCP", "UDP", "ICMP"]),
                "prediction": None,  # Para ser preenchido pelo consumer
                "confidence": None   # Para ser preenchido pelo consumer
            }
            
            success = producer.send_message(args.topic, message)
            
            if not success and args.delay > 0:
                # Aumentar delay temporariamente em caso de erro
                time.sleep(args.delay * 2)
            else:
                time.sleep(args.delay)
            
            # Log a cada batch
            if i % args.batch_size == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(f"Progresso: {i}/{args.num_messages} mensagens | Rate: {rate:.2f} msg/s")
                
    except KeyboardInterrupt:
        logger.info("Interrupção recebida pelo usuário")
    except Exception as e:
        logger.error(f"Erro no producer: {e}")
    finally:
        producer.close()
        
        # Estatísticas finais
        total_time = time.time() - start_time
        logger.info(f"Tempo total: {total_time:.2f}s")
        logger.info(f"Taxa média: {args.num_messages/total_time:.2f} msg/s" if total_time > 0 else "N/A")

if __name__ == "__main__":
    main()