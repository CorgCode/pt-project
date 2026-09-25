"""Общие константы и параметры пайплайна."""
from __future__ import annotations

from dataclasses import dataclass, field

RANDOM_SEED = 42

# Схема netflow-записи в стиле LANL Unified Host and Network Dataset.
# См. https://csr.lanl.gov/data/2017/ (NetFlow: Time, Duration, SrcDevice, ...)
NETFLOW_COLUMNS = [
    "time",          # epoch-секунды от начала наблюдения
    "duration",      # длительность потока, сек
    "src_device",    # инициатор соединения
    "dst_device",    # получатель
    "protocol",      # номер протокола (6=TCP, 17=UDP, 1=ICMP)
    "src_port",
    "dst_port",
    "src_packets",
    "dst_packets",
    "src_bytes",
    "dst_bytes",
]

PROTO_TCP, PROTO_UDP, PROTO_ICMP = 6, 17, 1

# Курируемый список сервисных портов. Индикаторы по этим портам входят в вектор
# признаков хоста (и как у "сервера", и как у "клиента"). Это сетевые признаки —
# роль хоста в разметке НЕ используется при построении embedding.
SERVICE_PORTS: dict[int, str] = {
    22: "ssh",
    25: "smtp",
    53: "dns",
    80: "http",
    88: "kerberos",
    135: "rpc",
    139: "netbios",
    143: "imap",
    389: "ldap",
    443: "https",
    445: "smb",
    587: "smtp-sub",
    636: "ldaps",
    993: "imaps",
    3128: "proxy",
    3306: "mysql",
    3389: "rdp",
    5432: "postgres",
    8080: "http-alt",
}

# Границы "рабочего дня" для временны́х признаков.
BUSINESS_HOURS = (8, 19)


@dataclass
class PipelineConfig:
    """Параметры сквозного запуска пайплайна."""

    data_dir: str = "data"
    artifacts_dir: str = "artifacts"

    # embedding
    embed_dim: int = 16          # размерность графового и итогового представления
    pca_dim: int = 16            # размерность признакового представления после PCA
    graph_walk_len: int = 4      # длина "окна" в NetMF-аппроксимации DeepWalk
    methods: tuple[str, ...] = ("features", "graph", "combined")

    # кластеризация
    k_min: int = 4
    k_max: int = 14

    # поиск похожих
    n_neighbors: int = 8

    # аномалии
    anomaly_top_n: int = 15
    anomaly_z_threshold: float = 3.0

    seed: int = RANDOM_SEED
