"""Синтетический генератор сетевых логов в формате LANL netflow.

Модель: у каждого хоста есть скрытая «роль» (workstation, web_server, ...),
которая определяет его сетевое поведение — с кем, по каким портам, в каком
объёме и в какое время он общается. Роль используется ТОЛЬКО для генерации
и для последующей валидации качества (ground truth). В признаки и embedding
она не попадает.

Генерируется несколько временных окон (дней). В поздние окна опционально
внедряются аномалии (сканирование/латеральное перемещение и смена роли),
чтобы проверить детектор переходов между кластерами.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    NETFLOW_COLUMNS,
    PROTO_TCP,
    PROTO_UDP,
    RANDOM_SEED,
)

# Сколько хостов каждой роли создаётся в сети.
ROLE_COUNTS = {
    "workstation": 80,
    "web_server": 6,
    "db_server": 4,
    "dns_server": 2,
    "domain_controller": 2,
    "file_server": 3,
    "mail_server": 2,
    "proxy": 2,
}

# Порты (и протокол), которые роль ОБСЛУЖИВАЕТ как сервер.
ROLE_SERVES = {
    "workstation": [],
    "web_server": [(80, PROTO_TCP), (443, PROTO_TCP)],
    "db_server": [(3306, PROTO_TCP), (5432, PROTO_TCP)],
    "dns_server": [(53, PROTO_UDP)],
    "domain_controller": [(88, PROTO_TCP), (389, PROTO_TCP), (636, PROTO_TCP),
                          (135, PROTO_TCP), (445, PROTO_TCP)],
    "file_server": [(445, PROTO_TCP), (139, PROTO_TCP)],
    "mail_server": [(25, PROTO_TCP), (143, PROTO_TCP), (587, PROTO_TCP), (993, PROTO_TCP)],
    "proxy": [(3128, PROTO_TCP), (8080, PROTO_TCP)],
}

# Исходящие взаимодействия роли как клиента:
#   (целевая_роль, порт, протокол, среднее_число_потоков_за_окно)
ROLE_CONNECTS = {
    "workstation": [
        ("web_server", 443, PROTO_TCP, 18),
        ("web_server", 80, PROTO_TCP, 6),
        ("dns_server", 53, PROTO_UDP, 25),
        ("domain_controller", 88, PROTO_TCP, 5),
        ("domain_controller", 389, PROTO_TCP, 4),
        ("domain_controller", 445, PROTO_TCP, 3),
        ("file_server", 445, PROTO_TCP, 8),
        ("mail_server", 993, PROTO_TCP, 4),
        ("mail_server", 587, PROTO_TCP, 2),
        ("proxy", 3128, PROTO_TCP, 10),
    ],
    "web_server": [
        ("db_server", 3306, PROTO_TCP, 40),
        ("db_server", 5432, PROTO_TCP, 20),
        ("dns_server", 53, PROTO_UDP, 15),
        ("domain_controller", 389, PROTO_TCP, 4),
    ],
    "db_server": [
        ("dns_server", 53, PROTO_UDP, 6),
        ("domain_controller", 389, PROTO_TCP, 2),
    ],
    "dns_server": [
        ("domain_controller", 389, PROTO_TCP, 2),
    ],
    "domain_controller": [
        ("dns_server", 53, PROTO_UDP, 8),
        ("domain_controller", 445, PROTO_TCP, 6),  # репликация между DC
    ],
    "file_server": [
        ("dns_server", 53, PROTO_UDP, 6),
        ("domain_controller", 88, PROTO_TCP, 3),
    ],
    "mail_server": [
        ("dns_server", 53, PROTO_UDP, 10),
        ("domain_controller", 389, PROTO_TCP, 3),
        ("__external__", 25, PROTO_TCP, 30),       # исходящая почта наружу
    ],
    "proxy": [
        ("__external__", 443, PROTO_TCP, 120),     # выход в интернет
        ("__external__", 80, PROTO_TCP, 60),
        ("dns_server", 53, PROTO_UDP, 20),
    ],
}

N_EXTERNAL = 40  # пул «внешних» устройств (интернет)


@dataclass
class Host:
    name: str
    role: str


class _VolumeModel:
    """Лог-нормальные объёмы трафика, зависящие от порта/сервиса."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def sample(self, port: int, n: int) -> dict[str, np.ndarray]:
        rng = self.rng
        # Базовые профили по типу сервиса.
        if port in (53,):                       # DNS — мелкие пакеты
            src_b = rng.lognormal(4.2, 0.4, n)
            dst_b = rng.lognormal(5.0, 0.5, n)
            src_p = rng.lognormal(0.3, 0.3, n) + 1
            dst_p = rng.lognormal(0.4, 0.3, n) + 1
            dur = rng.exponential(0.2, n)
        elif port in (445, 139, 3306, 5432):    # файлы/БД — крупные объёмы
            src_b = rng.lognormal(7.5, 1.0, n)
            dst_b = rng.lognormal(9.5, 1.2, n)
            src_p = rng.lognormal(3.0, 0.8, n) + 1
            dst_p = rng.lognormal(4.5, 0.9, n) + 1
            dur = rng.exponential(12.0, n)
        elif port in (80, 443, 8080, 3128):     # web — средние, dst > src
            src_b = rng.lognormal(6.5, 0.9, n)
            dst_b = rng.lognormal(8.8, 1.1, n)
            src_p = rng.lognormal(2.0, 0.7, n) + 1
            dst_p = rng.lognormal(3.6, 0.8, n) + 1
            dur = rng.exponential(5.0, n)
        else:                                    # прочее (auth, mail, rdp)
            src_b = rng.lognormal(6.0, 0.8, n)
            dst_b = rng.lognormal(6.5, 0.9, n)
            src_p = rng.lognormal(1.5, 0.6, n) + 1
            dst_p = rng.lognormal(1.8, 0.7, n) + 1
            dur = rng.exponential(3.0, n)
        return {
            "src_bytes": np.round(src_b).astype(np.int64),
            "dst_bytes": np.round(dst_b).astype(np.int64),
            "src_packets": np.round(src_p).astype(np.int64),
            "dst_packets": np.round(dst_p).astype(np.int64),
            "duration": np.round(dur, 3),
        }


def _build_inventory(rng: np.random.Generator) -> list[Host]:
    hosts: list[Host] = []
    counters = {role: 0 for role in ROLE_COUNTS}
    prefix = {
        "workstation": "Comp",
        "web_server": "Web",
        "db_server": "DB",
        "dns_server": "DNS",
        "domain_controller": "DC",
        "file_server": "FS",
        "mail_server": "Mail",
        "proxy": "Proxy",
    }
    for role, count in ROLE_COUNTS.items():
        for _ in range(count):
            counters[role] += 1
            hosts.append(Host(f"{prefix[role]}{counters[role]:04d}", role))
    rng.shuffle(hosts)
    return hosts


def _sample_hours(rng: np.random.Generator, n: int, role: str) -> np.ndarray:
    """Час суток для потоков: рабочие станции — в рабочее время, серверы — круглосуточно."""
    if n <= 0:
        return np.array([], dtype=float)
    if role == "workstation":
        hours = np.clip(rng.normal(13.0, 2.8, n), 0, 23.999)
    else:
        # Круглосуточно + лёгкий дневной бугор.
        base = rng.uniform(0, 24, n)
        bump = rng.normal(13.0, 3.0, int(n * 0.3))
        hours = np.concatenate([base, np.clip(bump, 0, 23.999)])[:n]
        if hours.shape[0] < n:  # добить при нехватке
            hours = np.concatenate([hours, rng.uniform(0, 24, n - hours.shape[0])])
    return hours


def _emit(rows, src, dst, proto, dport, sport, t, vol_i):
    rows.append((
        float(t), float(vol_i["duration"]), src, dst, int(proto),
        int(sport), int(dport),
        int(vol_i["src_packets"]), int(vol_i["dst_packets"]),
        int(vol_i["src_bytes"]), int(vol_i["dst_bytes"]),
    ))


def _generate_window(
    hosts: list[Host],
    externals: list[str],
    window_idx: int,
    rng: np.random.Generator,
    vmodel: _VolumeModel,
    anomalies: dict[str, dict] | None,
) -> pd.DataFrame:
    by_role: dict[str, list[str]] = {}
    for h in hosts:
        by_role.setdefault(h.role, []).append(h.name)

    rows: list[tuple] = []
    day_start = window_idx * 86400.0

    def target_pool(role_key: str) -> list[str]:
        if role_key == "__external__":
            return externals
        return by_role.get(role_key, [])

    # --- Нормальное поведение: перебор клиентов по их ролям ---
    for h in hosts:
        for (troles, port, proto, rate) in ROLE_CONNECTS.get(h.role, []):
            pool = target_pool(troles)
            if not pool:
                continue
            n = int(rng.poisson(rate))
            if n == 0:
                continue
            targets = rng.choice(pool, size=n, replace=True)
            vol = vmodel.sample(port, n)
            hours = _sample_hours(rng, n, h.role)
            sports = rng.integers(1024, 65535, n)
            for i in range(n):
                if targets[i] == h.name:
                    continue
                t = day_start + hours[i] * 3600.0
                vi = {k: v[i] for k, v in vol.items()}
                _emit(rows, h.name, targets[i], proto, port, sports[i], t, vi)

    # --- Внедрение аномалий (с определённого окна) ---
    if anomalies:
        all_internal = [h.name for h in hosts]
        for dev, spec in anomalies.items():
            if window_idx < spec["start_window"]:
                continue
            if spec["type"] == "scan":
                # Фан-аут «взрыв»: обращения ко множеству хостов по опасным портам.
                n_targets = rng.integers(120, 220)
                scan_ports = [445, 139, 3389, 22, 135]
                targets = rng.choice(all_internal, size=int(n_targets), replace=True)
                for tgt in targets:
                    if tgt == dev:
                        continue
                    port = int(rng.choice(scan_ports))
                    vol = vmodel.sample(port, 1)
                    vi = {k: v[0] for k, v in vol.items()}
                    # короткие "пробы"
                    vi["dst_bytes"] = int(vi["dst_bytes"] * 0.05) + 1
                    t = day_start + rng.uniform(0, 24) * 3600.0
                    _emit(rows, dev, tgt, PROTO_TCP, port, int(rng.integers(1024, 65535)), t, vi)
            elif spec["type"] == "role_shift":
                # Рабочая станция превращается в RDP-плацдарм (jump host):
                #  * к ней массово подключаются по RDP (она ОБСЛУЖИВАЕТ 3389);
                #  * и сама инициирует RDP/SMB к серверам (пивотинг по сети).
                clients = rng.choice(by_role.get("workstation", []), size=90, replace=True)
                for cl in clients:
                    if cl == dev:
                        continue
                    vol = vmodel.sample(3389, 1)
                    vi = {k: v[0] for k, v in vol.items()}
                    t = day_start + rng.uniform(0, 24) * 3600.0
                    _emit(rows, cl, dev, PROTO_TCP, 3389, int(rng.integers(1024, 65535)), t, vi)
                pivot_targets = list(by_role.get("file_server", [])) + \
                    list(by_role.get("db_server", [])) + list(by_role.get("domain_controller", []))
                for _ in range(40):
                    tgt = rng.choice(pivot_targets) if pivot_targets else dev
                    port = int(rng.choice([3389, 445, 22]))
                    vol = vmodel.sample(port, 1)
                    vi = {k: v[0] for k, v in vol.items()}
                    t = day_start + rng.uniform(0, 24) * 3600.0
                    _emit(rows, dev, tgt, PROTO_TCP, port, int(rng.integers(1024, 65535)), t, vi)

    df = pd.DataFrame(rows, columns=NETFLOW_COLUMNS)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def generate_dataset(
    out_dir: str,
    n_windows: int = 5,
    seed: int = RANDOM_SEED,
    inject_anomalies: bool = True,
) -> dict:
    """Сгенерировать датасет: по одному CSV на окно + разметка ролей и аномалий.

    Возвращает meta-словарь с путями и параметрами.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    vmodel = _VolumeModel(rng)

    hosts = _build_inventory(rng)
    externals = [f"Ext{i:04d}" for i in range(N_EXTERNAL)]

    anomalies: dict[str, dict] = {}
    if inject_anomalies and n_windows >= 3:
        workstations = [h.name for h in hosts if h.role == "workstation"]
        chosen = rng.choice(workstations, size=3, replace=False)
        start = max(2, n_windows // 2)
        anomalies[chosen[0]] = {"type": "scan", "start_window": start}
        anomalies[chosen[1]] = {"type": "scan", "start_window": start + 1 if start + 1 < n_windows else start}
        anomalies[chosen[2]] = {"type": "role_shift", "start_window": start}

    window_files = []
    for w in range(n_windows):
        df = _generate_window(hosts, externals, w, rng, vmodel, anomalies)
        path = os.path.join(out_dir, f"netflow_window_{w:02d}.csv")
        df.to_csv(path, index=False)
        window_files.append(os.path.basename(path))

    # Разметка ролей (ground truth, только для валидации).
    labels = pd.DataFrame([(h.name, h.role) for h in hosts], columns=["device", "role"])
    labels.to_csv(os.path.join(out_dir, "labels.csv"), index=False)

    anomalies_df = pd.DataFrame(
        [(d, s["type"], s["start_window"]) for d, s in anomalies.items()],
        columns=["device", "type", "start_window"],
    )
    anomalies_df.to_csv(os.path.join(out_dir, "anomalies_truth.csv"), index=False)

    meta = {
        "n_windows": n_windows,
        "seed": seed,
        "n_internal_hosts": len(hosts),
        "n_external_hosts": len(externals),
        "window_files": window_files,
        "roles": ROLE_COUNTS,
        "netflow_columns": NETFLOW_COLUMNS,
        "injected_anomalies": anomalies_df.to_dict(orient="records"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Генерация синтетических netflow-данных (формат LANL).")
    p.add_argument("--out", default="data", help="каталог для данных")
    p.add_argument("--windows", type=int, default=5, help="число временных окон (дней)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--no-anomalies", action="store_true")
    args = p.parse_args()
    meta = generate_dataset(args.out, args.windows, args.seed, not args.no_anomalies)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
