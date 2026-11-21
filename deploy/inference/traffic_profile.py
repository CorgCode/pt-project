"""Deterministic hostquery flow profiles, with no learned checkpoint."""

from math import hypot

from hostquery.features import N_BASE, N_FEATURES
from hostquery.store import dataset_from_csv


class TrafficProfile:
    def __init__(self, method, version, dimension):
        self.metadata = {"method": method, "model_version": version, "dimensions": dimension}

    def preprocess(self, csv):
        data = dataset_from_csv(csv)
        if data.kind != "flows":
            raise ValueError("Expected flow records with src and dst")
        if any(len(host) > 500 for host in data.ids):
            raise ValueError("Host identifier exceeds 500 characters")
        return data

    def infer(self, prepared):
        vectors = prepared.vectors[:, :self.metadata["dimensions"]].tolist()
        return [{"id": host, "v": row, "norm": hypot(*row)}
                for host, row in zip(prepared.ids, vectors, strict=True)]

    def validate_output(self, prepared, items):
        if [item["id"] for item in items] != prepared.ids:
            raise ValueError("Output hosts do not match the CSV")


def load(artifacts, version):
    if artifacts:
        raise ValueError("Traffic profiles do not take checkpoints")
    if version == "traffic-profile-v1":
        return TrafficProfile("traffic_profile", version, N_FEATURES)
    if version == "traffic-volume-v1":
        return TrafficProfile("traffic_volume", version, N_BASE)
    raise ValueError("Unknown built-in profile version")
