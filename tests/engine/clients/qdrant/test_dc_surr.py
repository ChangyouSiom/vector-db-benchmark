from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from enum import Enum
from loguru import logger
import uuid

class DataType(Enum):
    EMBEDING = 0
    GPS = 1


class QdrantUplload:
    def __init__(self) -> None:
        self.client = QdrantClient(host="172.16.1.235", prefer_grpc=True)
        self.collections = self.client.get_collections()

    def upload(self, datatype, collection):
        if datatype == DataType.EMBEDING:
            self.client.recreate_collection(collection, vectors_config=models.VectorParams(size=128,distance=models.Distance.COSINE))
        elif datatype == DataType.GPS:
            self.client.recreate_collection(collection, vectors_config=models.VectorParams(size=2,distance=models.Distance.EUCLID))
        logger.info('collection count {}'.format(self.client.count(collection).count))

if __name__ == '__main__':
    uploader = QdrantUplload()
    uploader.upload(DataType.EMBEDING, 'cowa_surr', 'SandCar_2')
    uploader.upload(DataType.GPS, 'cowa_surr_gps', 'GPS')