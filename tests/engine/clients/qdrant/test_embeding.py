from qdrant_client import QdrantClient
from qdrant_client.http import models
from loguru import logger
import uuid
import numpy as np

class QdrantHelper:
    def __init__(self):
        self.client = QdrantClient(host="172.16.1.235", prefer_grpc=True)
        self.collections = [item.name for item in self.client.get_collections().collections]

    def create_collection(self, collection, size):
        self.client.recreate_collection(collection, vectors_config=models.VectorParams(size=size,distance=models.Distance.COSINE))
        logger.info('collection count {}'.format(self.client.count(collection).count))
        self.collections = [item.name for item in self.client.get_collections().collections]

    @logger.catch
    def insert_data(self, collection, data, image_names, boxes):
        if (len(data) != len(image_names)) or (len(data) != len(boxes)):
            logger.error('size error')
            return
        if collection not in self.collections:
            logger.info('no collection named {}'.format(collection))
            return
        count0 = self.client.count(collection).count
        points=[]
        for i in range(len(data)):
            points.append(models.PointStruct(id=str(uuid.uuid3(uuid.NAMESPACE_DNS, image_names[i])),payload={'image_name': image_names[i], 'box': boxes[i]}, vector=data[i]))
        self.client.upsert(collection, points)
        count1 = self.client.count(collection).count
        if count1-count0 != len(data):
            logger.warning('insert error, data size {}'.format(len(data)))
        logger.info('collection count from {} to {}'.format(count0, count1))

    def delete_data(self):
        pass

    def search_data(self, collection, data, threshold=None, limit=1):
        result = [[]]*len(data)
        if collection not in self.collections:
            logger.info('no collection named {}'.format(collection))
            return result
        search_queries = []
        for item in data:
            search_queries.append(models.SearchRequest(vector=item, limit=limit, with_payload=['image_name', 'box'], with_vector=True, score_threshold=threshold))
        for i, item in enumerate(self.client.search_batch(collection, requests=search_queries)):
            result[i] = [{'score': it.score, 'image_name':it.payload['image_name'], 'box': it.payload['box'], 'vector': it.vector } for it in item]
        return result


if __name__ == '__main__':
    qdrant_helper = QdrantHelper()
    size = 10 #向量长度
    qdrant_helper.create_collection('embdeing_test', size)

    data=[]
    image_names=[]
    boxes=[]
    for i in range(20000):
        data.append(np.random.rand(size).tolist())
        image_names.append('image_name_'+str(i)) #使用image_name生成uuid,重复即覆盖
        boxes.append([-0.25, -0.25, 0.25, 0.25])
    qdrant_helper.insert_data('embdeing_test', data, image_names, boxes)

    search_data=[]
    for i in range(3):
        search_data.append(np.random.rand(size).tolist())
    result = qdrant_helper.search_data('embdeing_test', search_data, 0.5, 3)
    print(result)