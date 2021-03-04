from milvus import Milvus, DataType, __version__
from sklearn import preprocessing
import numpy as np
import logging
import random

logging.basicConfig(level=logging.INFO)


class Test:
    def __init__(self):
        self.cname = "benchmark"
        self.fname = "feature"
        self.dim = 128
        self.client = Milvus("localhost", 19530)
        self.prefix = '/sift1b/binary_128d_'
        self.suffix = '.npy'
        # 100,000 vector per file
        self.nfiles = 5
        self.insert_bulk_size = 5000

    def test(self):
        try:
            # step 1 create collection
            if self.client.has_collection(self.cname):
                self.client.drop_collection(self.cname)
            self.client.create_collection(self.cname, {
                "fields": [{
                    "name": self.fname,
                    "type": DataType.FLOAT_VECTOR,
                    "metric_type": "L2",
                    "params": {"dim": self.dim},
                    "indexes": [{"metric_type": "L2"}]
                }]
            })
            assert self.client.has_collection(self.cname)
            logging.info(f'after create collection: {self.cname}')

            # step 2 fill data
            for i in range(0, self.nfiles):
                filename = self.prefix + str(i).zfill(5) + self.suffix
                logging.debug(f'filename: {filename}')
                array = np.load(filename)
                logging.debug(f'numpy array shape: {array.shape}')
                step = self.insert_bulk_size
                for p in range(0, 100000, step):
                    logging.debug(f'numpy array slice: {p}, {p + step}')
                    entities = [
                        {"name": self.fname, "type": DataType.FLOAT_VECTOR, "values": array[p:p + step][:].tolist()}]
                    logging.debug(f'before insert slice: {p}, {p + step}')
                    self.client.insert(self.cname, entities)
                    logging.info(f'after insert slice: {p}, {p + step}')
                logging.info(f'after insert file: {filename}')
            logging.debug(f'before flush: {self.cname}')
            self.client.flush([self.cname])
            logging.info(f'after flush')
            stats = self.client.get_collection_stats(self.cname)
            logging.debug(stats)
            # TODO: remove str() conversion
            assert stats["row_count"] == str(self.nfiles * 100000)
            logging.info(f'after fill collection: {self.cname}')

            # step 3 load
            logging.debug(f'before load collection: {self.cname}')
            self.client.load_collection(self.cname)
            logging.info(f'after load collection')

            # step 4 search
            result = self.client.search(self.cname,
                                        {"bool": {"must": [{"vector": {
                                            "Vec": {
                                                "metric_type": "L2",
                                                "query": gen_vectors(10, self.dim),
                                                "topk": 10
                                            }
                                        }}]}}
                                        )
            logging.debug(f'{result}')
        except AssertionError as ae:
            logging.exception(ae)
        except Exception as e:
            print(f'test failed: {e}')
        return False


def gen_vectors(num, dim):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num)]
    vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
    return vectors.tolist()


def main():
    print(f"Run on pymilvus v{__version__}")
    test = Test()
    test.test()


if __name__ == "__main__":
    main()
