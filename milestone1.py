#! /usr/bin/env python3

from milvus import Milvus, DataType, __version__
from sklearn import preprocessing
import numpy as np
import logging
import random
import time

logging.basicConfig(level=logging.INFO)


class Test:
    def __init__(self):
        self.cname = "benchmark"
        self.fname = "feature"
        self.dim = 128
        self.client = Milvus("localhost", 19530)
        self.prefix = '/sift1b/binary_128d_'
        self.suffix = '.npy'
        self.nvec = 5000
        self.insert_cost = 0

    def test(self):
        try:
            # step 1 create collection
            logging.info(f'step 1 create collection')
            if self.client.has_collection(self.cname):
                logging.debug(f'collection {self.cname} existed')
                self.client.drop_collection(self.cname)
                logging.info(f'drop collection {self.cname}')
            logging.debug(f'before create collection: {self.cname}')
            self.client.create_collection(self.cname, {
                "fields": [{
                    "name": self.fname,
                    "type": DataType.FLOAT_VECTOR,
                    "metric_type": "L2",
                    "params": {"dim": self.dim},
                    "indexes": [{"metric_type": "L2"}]
                }]
            })
            logging.info(f'created collection: {self.cname}')
            assert self.client.has_collection(self.cname)
            logging.info(f'step 1 complete')

            # step 2 fill 5,000 data
            logging.info(f'step 2 fill data')
            filename = self.prefix + str(0).zfill(5) + self.suffix
            logging.debug(f'filename: {filename}')
            array = np.load(filename)
            logging.debug(f'numpy array shape: {array.shape}')
            assert 0 <= self.nvec <= 100000
            entities = [
                {"name": self.fname, "type": DataType.FLOAT_VECTOR, "values": array[0:self.nvec][:].tolist()}]
            logging.debug(f'before insert')
            begin = time.time()
            self.client.insert(self.cname, entities)
            self.insert_cost = time.time() - begin
            logging.info(f'after insert file: {filename}')
            logging.debug(f'before flush: {self.cname}')
            self.client.flush([self.cname])
            logging.info(f'after flush')
            stats = self.client.get_collection_stats(self.cname)
            logging.debug(stats)
            assert stats["row_count"] == self.nvec
            logging.info(f'step 2 complete')

            # step 3 create index
            logging.info(f'step 3 create index')
            self.client.create_index(self.cname, self.fname, {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            })
            logging.info(f'step 3 complete')

            # step 4 load
            logging.info(f'step 4 load')
            logging.debug(f'before load collection: {self.cname}')
            self.client.load_collection(self.cname)
            logging.info(f'step 4 complete')

            # step 5 search
            logging.info(f'step 5 search')
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
            logging.info(f'step 5 complete')
        except AssertionError as ae:
            logging.exception(ae)
        except Exception as e:
            logging.error(f'test failed: {e}')
        finally:
            if self.insert_cost > 0:
                logging.info(f'insert speed: {self.nvec / self.insert_cost} vector per second')
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
