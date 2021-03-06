from milvus import Milvus, DataType, __version__
from sklearn import preprocessing
import numpy as np
import logging
import random
import time


class Test:
    def __init__(self, nvec):
        self.cname = "benchmark"
        self.fname = "feature"
        self.dim = 128
        self.client = Milvus("localhost", 19530)
        self.prefix = '/sift1b/binary_128d_'
        self.suffix = '.npy'
        self.vecs_per_file = 100000
        self.maxfiles = 1000
        self.insert_bulk_size = 5000
        self.nvec = nvec
        self.insert_cost = 0
        self.flush_cost = 0
        self.create_index_cost = 0
        self.search_cost = 0
        assert self.nvec >= self.insert_bulk_size & self.nvec % self.insert_bulk_size == 0

    def run(self, suite):
        report = dict()
        try:
            # step 1 create collection
            logging.info(f'step 1 create collection')
            self._create_collection()
            logging.info(f'step 1 complete')

            # step 2 fill data
            logging.info(f'step 2 insert')
            start = time.time()
            self._insert()
            self.insert_cost = time.time() - start
            report["insert-speed"] = {
                "value": format(self.nvec / self.insert_cost, ".2f"),
                "unit": "vec/sec"
            }
            logging.info(f'step 2 complete')

            # step 3 flush
            logging.info(f'step 3 flush')
            start = time.time()
            self._flush()
            self.flush_cost = time.time() - start
            report["flush-cost"] = {
                "value": format(self.flush_cost, ".2f"),
                "unit": "s"
            }
            logging.info(f'step 3 complete')

            # step 4 create index
            logging.info(f'step 4 create index')
            start = time.time()
            self._create_index()
            self.create_index_cost = time.time() - start
            report["create-index-cost"] = {
                "value": format(self.create_index_cost, ".2f"),
                "unit": "s"
            }
            logging.info(f'step 4 complete')

            # step 5 load
            logging.info(f'step 5 load')
            self._load_collection()
            logging.info(f'step 5 complete')

            # step 6 search
            logging.info(f'step 6 search')
            for nq in suite["nq"]:
                for topk in suite["topk"]:
                    for nprobe in suite["nprobe"]:
                        start = time.time()
                        self._search(nq=nq, topk=topk, nprobe=nprobe)
                        self.search_cost = time.time() - start
                        report[f"search-q{nq}-k{topk}-p{nprobe}-cost"] = {
                            "value": format(self.search_cost, ".2f"),
                            "unit": "s"
                        }
            logging.info(f'step 6 complete')
        except AssertionError as ae:
            logging.exception(ae)
        except Exception as e:
            logging.error(f'test failed: {e}')
        finally:
            return report

    def _create_collection(self):
        logging.debug(f'create_collection() start')

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
        logging.debug(f'create_collection() finished')

    def _insert(self):
        logging.debug(f'insert() start')

        count = 0
        for i in range(0, self.maxfiles):
            filename = self.prefix + str(i).zfill(5) + self.suffix
            logging.debug(f'filename: {filename}')

            array = np.load(filename)
            logging.debug(f'numpy array shape: {array.shape}')

            step = self.insert_bulk_size
            for p in range(0, self.vecs_per_file, step):
                entities = [
                    {"name": self.fname, "type": DataType.FLOAT_VECTOR, "values": array[p:p + step][:].tolist()}]
                logging.debug(f'before insert slice: {p}, {p + step}')

                self.client.insert(self.cname, entities)
                logging.info(f'after insert slice: {p}, {p + step}')

                count += step
                logging.debug(f'insert count: {count}')

                if count == self.nvec:
                    logging.debug(f'inner break')
                    break
            if count == self.nvec:
                logging.debug(f'outer break')
                break
        logging.debug(f'insert() finished')

    def _flush(self):
        logging.debug(f'flush() start')

        logging.debug(f'before flush: {self.cname}')
        self.client.flush([self.cname])
        logging.info(f'after flush')

        stats = self.client.get_collection_stats(self.cname)
        logging.debug(stats)

        assert stats["row_count"] == self.nvec
        logging.debug(f'flush() finished')

    def _create_index(self):
        logging.debug(f'create_index() start')

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.client.create_index(self.cname, self.fname, index_params)
        logging.debug(f'create index {self.cname} : {self.fname} : {index_params}')
        logging.debug(f'create_index() finished')

    def _load_collection(self):
        logging.debug(f'load_collection() start')

        logging.debug(f'before load collection: {self.cname}')
        self.client.load_collection(self.cname)
        logging.debug(f'load_collection() finished')

    def _search(self, nq, topk, nprobe):
        logging.debug(f'search() start')

        result = self.client.search(self.cname,
                                    {"bool": {"must": [{"vector": {
                                        self.fname: {
                                            "metric_type": "L2",
                                            "query": _gen_vectors(nq, self.dim),
                                            "topk": topk,
                                            "params": {"nprobe": nprobe}
                                        }
                                    }}]}}
                                    )
        logging.debug(f'{result}')
        logging.debug(f'search() finished')


def _gen_vectors(num, dim):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num)]
    vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
    return vectors.tolist()
