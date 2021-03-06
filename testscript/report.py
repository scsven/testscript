import json

class Report:
    def __init__(self, data):
        self.data = data

    def dump(self):
        return json.dumps(self.data, indent=4, sort_keys=False)

    def file(self, filename):
        with open(filename, 'w+') as f:
            f.write(json.dumps(self.data, indent=4, sort_keys=False))

