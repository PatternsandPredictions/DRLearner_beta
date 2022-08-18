import json
from collections import defaultdict
from statistics import mean, median


def read_config(path):
    with open(path, 'r') as file:
        config = json.load(file)

    return config


class StatsCheckpointer:
    def __init__(self):
        self.values = defaultdict(list)
        self.statistics = defaultdict(dict)
        self._config = read_config('./configs/config.json')

        self._target_statistics = {
            'min': min,
            'max': max,
            'mean': mean,
            'median': median,
        }
        self._target_metrics = ['episode_return']

    def update(self, result, log=False):
        for metric in self._target_metrics:
            value = float(result[metric])
            self.values[metric].append(value)

        self.evaluate()
        self.save()

        if log:
            self.log()

    def evaluate(self):
        for metric in self._target_metrics:
            values = self.values[metric]

            for statistic, function in self._target_statistics.items():
                self.statistics[metric][statistic] = round(function(values), 5)

    def save(self):
        path = self._config['statistics_path']

        with open(path, 'w') as file:
            json.dump(self.statistics, file)

    def log(self):
        print('=' * 30)
        for metric in self._target_metrics:
            print(f"{metric.replace('_', ' ').upper()}")

            for statistic, value in self.statistics[metric].items():
                print(f'{statistic}: {value}')

        print('=' * 30)

    def __repr__(self):
        return str(self.statistics)
