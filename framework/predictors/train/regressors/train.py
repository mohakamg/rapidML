import sys
import argparse
import json
import pandas as pd
import uuid
from framework.predictors.train.regressors.executor import Executor
from framework.stock.classifiers import get_supported_classifiers
from framework.stock.regressors import get_supported_regressors


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description='Training execution.')

    parser.add_argument('--config', help='path to configuration file', default=None)
    parser.add_argument(
        '--data_path',
        help='path to dataset (.csv)',
        default=None)
    parser.add_argument(
        '--output_path',
        help='path where you want output sent to',
        default='.')

    parsed_args, _ = parser.parse_known_args(args)

    config = None
    if parsed_args.config is not None:
        with open(parsed_args.config) as f_in:
            config = json.load(f_in)

    dataset = None
    if parsed_args.data_path is not None:
        dataset = pd.read_csv(parsed_args.data_path)

    # TODO: Include a check to make sure the output path is a valid path?

    output_path = parsed_args.output_path

    return config, dataset, output_path


if __name__ == '__main__':
    config, dataset, output_path = parse_arguments(sys.argv)

    predictor_type = config['predictor_type']
    predictor_name = config['predictor_name']

    predictor = None

    if predictor_type == "regressor":
        predictor = get_supported_regressors().get(predictor_name, None)
    elif predictor_type == 'classifier':
        predictor = get_supported_classifiers().get(predictor_name, None)

    if predictor is not None:

        executor = Executor(predictor_class_ref=predictor,
                            data=dataset,
                            coa_mapping=config['cao_mapping'],
                            data_split=config['data_split'],
                            model_params=config['model_params'],
                            metrics=config['metrics'],
                            executor_name=str(uuid.uuid4()),
                            model_metadata=config['model_metadata']
                            )

        executor.execute()
    else:
        print('No Valid Predictor Selected.')
        exit(0)
