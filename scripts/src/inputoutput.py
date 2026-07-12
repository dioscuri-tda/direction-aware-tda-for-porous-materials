import datetime
import json
import pickle


def get_experiment_name(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiment_name = f"{timestamp}_{args['model_name']}_{args['optimizer']}_lr{args['lr']}"
    if args["suffix"] != "":
        experiment_name = f"{experiment_name}_{args['suffix']}"
    return experiment_name


def save_to_pickle(data, filename):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle)


def save_to_json(data, filename):
    with open(filename, "w") as handle:
        json.dump(data, handle, indent=4)
