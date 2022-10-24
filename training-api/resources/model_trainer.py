import logging
import os

import pickle

from flask import jsonify

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from google.cloud import storage

def train(dataset):
    X = dataset.drop(['MEDV'], axis=1)
    y = dataset['MEDV']

    model = RandomForestRegressor()
    model.fit(X, y)

    # evaluate the model
    accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    r2 = cross_val_score(model, X, y, cv=5, scoring='r2')

    text_out = {
        "accuracy:": accuracy,
        "r2": r2,
    }
    logging.info(text_out)
    print(text_out)
    # Saving model in a given location provided as an env. variable
    model_repo = os.environ['MODEL_REPO']
    if model_repo:
        file_path = os.path.join(model_repo, "model.sav")
        pickle.dump(model, file_path)
        logging.info("Saved the model to the location : " + model_repo)
        return jsonify(text_out), 200
    else:
        pickle.dumps(model)
        return jsonify({'message': 'The model was saved locally.'}), 200
