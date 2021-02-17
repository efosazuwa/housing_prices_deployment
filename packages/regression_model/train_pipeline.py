import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import pipeline

def run_training() -> None:
    """Train the model"""

    data = pd.read_csv(TRAINING_DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(data[FEATURES],
                                                        data[TARGET],
                                                        test_size=0.1,
                                                        random_state=0)
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    pipeline.price_pipe.fit(X_train[FEATURES], y_train)

    save_pipeline(pipeline_to_persist=pipeline.price_pipe)

if __name__ == '__main__':
    run_training()
