import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cxplain.util.test_util import TestUtil

from tensorflow.python.keras.losses import mean_squared_error
from cxplain import MLPModelBuilder, ZeroMasking, CXPlain


def fit_explainer():
    model_builder = MLPModelBuilder(num_layers=2, num_units=24, activation="selu", p_dropout=0.2, verbose=0,
                                    batch_size=8, learning_rate=0.01, num_epochs=250, early_stopping_patience=15)
    masking_operation = ZeroMasking()
    loss = mean_squared_error

    explainer = CXPlain(explained_model, model_builder, masking_operation, loss, num_models=30)

    explainer.fit(x_train, y_train)

    attributions, confidence = explainer.explain(x_test, confidence_level=0.95)

    return explainer, attributions, confidence