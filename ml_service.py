import sys
import logging

from keras import Model


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(images, labels, model: Model) -> Model:
    logger.info('Started training model...')
    model.fit(x=images, y=labels, epochs=1, verbose=1)
    logger.info('Finished training model.')
    return model


def test(images, labels, model: Model):
    logger.info('Testing model...')
    return model.evaluate(x=images, y=labels, verbose=0)


def predict(image, model: Model):
    return float(model(image))
