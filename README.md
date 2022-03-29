# TCC-App: Tuberculosis prediction from X-Ray images using CNN

## To install dependencies:
1. Run `pip install -r requirements.txt`.

## To train model:
1. Update the model layers in `ml_model.py`.
2. Run the `train_job.py` script. That already runs the train set agains the model and saves the weights to npy file.

## To test model:
1. Run the `test_job.py` script. That'll output the loss and accuracy of the model, as well as create a file with the confusion matrix.

## To run app:
1. Run the flask app either with `flask run` or by running `app.py`.
2. You can hit `localhost:5000/predict` with a Multipart Form body with an image from the dataset.