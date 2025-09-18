# Cat vs Dog Image Classification using K-Nearest Neighbors (KNN)

Welcome! My name is Yasin Pourraisi.

This project demonstrates how to classify images of cats and dogs using the K-Nearest Neighbors (KNN) algorithm with OpenCV. The workflow includes data loading, preprocessing, model training, hyperparameter tuning, and prediction on new images.

## Project Structure

- `Cat-vs-Dog-knn.ipynb` — Main Jupyter Notebook for training and evaluating the KNN model.
- `cats_dogs_images/` — Folder containing the image dataset and annotation file (ignored by git).
- `knn_best_model.yml` — Saved KNN model for future predictions.
- `predict_cat_dog_knn.py` — Example script for loading the trained model and predicting new images.
- `.gitignore` — Ensures the image dataset is not tracked by git.

## Dataset

- Images and labels are stored in `cats_dogs_images/`.
- Annotation file: `_annotations.json`
- Download the dataset [here](https://cv-studio-accessible.s3.us-south.cloud-object-storage.appdomain.cloud/cats_dogs_images_.zip).

## How to Run

1. **Install requirements**  
   Make sure you have Python, OpenCV, NumPy, and scikit-learn installed.

2. **Train the model**  
   Open and run `Cat-vs-Dog-knn.ipynb` to preprocess data, train the KNN model, and find the best value for `k`.

3. **Save the model**  
   The notebook saves the trained model as `knn_best_model.yml`.

4. **Predict new images**  
   Use `predict_cat_dog_knn.py` to load the saved model and classify new images.  
   Set the image path and `k_best` value as found in the notebook.

## Socials

- GitHub: [yasinpurraisi](https://github.com/yasinpurraisi)
- Email: yasinpourraisi@gmail.com
- Telegram: @yasinprsy

