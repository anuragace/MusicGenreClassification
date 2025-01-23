# MusicGenreClassification

## Description

This project aims to develop a robust and accurate model for automatically classifying music tracks into predefined genres using machine learning techniques. The project leverages audio features and advanced algorithms to achieve high accuracy in genre classification.

## Dataset

The project utilizes the GTZAN Genre Collection dataset, a popular benchmark dataset for music genre classification. This dataset consists of 1000 audio tracks, each 30 seconds long, categorized into 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

You can download the dataset from this link: [http://opihi.cs.uvic.ca/sound/genres.tar.gz](http://opihi.cs.uvic.ca/sound/genres.tar.gz)

## Methodology

1.  **Data Preprocessing:** The audio tracks are preprocessed to extract relevant features, such as Mel-frequency cepstral coefficients (MFCCs), spectral centroid, spectral roll-off, and zero-crossing rate. These features capture essential characteristics of the audio signals that are relevant for genre classification.

2.  **Model Selection:** The project explores and compares the performance of two different machine learning algorithms:
    *   K-Nearest Neighbors (KNN): A simple and effective algorithm for classification based on distance metrics.
    *   Convolutional Neural Networks (CNNs): A powerful deep learning technique well-suited for analyzing audio data and extracting complex patterns.

3.  **Model Training and Evaluation:** The selected models are trained on the preprocessed audio features and evaluated using metrics such as accuracy, precision, recall, and F1-score. The performance of KNN and CNNs is compared to identify the most effective approach for music genre classification.

## Results

The project achieved high accuracy in classifying music tracks into the 10 predefined genres. The CNN model outperformed the KNN model, demonstrating the effectiveness of deep learning techniques for this task.

## Usage

To run this project, you need to have the following dependencies installed:

*   Python 3.x
*   Librosa (for audio analysis)
*   Scikit-learn (for machine learning algorithms)
*   TensorFlow/Keras (for deep learning)

You can install these dependencies using pip:

```bash
pip install librosa scikit-learn tensorflow

The code for this project is organized into separate scripts for data preprocessing, model training, and evaluation. You can run these scripts in a Python environment to reproduce the results.

Conclusion
This project successfully demonstrates the application of machine learning techniques for music genre classification. The results show that deep learning models, particularly CNNs, can achieve high accuracy in classifying music genres based on audio features.

Future Work
Explore other deep learning architectures, such as recurrent neural networks (RNNs) or transformers, to potentially improve classification accuracy.
Investigate the use of data augmentation techniques to increase the size of the training dataset and improve model generalization.
Develop a user interface or web application to allow users to classify their own music files.