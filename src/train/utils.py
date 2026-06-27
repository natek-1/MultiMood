from typing import List, Tuple

from tqdm import tqdm
import torch

from src.dataset.meld_dataset import MELDDataset


def compute_class_weight(dataset: MELDDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Compute class weights for the MELD dataset.

    Args:
        dataset (MELDDataset): MELD dataset.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the emotion and sentiment class weights.
    '''

    df = dataset.data
    emotion_map = dataset.emotion_map
    sentiment_map = dataset.sentiment_map

    print("Class distribution")
    print("Emotions:")
    emotion_weights = df['Emotion'].value_counts(normalize=True).to_dict()
    for emotion, weight in emotion_weights.items():
        print(f"{emotion}: {weight:.4f}")
    print("\nSentiments:")
    sentiment_weights = df['Sentiment'].value_counts(normalize=True).to_dict()
    for sentiment, weight in sentiment_weights.items():
        print(f"{sentiment}: {weight:.4f}")

    ordered_emotion_weights = [0.0] * len(emotion_map)
    for emotion, idx in emotion_map.items():
        ordered_emotion_weights[idx] = 1 / emotion_weights[emotion]
    ordered_emotion_weights = torch.Tensor(ordered_emotion_weights)

    ordered_sentiment_weights = [0.0] * len(sentiment_map)
    for sentiment, idx in sentiment_map.items():
        ordered_sentiment_weights[idx] = 1/sentiment_weights[sentiment]
    ordered_sentiment_weights = torch.Tensor(ordered_sentiment_weights)

    return ordered_emotion_weights, ordered_sentiment_weights






