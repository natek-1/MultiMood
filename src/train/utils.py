from typing import Tuple

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

    emotion_counts = torch.zeros(7)
    sentiment_counts = torch.zeros(3)
    skipped = 0
    total = len(dataset)

    for sample in tqdm(iter(dataset), total=total, leave=False):
        if sample is None:
            skipped += 1
            continue
        
        emotion_label = int(sample['emotion_label'].item())
        sentiment_label = int(sample['sentiment_label'].item())

        emotion_counts[emotion_label] += 1
        sentiment_counts[sentiment_label] += 1
    
    valid = total - skipped
    print(f"Skipped samples: {skipped}/{total}")

    print("\nClass distribution")
    print("Emotions:")

    emotion_map = {idx: emotion for emotion, idx in  dataset.emotion_map.items()}
    sentiment_map = {idx: sentiment for sentiment, idx in  dataset.sentiment_map.items()}

    print("\nClass distribution")
    print("Emotions:")
    for idx, count in enumerate(emotion_counts):
        print(f"{emotion_map[idx]}: {count/valid:.2f}")
    
    print("\nSentiments:")
    for idx, count in enumerate(sentiment_counts):
        print(f"{sentiment_map[idx]}: {count/valid:.2f}")

    # class weights normalized
    emotion_weights = emotion_counts / emotion_counts.sum()
    sentiment_weights = sentiment_counts / sentiment_counts.sum()

    return emotion_weights, sentiment_weights






