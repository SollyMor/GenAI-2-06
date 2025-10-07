from analysis import analyze_predictions_distribution
from pathlib import Path
from enum import Enum
import os.path
import torch

from transformers import pipeline

from parser import get_parser
from config_loader import load_config, Config


class Labels(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


def check_labels(labels: list[str]) -> bool:
    '''Check that labels-file is valid.
    
    Parameters
    ----------
    labels: list[str]
        List of labels in file
    
    Returns
    -------
    bool
        Valid or not valid    
    '''
    for label in labels:
        label = Labels(label)
        if ((label != Labels.POSITIVE) and (label != Labels.NEGATIVE) and (label != Labels.NEUTRAL)):
            return False
    return True

def count_metrics(labels_path: str | Path, data: list[dict]) -> int:
    '''Calculate some metric(now only accuracy).
    
    Parameters
    ------------
    labels_path: str | Path
        Path to labels-file
    data: list[dict]
        List of dicts in format:
        - phrase: str
        - predict: str
    Returns
    ------------
    int
        Values of metrics (now only accuracy)
    '''
    good_preds_count = 0

    with open(labels_path, "r", encoding='utf-8') as file:
        lines = [line.strip().lower() for line in file.readlines() if line.strip()]

        if not check_labels(lines):
            raise ValueError("Неправильный формат меток!")

        for i, label in enumerate(lines):
            dict_i = data[i]
            text = dict_i['text']
            predict = dict_i['label']

            print(f"{text} : {predict} : {label}")

            if label == predict:
                good_preds_count += 1
    
    return good_preds_count


def convert_to_readable(predicts: list[dict], texts: list[str]) -> list[dict]:
    """We need predicts in readable format.
    
    Parameters
    ----------
    predicts: list[dict]
        List of dicts in format: predicts[i] = {'label': n star, 'score': x}
    texts: list[str]
        Phrases in test data
    
    Returns
    -------
    list[dict]
        New list of dicts in readable format: predicts[i] = {'text': <phrase>, 'label': <sentiment>, 'score': x}
    """
    star_to_sentiment = {
        '1 star': Labels.NEGATIVE.value,
        '2 stars': Labels.NEGATIVE.value,
        '3 stars': Labels.NEUTRAL.value, 
        '4 stars': Labels.POSITIVE.value,
        '5 stars': Labels.POSITIVE.value
    }
    
    readable_results = []
    for text, pred in zip(texts, predicts):
        label = pred['label']
        score = pred['score']
        readable_results.append({
            'text': text,
            'label': star_to_sentiment[label],
            'confidence': round(score, 4)
        })
    
    return readable_results


def sentiment_classification(opts: Config):
    '''Create sentiment classifier.
    
    Parameters
    ------------
    opts: Config
        Configuration for this task
    '''

    if not os.path.exists(opts.data_path):
        raise Exception(f"Файл {opts.data_path} не найден!")
    
    if not os.path.exists(opts.labels_path):
        raise Exception(f"Файл {opts.labels_path} не найден!")

    try:
        classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    except:
        raise Exception("Failed in download classifier")

    print("Начинаем обработку фраз!")

    with open(opts.data_path, "r", encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]

        predicts = classifier(lines)

        predicts = [predict for predict in predicts]

        readable_predicts = convert_to_readable(predicts, lines)

        analyze_predictions_distribution(predicts)

        print("\nРезультаты классификации:")
        print("-" * 50)
        print(f"<phrase> : <predict> : <label>")

        good_preds_count = count_metrics(opts.labels_path, readable_predicts)
        accuracy = float(good_preds_count)/len(readable_predicts) if readable_predicts else 0

        print("-" * 50)
        print(f"Точность(Accuracy) предсказаний модели: {accuracy}")


def main():
    # парсим аргументы командной строки
    parser = get_parser()
    args = parser.parse_args()

    # создаем отдельный объект класса Config 
    # (удобно для дальнейшей работы и большого коо-ва параметров)
    opts = load_config(args.config_path)

    sentiment_classification(opts)


if __name__ == "__main__":
    main()