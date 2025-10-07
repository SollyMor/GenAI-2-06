import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict
import numpy as np


def analyze_predictions_distribution(predicts: List[Dict]) -> None:
    """
    Анализирует распределение предсказаний модели и визуализирует результаты.
    
    Выполняет полный анализ предсказаний модели: преобразует звездные оценки в числа,
    категоризирует по настроению, строит столбчатые диаграммы и выводит статистику.
    
    Parameters
    ----------
    predicts : List[Dict]
        Список словарей с предсказаниями модели в формате:
        [{'label': '4 stars', 'score': 0.95}, ...]
    
    Returns
    -------
    None
        Функция выводит результаты на экран и отображает графики.
    """
    print("\n" + "="*50)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ ПРЕДСКАЗАНИЙ:")
    print("="*50)
    
    # 1. Получаем оценки от модели
    star_predictions = [pred['label'] for pred in predicts]
    
    # 2. Преобразуем в числа
    star_to_number = {
        '1 star': 1, '2 stars': 2, '3 stars': 3, '4 stars': 4, '5 stars': 5
    }
    numeric_scores = [star_to_number[pred] for pred in star_predictions]
    
    # 3. Назначаем категории
    star_to_sentiment = {
        '1 star': 'negative', '2 stars': 'negative',
        '3 stars': 'neutral', 
        '4 stars': 'positive', '5 stars': 'positive'
    }
    categories = [star_to_sentiment[pred] for pred in star_predictions]
    
    # 4. Строим bar plot распределения
    _create_plots(numeric_scores, categories, star_predictions)
    
    # 5. Выводим доли
    _print_statistics(numeric_scores, categories, len(predicts))


def _create_plots(numeric_scores: List[int], categories: List[str], star_predictions: List[str]) -> None:
    """
    Создает столбчатые диаграммы распределения предсказаний.
    
    Строит два графика: распределение по звездным оценкам и распределение по категориям настроения.
    
    Parameters
    ----------
    numeric_scores : List[int]
        Список числовых оценок (1-5)
    categories : List[str]
        Список категорий настроения ('negative', 'neutral', 'positive')
    star_predictions : List[str]
        Список исходных звездных оценок
    
    Returns
    -------
    None
        Отображает matplotlib графики.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot для звезд
    star_counts = Counter(numeric_scores)
    stars = sorted(star_counts.keys())
    counts = [star_counts[star] for star in stars]
    
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    bars1 = ax1.bar(stars, counts, color=colors[:len(stars)])
    ax1.set_xlabel('Количество звезд')
    ax1.set_ylabel('Количество отзывов')
    ax1.set_title('Распределение по звездам')
    ax1.set_xticks(stars)
    
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{count}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Bar plot для категорий
    category_counts = Counter(categories)
    categories_list = ['negative', 'neutral', 'positive']
    category_colors = ['red', 'yellow', 'green']
    counts_cat = [category_counts[cat] for cat in categories_list]
    
    bars2 = ax2.bar(categories_list, counts_cat, color=category_colors)
    ax2.set_xlabel('Категория настроения')
    ax2.set_ylabel('Количество отзывов')
    ax2.set_title('Распределение по категориям настроения')
    
    for bar, count in zip(bars2, counts_cat):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{count}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def _print_statistics(numeric_scores: List[int], categories: List[str], total: int) -> None:
    """
    Выводит статистику распределения предсказаний.
    
    Parameters
    ----------
    numeric_scores : List[int]
        Список числовых оценок (1-5)
    categories : List[str]
        Список категорий настроения
    total : int
        Общее количество предсказаний
    
    Returns
    -------
    None
        Выводит статистику в консоль.
    """
    star_counts = Counter(numeric_scores)
    category_counts = Counter(categories)
    
    print("\nСТАТИСТИКА РАСПРЕДЕЛЕНИЯ:")
    print("-" * 40)
    
    print("\nПо звездам:")
    for star in sorted(star_counts.keys()):
        count = star_counts[star]
        percentage = (count / total) * 100
        print(f"  {star} звезд: {count:3d} отзывов ({percentage:5.1f}%)")
    
    print("\nПо категориям настроения:")
    categories_list = ['negative', 'neutral', 'positive']
    for category in categories_list:
        count = category_counts[category]
        percentage = (count / total) * 100
        print(f"  {category:8}: {count:3d} отзывов ({percentage:5.1f}%)")
    
    print(f"\nОбщее количество отзывов: {total}")
    print(f"Средний рейтинг: {np.mean(numeric_scores):.2f}")
    print(f"Медианный рейтинг: {np.median(numeric_scores):.1f}")


def analyze_confidence_scores(predicts: List[Dict]) -> None:
    """
    Анализирует распределение уверенности модели в предсказаниях.
    
    Строит гистограмму уверенности модели и выводит описательную статистику.
    
    Parameters
    ----------
    predicts : List[Dict]
        Список словарей с предсказаниями модели
    
    Returns
    -------
    None
        Отображает гистограмму и выводит статистику в консоль.
    """
    confidence_scores = [pred['score'] for pred in predicts]
    
    print("\n" + "="*50)
    print("АНАЛИЗ УВЕРЕННОСТИ МОДЕЛИ:")
    print("="*50)
    
    plt.figure(figsize=(10, 5))
    plt.hist(confidence_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Уверенность модели')
    plt.ylabel('Количество предсказаний')
    plt.title('Распределение уверенности модели в предсказаниях')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Средняя уверенность: {np.mean(confidence_scores):.3f}")
    print(f"Медианная уверенность: {np.median(confidence_scores):.3f}")
    print(f"Минимальная уверенность: {np.min(confidence_scores):.3f}")
    print(f"Максимальная уверенность: {np.max(confidence_scores):.3f}")