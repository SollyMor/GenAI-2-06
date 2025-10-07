# GenAI-2-06. ML Model Ratings Analysis

A project for analyzing and visualizing ratings obtained from an ML model.

## Description

This project implements a pipeline for processing ratings from an ML model:
1. Getting textual ratings (e.g., '4 stars')
2. Converting to numerical format
3. Categorizing ratings
4. Building bar plot of rating distribution
5. Calculating and displaying category proportions

## Project Structure
```
├── analysis.py # Main ratings analysis module
├── config_loader.py # Configuration loader
├── data.txt # Analysis data
├── labels.txt # Labels/categories
├── main.py # Main script
├── parser.py # Data parser
└── pop.yaml # Configuration file
```

## Main Functions (analysis.py)

- **parse_rating(text)**: Extracts numerical value from textual rating
- **categorize_rating(rating)**: Assigns category based on numerical rating
- **plot_rating_distribution(ratings)**: Builds bar plot of rating distribution
- **calculate_proportions(ratings)**: Calculates proportions for each category
- **main()**: Main function combining all pipeline stages

## Dependencies
    pandas
    matplotlib
    numpy
    (other dependencies specified in requirements.txt)

## Configuration

Project settings can be modified in the pop.yaml file
