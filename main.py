"""Учебен класификатор на фитнес ниво."""
import argparse
import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["training_minutes", "endurance_minutes"]
LEVELS = ["Начинаещ", "Средно", "Напреднал"]


def run():
    parser = argparse.ArgumentParser(description="Учебна прогноза за фитнес ниво")
    parser.add_argument("--predict", nargs=2, type=int, metavar=("ТРЕНИРОВКИ", "ТЕСТ"))
    args = parser.parse_args()
    if args.predict and not (0 <= args.predict[0] <= 300 and 0 <= args.predict[1] <= 30):
        parser.error("Тренировки: 0–300 минути. Тест: 0–30 минути.")

    with Path(__file__).with_name("data.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    features = [[int(row[name]) for name in FEATURES] for row in rows]
    levels = [row["level"] for row in rows]

    x_train, x_test, y_train, y_test = train_test_split(
        features, levels, test_size=0.2, random_state=42, stratify=levels
    )
    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=4, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(f"Обучение: {len(x_train)} примера | Проверка: {len(x_test)} примера")
    correct = int(accuracy_score(y_test, predictions, normalize=False))
    print(f"Точност: {correct}/{len(y_test)} ({correct / len(y_test):.1%})")
    print("Матрица на грешките (ред: записано ниво, колона: прогнозирано ниво)")
    print("Ред и колони: " + " | ".join(LEVELS))
    print(confusion_matrix(y_test, predictions, labels=LEVELS))
    for values, expected, predicted in zip(x_test, y_test, predictions):
        if expected != predicted:
            print(f"Пример за грешка: {values[0]} мин тренировки, {values[1]} мин тест."
                  f" Записано: {expected}. Прогноза: {predicted}.")
            break

    examples = [args.predict] if args.predict else [[25, 5], [100, 12], [200, 21]]
    names = ["Седмични тренировки", "Издръжливост"]
    for example in examples:
        print(f"\n{example[0]} мин тренировки, {example[1]} мин тест → "
              f"{model.predict([example])[0]}")
        for node in model.decision_path([example]).indices:
            feature = model.tree_.feature[node]
            if feature >= 0:
                threshold = model.tree_.threshold[node]
                comparison = "≤" if example[feature] <= threshold else ">"
                print(f"  {names[feature]}: {example[feature]} {comparison} {threshold:g}")


if __name__ == "__main__":
    run()
