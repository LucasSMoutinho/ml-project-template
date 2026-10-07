# Water Potability Prediction — LightGBM

## 🇧🇷 Português

### 📌 Contexto

A qualidade da água está diretamente relacionada à saúde e à segurança para consumo humano. Entretanto, a avaliação da potabilidade pode envolver diversas características físico-químicas da água, tornando interessante analisar essas informações de forma conjunta.

Este projeto utiliza **Machine Learning** para investigar se é possível prever se uma determinada amostra de água é **potável ou não potável** a partir de suas características físico-químicas.

O projeto faz parte de uma série de estudos sobre diferentes algoritmos de Machine Learning e utiliza o **LightGBM (Light Gradient Boosting Machine)** como modelo principal.

> **Notebook de referência:** https://www.kaggle.com/code/pumalin/lightgbm-tutorial/notebook

---

## 🎯 Problema

Dada uma amostra de água com determinadas características físico-químicas, queremos responder:

> **É possível prever se essa água é potável ou não?**

O problema pode ser representado como:

$$
X \rightarrow Y
$$

onde:

- **X** = características físico-químicas da água;
- **Y** = potabilidade da amostra.

A variável `Potability` representa o resultado:

- `0` → água não potável;
- `1` → água potável.

Portanto, trata-se de um problema de **classificação binária supervisionada**.

---

## 🎯 Objetivo

O objetivo deste projeto é desenvolver e avaliar um modelo de Machine Learning capaz de classificar amostras de água como **potáveis ou não potáveis**, utilizando suas características físico-químicas como variáveis preditoras.

Além da capacidade preditiva, o projeto busca compreender:

- se existe relação entre as características da água e sua potabilidade;
- se o modelo consegue aprender esses padrões;
- se o modelo consegue generalizar para novas amostras;
- quais características possuem maior influência nas previsões;
- quais métricas são mais adequadas para avaliar o desempenho do modelo.

---

## 📊 Dataset

O dataset contém aproximadamente **3.276 amostras de água**, com características físico-químicas utilizadas para prever a variável `Potability`.

### Variáveis preditoras

| Variável | Descrição |
|---|---|
| `pH` | Medida de acidez ou alcalinidade da água |
| `Hardness` | Dureza da água, relacionada principalmente à presença de cálcio e magnésio |
| `Solids` | Quantidade de sólidos dissolvidos na água |
| `Chloramines` | Concentração de cloraminas |
| `Sulfate` | Concentração de sulfatos |
| `Conductivity` | Capacidade da água de conduzir eletricidade |
| `Organic_carbon` | Quantidade de carbono orgânico |
| `Trihalomethanes` | Concentração de trihalometanos |
| `Turbidity` | Medida relacionada à turbidez da água |

### Variável-alvo

| Variável | Descrição |
|---|---|
| `Potability` | Indica se a água é potável (`1`) ou não potável (`0`) |

---

## 🧠 Por que utilizar Machine Learning?

Uma característica importante desse problema é que a potabilidade não precisa depender de apenas uma variável.

Por exemplo, analisar somente o pH pode não ser suficiente para determinar se uma amostra é potável. Entretanto, a combinação de:

- pH;
- sólidos dissolvidos;
- sulfatos;
- turbidez;
- condutividade;
- carbono orgânico;
- entre outras características,

pode apresentar padrões úteis para diferenciar as classes.

O Machine Learning permite que o algoritmo encontre essas relações a partir dos exemplos disponíveis nos dados.

De forma simplificada, queremos aprender uma função:

$$
f(X) \approx Y
$$

ou, em uma abordagem probabilística:

$$
P(Y=1|X)
$$

Ou seja:

> **Qual é a probabilidade de uma amostra ser potável dadas suas características?**

---

## 🌳 Por que LightGBM?

O **LightGBM** é um algoritmo baseado em **Gradient Boosting com árvores de decisão**.

A ideia é construir várias árvores sequencialmente, fazendo com que novas árvores contribuam para corrigir os erros cometidos pelo conjunto de árvores anteriores.

De forma simplificada:

```text
Dados
  ↓
Árvore 1
  ↓
Identificação dos erros
  ↓
Árvore 2 → tenta corrigir os erros
  ↓
Identificação dos novos erros
  ↓
Árvore 3 → tenta melhorar novamente
  ↓
...
  ↓
Modelo final
```

O LightGBM é especialmente utilizado em problemas com dados tabulares e pode apresentar bom desempenho em tarefas de classificação e regressão.

Neste projeto, ele é utilizado para aprender a relação entre as características físico-químicas da água e a variável `Potability`.

---

## 🔄 Pipeline do projeto

O fluxo geral do projeto pode ser representado da seguinte maneira:

```text
                    Dataset
                       │
                       ▼
              Análise exploratória
                       │
                       ▼
             Tratamento dos dados
                       │
                       ▼
            Tratamento dos valores
                 ausentes
                       │
                       ▼
             Separação X e y
                       │
                       ▼
             Train / Test Split
                  │         │
                  ▼         │
              Treinamento   │
              LightGBM      │
                  │         │
                  └────┬────┘
                       ▼
                   Predições
                       │
                       ▼
                    Avaliação
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          Accuracy  Precision  Recall
             │         │         │
             └─────────┼─────────┘
                       ▼
                  Interpretação
```

---

## 🧹 Pré-processamento

Antes do treinamento, os dados precisam ser analisados e preparados.

Um dos pontos importantes deste dataset é a presença de **valores ausentes** em algumas variáveis.

Isso torna necessário avaliar estratégias de tratamento dos dados antes da modelagem.

O fluxo de preparação pode incluir:

1. inspeção da estrutura dos dados;
2. análise estatística das variáveis;
3. identificação de valores ausentes;
4. tratamento dos valores ausentes;
5. separação das variáveis preditoras e da variável-alvo;
6. divisão entre dados de treinamento e teste;
7. treinamento do modelo.

---

## 📈 Avaliação do modelo

Como o problema é de classificação binária, diferentes métricas podem ser utilizadas.

### Accuracy

Representa a proporção de previsões corretas:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Apesar de ser uma métrica intuitiva, a accuracy deve ser analisada em conjunto com outras métricas, especialmente quando existe desequilíbrio entre as classes.

### Precision

Mede, entre as observações classificadas como positivas, quantas realmente são positivas:

$$
Precision = \frac{TP}{TP + FP}
$$

### Recall

Mede, entre as observações que realmente são positivas, quantas foram identificadas corretamente:

$$
Recall = \frac{TP}{TP + FN}
$$

### F1-Score

Representa a média harmônica entre Precision e Recall:

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

### ROC-AUC

A ROC-AUC permite avaliar a capacidade do modelo de separar as duas classes considerando diferentes thresholds de classificação.

Essa métrica é especialmente útil porque não depende de um único threshold, como `0.5`.

---

## ⚠️ Por que não devemos olhar apenas para Accuracy?

Imagine que o modelo tenha uma accuracy elevada, mas apresente dificuldade em identificar corretamente uma das classes.

Nesse cenário, a accuracy pode transmitir uma percepção exageradamente positiva do desempenho.

Por isso, a avaliação deve considerar diferentes aspectos do modelo:

```text
Accuracy
   +
Precision
   +
Recall
   +
F1-Score
   +
ROC-AUC
   +
Matriz de Confusão
```

Além das métricas, também é importante analisar se existe diferença significativa entre o desempenho no treinamento e no teste.

Essa comparação ajuda a identificar possíveis problemas de **overfitting**.

---

## 🔬 Perguntas que o projeto busca responder

O projeto não tem como objetivo apenas obter uma determinada porcentagem de acerto.

As principais perguntas são:

### 1. Existe relação entre as características físico-químicas e a potabilidade?

O modelo consegue encontrar padrões que diferenciem as classes?

### 2. O LightGBM consegue aprender esses padrões?

O modelo apresenta desempenho suficiente para realizar classificações?

### 3. O modelo generaliza?

O desempenho permanece consistente em dados que não foram utilizados durante o treinamento?

### 4. Quais características são mais relevantes?

Podemos utilizar técnicas de interpretação e importância de features para entender quais variáveis mais contribuem para o modelo.

### 5. Qual é o tipo de erro mais relevante?

A matriz de confusão permite analisar:

- Verdadeiros Positivos;
- Verdadeiros Negativos;
- Falsos Positivos;
- Falsos Negativos.

Em uma aplicação real relacionada à qualidade da água, os diferentes tipos de erro podem possuir impactos diferentes.

---

## 💡 Visão de Machine Learning

Este projeto segue uma ideia fundamental em Ciência de Dados:

> **O modelo não é o ponto de partida. O ponto de partida é o problema.**

A sequência de raciocínio é:

```text
Problema
   ↓
Pergunta de negócio/científica
   ↓
Dados necessários
   ↓
Definição do target
   ↓
Preparação dos dados
   ↓
Modelo
   ↓
Avaliação
   ↓
Interpretação
   ↓
Decisão
```

No projeto:

```text
Problema:
Avaliar a potabilidade da água

        ↓

Dados:
Características físico-químicas

        ↓

Target:
Potability

        ↓

Tipo de problema:
Classificação binária

        ↓

Modelo:
LightGBM

        ↓

Resultado:
Probabilidade/classificação de potabilidade
```

---

## 🚀 Próximos passos

Possíveis extensões do projeto:

- comparar LightGBM com Logistic Regression;
- comparar com Decision Tree e Random Forest;
- testar XGBoost e CatBoost;
- realizar validação cruzada;
- otimizar hiperparâmetros;
- analisar importância das variáveis;
- utilizar SHAP para interpretação das previsões;
- avaliar calibração das probabilidades;
- investigar o impacto do desbalanceamento das classes;
- comparar diferentes thresholds;
- construir uma análise mais completa de erro.

---

# 🇺🇸 English

## 📌 Context

Water quality is directly related to human health and safety. However, determining whether water is suitable for consumption can involve several physicochemical characteristics, making it useful to analyze these variables jointly.

This project uses **Machine Learning** to investigate whether it is possible to predict whether a water sample is **potable or non-potable** based on its physicochemical characteristics.

The project is part of a series of studies focused on different Machine Learning algorithms and uses **LightGBM (Light Gradient Boosting Machine)** as the main model.

> **Reference notebook:** https://www.kaggle.com/code/pumalin/lightgbm-tutorial/notebook

---

## 🎯 Problem

Given a water sample with a set of physicochemical characteristics, we want to answer:

> **Can we predict whether this water is potable or not?**

The problem can be represented as:

$$
X \rightarrow Y
$$

where:

- **X** = physicochemical characteristics of the water;
- **Y** = sample potability.

The `Potability` variable represents the outcome:

- `0` → non-potable water;
- `1` → potable water.

Therefore, this is a **supervised binary classification problem**.

---

## 🎯 Objective

The objective of this project is to develop and evaluate a Machine Learning model capable of classifying water samples as **potable or non-potable**, using their physicochemical characteristics as predictive variables.

Beyond predictive performance, the project aims to understand:

- whether there is a relationship between water characteristics and potability;
- whether the model can learn these patterns;
- whether the model can generalize to unseen samples;
- which features have the greatest influence on predictions;
- which evaluation metrics are most appropriate for this problem.

---

## 📊 Dataset

The dataset contains approximately **3,276 water samples**, with physicochemical characteristics used to predict the `Potability` target variable.

### Predictive variables

| Variable | Description |
|---|---|
| `pH` | Measure of water acidity or alkalinity |
| `Hardness` | Water hardness, mainly related to calcium and magnesium |
| `Solids` | Amount of dissolved solids in the water |
| `Chloramines` | Chloramine concentration |
| `Sulfate` | Sulfate concentration |
| `Conductivity` | Water's ability to conduct electricity |
| `Organic_carbon` | Amount of organic carbon |
| `Trihalomethanes` | Trihalomethane concentration |
| `Turbidity` | Measure related to water turbidity |

### Target variable

| Variable | Description |
|---|---|
| `Potability` | Indicates whether the water is potable (`1`) or non-potable (`0`) |

---

## 🧠 Why Machine Learning?

An important characteristic of this problem is that potability does not necessarily depend on a single variable.

For example, analyzing only pH may not be enough to determine whether a sample is potable. However, the combination of:

- pH;
- dissolved solids;
- sulfates;
- turbidity;
- conductivity;
- organic carbon;
- and other characteristics,

may contain useful patterns for distinguishing between the two classes.

Machine Learning allows the algorithm to learn these relationships from the available examples.

In simplified form, we want to learn a function:

$$
f(X) \approx Y
$$

or, from a probabilistic perspective:

$$
P(Y=1|X)
$$

In other words:

> **What is the probability that a sample is potable given its characteristics?**

---

## 🌳 Why LightGBM?

**LightGBM** is an algorithm based on **Gradient Boosting with decision trees**.

The basic idea is to build multiple trees sequentially, with new trees helping correct errors made by the previous ensemble.

Simplified:

```text
Data
  ↓
Tree 1
  ↓
Identify errors
  ↓
Tree 2 → attempts to correct errors
  ↓
Identify new errors
  ↓
Tree 3 → improves predictions again
  ↓
...
  ↓
Final model
```

LightGBM is widely used for tabular Machine Learning problems and can provide strong performance in both classification and regression tasks.

In this project, it is used to learn the relationship between the physicochemical characteristics of water and the `Potability` target.

---

## 🔄 Project Pipeline

The overall project workflow can be summarized as:

```text
                    Dataset
                       │
                       ▼
              Exploratory analysis
                       │
                       ▼
              Data preprocessing
                       │
                       ▼
             Missing value handling
                       │
                       ▼
                X / y separation
                       │
                       ▼
                Train / Test Split
                  │         │
                  ▼         │
              Training      │
              LightGBM      │
                  │         │
                  └────┬────┘
                       ▼
                  Predictions
                       │
                       ▼
                   Evaluation
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          Accuracy  Precision  Recall
             │         │         │
             └─────────┼─────────┘
                       ▼
                Interpretation
```

---

## 🧹 Preprocessing

Before training, the dataset needs to be analyzed and prepared.

One important aspect of this dataset is the presence of **missing values** in some variables.

Therefore, appropriate data preprocessing is required before model training.

The preparation process may include:

1. inspecting the dataset structure;
2. performing statistical analysis;
3. identifying missing values;
4. handling missing values;
5. separating predictors from the target;
6. splitting the data into training and testing sets;
7. training the model.

---

## 📈 Model Evaluation

Since this is a binary classification problem, several metrics can be used.

### Accuracy

The proportion of correct predictions:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Although intuitive, accuracy should be analyzed together with other metrics, especially when class imbalance is present.

### Precision

Among observations predicted as positive, precision measures how many are actually positive:

$$
Precision = \frac{TP}{TP + FP}
$$

### Recall

Among observations that are actually positive, recall measures how many were correctly identified:

$$
Recall = \frac{TP}{TP + FN}
$$

### F1-Score

The harmonic mean between Precision and Recall:

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

### ROC-AUC

ROC-AUC evaluates the model's ability to distinguish between the two classes across different classification thresholds.

This is useful because it does not depend on a single threshold such as `0.5`.

---

## ⚠️ Why should we not rely only on Accuracy?

A model may achieve high accuracy while still having difficulty correctly identifying one of the classes.

In such a scenario, accuracy alone can give an overly optimistic view of model performance.

Therefore, evaluation should consider:

```text
Accuracy
   +
Precision
   +
Recall
   +
F1-Score
   +
ROC-AUC
   +
Confusion Matrix
```

It is also important to compare training and test performance to identify possible **overfitting**.

---

## 🔬 Questions This Project Aims to Answer

### 1. Is there a relationship between physicochemical characteristics and potability?

Can the model identify patterns that distinguish the two classes?

### 2. Can LightGBM learn these patterns?

Does the model achieve sufficient predictive performance?

### 3. Does the model generalize?

Does performance remain consistent on data that was not used during training?

### 4. Which features are most relevant?

Feature importance and model interpretation techniques can help identify which variables contribute most to the predictions.

### 5. What types of errors does the model make?

The confusion matrix allows us to analyze:

- True Positives;
- True Negatives;
- False Positives;
- False Negatives.

In a real-world water-quality application, different types of errors may have different consequences.

---

## 💡 Machine Learning Perspective

This project follows a fundamental principle in Data Science:

> **The model is not the starting point. The problem is.**

The reasoning process is:

```text
Problem
   ↓
Business/scientific question
   ↓
Required data
   ↓
Target definition
   ↓
Data preparation
   ↓
Model
   ↓
Evaluation
   ↓
Interpretation
   ↓
Decision
```

For this project:

```text
Problem:
Assess water potability

        ↓

Data:
Physicochemical characteristics

        ↓

Target:
Potability

        ↓

Problem type:
Binary classification

        ↓

Model:
LightGBM

        ↓

Output:
Potability probability/classification
```

---

## 🚀 Next Steps

Possible extensions of the project include:

- comparing LightGBM with Logistic Regression;
- comparing it with Decision Tree and Random Forest;
- testing XGBoost and CatBoost;
- performing cross-validation;
- tuning hyperparameters;
- analyzing feature importance;
- using SHAP for model interpretation;
- evaluating probability calibration;
- investigating class imbalance;
- comparing different classification thresholds;
- performing a deeper error analysis.

---

## 📚 References

- Kaggle — LightGBM Tutorial: https://www.kaggle.com/code/pumalin/lightgbm-tutorial/notebook
- LightGBM Documentation: https://lightgbm.readthedocs.io/
