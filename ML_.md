ROADMAP ML: SETTIMANA 01 – Riassunto e Setup Completo
Obiettivo settimana: Essere a tuo agio con Python per data science e iniziare a giocare con dati reali. Tempo totale: 8 ore.

📚 SETUP Ambiente (30 minuti – FARE ORA)
1. Installa Anaconda/Miniconda (gestisce tutto)

text
https://docs.conda.io/projects/miniconda/en/latest/
• Scarica Miniconda Windows/Linux (più leggero).
• Apri Anaconda Prompt → conda create -n ml_env python=3.11
• conda activate ml_env
2. Librerie core

bash
conda install numpy pandas matplotlib seaborn jupyter scikit-learn
pip install plotly openpyxl
3. VSCode + Estensioni

text
Estensioni: Python (Microsoft), Jupyter, GitLens
4. GitHub

text
Crea repo: "ml-projects-2026"

📖 RISORSE GRATUITE (PDF + Video)
Python per ML (4 ore)

1. "Python Data Science Handbook" (GRATIS PDF)
   ↓ Scarica: https://jakevdp.github.io/PythonDataScienceHandbook/
   Capitoli 2 (NumPy) + 3 (Pandas) = 80 pagine totali
   
2. Video: "Python for Data Science - Full Course" (freeCodeCamp, 6h)
   ↓ https://www.youtube.com/watch?v=ua-CiD60rO4
   Guarda 1h: 0:00‑1:00 (Python basics + NumPy/Pandas)

3. Notebook interattivo GRATIS:
   ↓ https://www.kaggle.com/learn/python

Plotting (2 ore)

1. "Python Data Science Handbook" Cap. 4 (Matplotlib)
2. Video: "Matplotlib Tutorial" (Corey Schafer, 30min)
   ↓ https://www.youtube.com/watch?v=DAQNHzq6IcY

Mini‑Progetto (2 ore)

Dataset: "Titanic" (Kaggle)
↓ https://www.kaggle.com/c/titanic/data

📅 PIANO SETTIMANALE CONCRETO (8 ore)
Lunedì/Venerdì (2h/giorno) – Python Base

text
Ore 1: Python basics (variabili, liste, dict, funzioni)
Ore 2: NumPy → array, operazioni matematiche

Esercizi:

python
import numpy as np
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
print(a + b)  # Output: [6 8 10 12]
Martedì/Sabato (2h/giorno) – Pandas

text
Ore 3: DataFrame, lettura CSV/Excel
Ore 4: Filtraggio, groupby, pivot
Esercizio Titanic:

python
import pandas as pd
df = pd.read_csv("titanic.csv")
print(df.head())
print(df['Age'].mean())  # Età media passeggeri
Mercoledì/Domenica (2h/giorno) – Plotting

text
Ore 5: Matplotlib base (line, scatter, hist)
Ore 6: Seaborn per grafici belli
Esercizio:

python
import matplotlib.pyplot as plt
df['Age'].hist(bins=20)
plt.title("Distribuzione Età Titanic")
plt.show()
Giovedì (2 ore) – Mini‑Progetto

text
Carica Titanic → EDA completo:
- Grafici: età, sesso, classe vs sopravvivenza
- Correlazioni
- Salva notebook in GitHub


🎯 OUTPUT SETTIMANA 1 (Portfolio)

text
GitHub: "ml-projects-2026/notebook1_titanic_eda.ipynb"
Contenuto:
- Descrizione dataset
- 6‑8 grafici informativi
- 3‑5 insight scritti
- README.md con link + screenshot

📋 CHECKLIST FINE SETTIMANA

text
☑️ Miniconda + env "ml_env" creato
☑️ Jupyter notebook funziona
☑️ Repo GitHub pubblico con primo notebook
☑️ Hai fatto almeno 2 grafici su dati reali
☑️ Capisci cosa sono array NumPy e DataFrame Pandas


💡 TRUCCHI per NON MOLLARE

text
1. Inizia con Titanic: dataset "perfetto", piccolo, famoso
2. Copia/incolla codice da Kaggle Notebook → modificalo
3. 1 ora al giorno = sostenibile
4. Ogni progetto = +1 CV point




ROADMAP ML: SETTIMANA 02 – Pandas + Plotting + Primo EDA
Obiettivo: Padroneggiare Pandas per analisi dati e visualizzazioni + finire il tuo primo notebook pubblico su GitHub.
Tempo: 8 ore. Dataset: Titanic (continuiamo lo stesso).

📚 RISORSE GRATUITE (PDF/Video)
Pandas (4 ore)

text
1. "Python Data Science Handbook" (GRATIS PDF)
   ↓ Capitolo 3: https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction.html
   Leggi: DataFrame basics, Indexing, Grouping, Pivot
2. Video: "Pandas Tutorial (2025 Edition)" (freeCodeCamp, 2h)
   ↓ https://www.youtube.com/watch?v=vmEHCJofslg
   Guarda 0:00–1:00 (DataFrame, filtraggio, groupby)

3. Kaggle Course (interattivo GRATIS):
   ↓ https://www.kaggle.com/learn/pandas
Plotting (2 ore)

text
1. "Python Data Science Handbook" Cap. 4 Matplotlib
   ↓ https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction.html
2. Video: "Matplotlib Tutorial" (Corey Schafer, 30min)
   ↓ https://www.youtube.com/watch?v=DAQNHzq6IcY
3. Seaborn per grafici belli:
   ↓ https://seaborn.pydata.org/tutorial.html (5 minuti)

📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – Pandas Core (4h totali)

text
Ore 1: DataFrame basics

python
import pandas as pd
df = pd.read_csv('titanic.csv')
print(df.head(3))
print(df.shape)
print(df.columns)
print(df.dtypes)

text
Ore 2: Filtraggio e selezione

python
# Filtra sopravvissuti di 1a classe
sopravvissuti_prima = df[(df['Survived'] == 1) & (df['Pclass'] == 1)]
print(sopravvissuti_prima['Age'].mean())
# Top 10 passeggeri più ricchi
df.nlargest(10, 'Fare')
Martedì/Sabato (2h/giorno) – Pandas Avanzato (4h totali)

text
Ore 3: Grouping e aggregazioni

python
# Sopravvivenza per classe
df.groupby('Pclass')['Survived'].mean()

text
Ore 4: Pivot tables e crosstab

python
pd.crosstab(df['Sex'], df['Survived'], margins=True)
df.pivot_table('Age', index='Sex', columns='Survived')
Mercoledì/Domenica (1h/giorno) – Plotting (2h totali)

text
Ore 5: Matplotlib base

python
import matplotlib.pyplot as plt
df['Age'].hist(bins=20, alpha=0.7)
plt.title('Distribuzione Età Passeggeri Titanic')
plt.xlabel('Età')
plt.ylabel('Frequenza')
plt.show()

text
Ore 6: Seaborn + multi‑plot


python
import seaborn as sns
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Sopravvivenza per Sesso')
plt.show()
Giovedì (2 ore) – PROGETTO FINALE + GitHub

text
1. Crea notebook "titanic_eda.ipynb":
   - Descrizione dataset (2 frasi)
   - 8 grafici: età, sesso, classe, porto imbarco, etc.
   - 5 insight scritti (es. "Donne 74% sopravvissute vs 19% uomini")
   - Tabella correlazioni
2. Push su GitHub:

bash
git add .
git commit -m "Titanic EDA Week 1"
git push origin main

text
3. Crea README.md:
Titanic EDA - ML Roadmap 2026
Obiettivo: Analisi esplorativa dati per primo progetto ML.
Risultati principali
• Tasso sopravvivenza: X%
• Donne: Y% vs Uomini: Z%
• 1a classe: migliori chance

text
undefined

🎯 OUTPUT SETTIMANA 2

text
✅ Notebook "titanic_eda.ipynb" pubblico su GitHub
✅ 8 grafici diversi su Titanic
✅ README con 3‑5 insight chiave
✅ Pandas fluido (groupby, pivot, filtraggio)
✅ Matplotlib/Seaborn base

📋 CHECKLIST FINE SETTIMANA

text
☑️ df.groupby('Pclass')['Survived'].mean() → funziona
☑️ df['Age'].hist() → grafico bello
☑️ GitHub repo aggiornato con README + screenshot
☑️ Tempo totale ~8 ore


💡 TRUCCHI PER RIUSCIRCI

text
1. Copia codice da Kaggle → modificalo
2. Dataset Titanic è "perfetto" (piccolo, pulito, famoso)
3. 1 ora al giorno = 7 ore/settimana
4. Screenshot di ogni grafico → mettili nel README
5. Se ti blocchi: StackOverflow + "pandas [errore]" = soluzione in 2min




ROADMAP ML: SETTIMANA 03 – Supervised Learning + Valutazione
Obiettivo: Imparare regressione e classificazione con scikit-learn + capire come valutare i modelli. Tempo: 8 ore. Continuiamo con Titanic per classificazione sopravvivenza.

📚 RISORSE GRATUITE
Supervised Learning (4 ore)

text
1. Kaggle Course GRATIS (interattivo):
   ↓ https://www.kaggle.com/learn/intro-to-machine-learning
   - Lesson 2: Regressione
   - Lesson 3: Classificazione
   
2. Video: "Machine Learning Basics" (StatQuest, 1h)
   ↓ https://www.youtube.com/watch?v=k66Q6_6maW4
   
3. PDF: "Hands‑On Machine Learning" Cap. 2 (GRATIS primo capitolo)
   ↓ https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb
Metriche e Validazione (2 ore)

text
1. Kaggle Lesson 4: Validation + Metrics
2. Video: "Classification Metrics" (StatQuest, 15min)
   ↓ https://www.youtube.com/watch?v=85neA9BhAuw

📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – Regressione (4h totali)

text
Ore 1: Regressione Lineare

python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Predire età da altre feature (esercizio)
X = df[['SibSp', 'Parch', 'Fare']]  # Features
y = df['Age'].fillna(df['Age'].mean())  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

text
Ore 2: Regressione + Overfitting

python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)
Martedì/Sabato (2h/giorno) – Classificazione Titanic (4h totali)

text
Ore 3: Logistic Regression + Decision Tree

python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# Target: Survived (0/1)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].fillna(0)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Accuracy:", logreg.score(X_test, y_test))

text
Ore 4: Random Forest

python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("RF Accuracy:", rf.score(X_test, y_test))
Mercoledì/Domenica (1h/giorno) – Metriche + Validazione (2h totali)

text
Ore 5: Metriche classificazione

python
from sklearn.metrics import classification_report, confusion_matrix
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

text
Ore 6: Cross‑validation


python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, X, y, cv=5)
print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Giovedì (2 ore) – AGGIORNAMENTO PROGETTO 1

text
1. Aggiungi sezione ML al notebook "titanic_eda.ipynb":
   - 3 modelli: Logistic, Decision Tree, Random Forest
   - Tabella confronto accuracy/CV
   - Confusion matrix plot
   - Feature importance dal RF
2. Commit e push:

bash
git add titanic_eda.ipynb
git commit -m "Week 3: ML models + evaluation metrics"
git push

text
3. Screenshot risultati → README


🎯 OUTPUT SETTIMANA 3

text
✅ Notebook aggiornato con 3 modelli ML
✅ Cross‑validation + metriche complete
✅ Feature importance visualizzata
✅ README con tabella risultati:
  | Modello | Accuracy | CV Score |
  |---------|----------|----------|
  | Logistic| 0.78    | 0.76     |
  | RF      | 0.82    | 0.80     |

📋 CHECKLIST FINE SETTIMANA

text
☑️ LogisticRegression.score() → funziona
☑️ classification_report → interpretato
☑️ cross_val_score → CV score calcolato
☑️ GitHub aggiornato con nuovi risultati
☑️ Capisci overfitting/underfitting


💡 TRUCCHI PER QUESTA SETTIMANA

text
1. Usa random_state=42 ovunque (risultati riproducibili)
2. Titanic è "perfetto" per classificazione binaria
3. Confronta SEMPRE modelli (non fidarti del primo)
4. Screenshot confusion matrix → super visivo per CV
5. Errore "convergence warning"? Aggiungi max_iter=1000




ROADMAP ML: SETTIMANA 04 – ML Avanzato + Feature Engineering + Progetto 1 Finale
Obiettivo: Imparare modelli potenti (Random Forest, Boosting) + feature engineering + chiudere Progetto 1 con pipeline completa. Tempo: 8 ore.

📚 RISORSE GRATUITE
Modelli Ensemble (4 ore)

text
1. Kaggle Course GRATIS:
   ↓ https://www.kaggle.com/learn/intermediate-machine-learning
   - Lesson 5: Random Forest
   - Lesson 6: Gradient Boosting
   
2. Video StatQuest: "Random Forest" (15min)
   ↓ https://www.youtube.com/watch?v=j7AyJL0X2bY
   
3. Video StatQuest: "XGBoost" (20min)
   ↓ https://www.youtube.com/watch?v=OtD8wVaFm6E
Feature Engineering (2 ore)

text
1. Kaggle Lesson 2: Missing Values + Imputation
2. "Feature Engineering for Machine Learning" (GRATIS PDF Cap. 1)
   ↓ https://www.traininngdata.io/feature-engineering-for-machine-learning/

📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – Random Forest + XGBoost (4h totali)

text
Ore 1: Random Forest

python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
print("RF Accuracy:", rf.score(X_test, y_test))
# Feature Importance
importances = rf.feature_importances_
features = X.columns
pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)

text
Ore 2: XGBoost (più potente)

python
!pip install xgboost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
print("XGBoost Accuracy:", xgb_model.score(X_test, y_test))
Martedì/Sabato (2h/giorno) – Feature Engineering (4h totali)

text
Ore 3: Gestione missing values

python
# Imputazione Age con mediana
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])
# Age categories
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 55, 100], labels=['Bambino', 'Giovane', 'Adulto', 'Anziano'])

text
Ore 4: Encoding categorici + Pipeline

python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Encoding Sex + Embarked
categorical_features = ['Sex', 'Embarked']
numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])
X_processed = preprocessor.fit_transform(X)
Mercoledì/Domenica (1h/giorno) – Hyperparameter Tuning (2h totali)

text
Ore 5: GridSearchCV

python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

text
Ore 6: Cross‑validation completa


python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
Giovedì (2 ore) – PROGETTO 1 FINALE

text
1. AGGIORNA "titanic_eda.ipynb":
   - Sezione 4: "Modelli Avanzati"
     - RF + XGBoost
     - Feature engineering (Age_Group, FamilySize = SibSp+Parch)
     - GridSearch + CV scores
     - Tabella finale:
       
       | Modello | CV Score | Feature Importanti |
       |---------|----------|--------------------|
       | Logistic| 0.78     | Pclass, Sex        |
       | RF      | 0.82     | Sex, Fare, Pclass  |
       | XGBoost | 0.83     | Sex, Pclass, Age   |
   - Conclusioni: "Sex e Pclass i predittori più forti"
2. Screenshot tabella + confusion matrix → README

3. Push:

bash
git add .
git commit -m "Week 4: Feature engineering + XGBoost + tuning"
git push

text

---
## 🎯 **OUTPUT SETTIMANA 4**

✅ XGBoost.score() → >0.80 accuracy su Titanic
✅ Pipeline con preprocessing (imputer + encoding)
✅ GridSearchCV con 2‑3 parametri ottimizzati
✅ Notebook con sezione "Modelli Avanzati" completa
✅ README con tabella confronto modelli

text

## 📋 **CHECKLIST FINE SETTIMANA**

☑️ xgb.XGBClassifier() → funziona
☑️ ColumnTransformer → preprocess automatico
☑️ GridSearchCV.best_score_ → tuning fatto
☑️ GitHub aggiornato con risultati migliori
☑️ Capisci perché RF > Logistic

text

---
## 💡 **TRUCCHI SETTIMANA 4**

1. XGBoost è "magic": quasi sempre batte tutto
2. FamilySize = SibSp + Parch → nuova feature potente
3. GridSearchCV piccolo (2‑3 parametri) = veloce
4. Screenshot tabella risultati → super professionale
5. Errore "XGBoost not found"? pip install xgboost







ROADMAP ML: SETTIMANA 05 – Deep Learning PyTorch
Obiettivo: Entrare nel mondo Deep Learning con PyTorch + primo CNN su immagini. Tempo: 8 ore. Nuovo dataset: Fashion MNIST (più semplice di CIFAR, ma reale).

📚 RISORSE GRATUITE
PyTorch + DL Fundamentals (4 ore)

text
1. PyTorch Tutorials UFFICIALI (GRATIS):
   ↓ https://pytorch.org/tutorials/beginner/basics/intro.html
   - Quickstart + Data Loading
   
2. Video: "PyTorch Tutorial for Beginners" (Aladdin Persson, 1h)
   ↓ https://www.youtube.com/watch?v=V_xro1bcAuA
   
3. "Deep Learning with PyTorch" (Eli Stevens) Cap. 1‑2 GRATIS:
   ↓ https://www.manning.com/books/deep-learning-with-pytorch/first‑chapter
CNN Basics (2 ore)

text
1. Video StatQuest: "CNNs Clearly Explained" (15min)
   ↓ https://www.youtube.com/watch?v=HGwBXDKFZ5I
   
2. Kaggle: "Intro to Deep Learning"
   ↓ https://www.kaggle.com/learn/intro-to-deep-learning

📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – PyTorch Basics (4h totali)

text
Ore 1: Tensori e autograd

python
import torch
import torch.nn as nn
# Tensor base
x = torch.tensor([[1., 2.], [3., 4.]])
print(x.shape, x.dtype)
# Operazioni
y = x @ x.T  # Prodotto matrici
print(y)
# Autograd
x.requires_grad_(True)
z = torch.sum(x * x)
z.backward()
print(x.grad)  # Gradiente calcolato!

text
Ore 2: Dataset + DataLoader

python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Fashion MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
for images, labels in train_loader:
    print(images.shape, labels.shape)  # torch.Size([64, 1, 28, 28])
    break
Martedì/Sabato (2h/giorno) – Prima Rete Neurale (4h totali)

text
Ore 3: Rete Dense (MLP)

python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
model = SimpleNet()
print(model)

text
Ore 4: Training Loop

python
import torch.optim as optim
import torch.nn.functional as F
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(3):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
Mercoledì/Domenica (1h/giorno) – CNN Base (2h totali)

text
Ore 5: CNN Architecture

python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

text
Ore 6: Training CNN

python
model_cnn = CNN()
optimizer = optim.Adam(model_cnn.parameters())
# Stesso training loop di sopra
Giovedì (2 ore) – PROGETTO 2 Inizio + GitHub

text
1. Crea "fashion_mnist_cnn.ipynb":
   - Carica Fashion MNIST
   - CNN base
   - Training 5 epoche
   - Accuracy su test set (>85% target)
   - Confusion matrix con seaborn
2. Push:

bash
git add .
git commit -m "Week 5: PyTorch CNN Fashion MNIST"
git push

🎯 OUTPUT SETTIMANA 5

text
✅ CNN accuracy >85% su Fashion MNIST
✅ Training loop PyTorch scritto da zero
✅ model.forward() → funziona
✅ GitHub con notebook DL funzionante
✅ Capisci Conv2D + MaxPool2D

📋 CHECKLIST FINE SETTIMANA

text
☑️ torch.nn.Conv2d() → CNN definita
☑️ DataLoader con Fashion MNIST
☑️ Loss diminuisce durante training
☑️ Test accuracy >80%
☑️ Repo aggiornato


💡 TRUCCHI SETTIMANA 5

text
1. Usa Google Colab se non hai GPU (Runtime → GPU)
2. Fashion MNIST = MNIST ma con vestiti (facile!)
3. Batch size 64 = buon compromesso velocità/memoria
4. lr=0.001 Adam = quasi sempre funziona
5. Errore CUDA? Cambia runtime su Colab
6. Screenshot loss curve → README






ROADMAP ML: SETTIMANA 06 – Transfer Learning + Progetto 2 (Computer Vision)
Obiettivo: Transfer learning con modelli pre‑addestrati + Progetto 2: Riconoscimento difetti su immagini. Tempo: 8 ore. Dataset: Tiny CIFAR‑10 o custom dataset industriale.

📚 RISORSE GRATUITE
Transfer Learning (4 ore)

text
1. PyTorch Tutorials UFFICIALI:
   ↓ https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
   
2. Video: "Transfer Learning PyTorch" (Aladdin Persson, 30min)
   ↓ https://www.youtube.com/watch?v=6zM4sU5wFis
   
3. Kaggle: "Computer Vision Course" Lesson 3
   ↓ https://www.kaggle.com/learn/computer-vision
Dataset e Metriche (2 ore)

text
Dataset: CIFAR‑10 (10 classi oggetti)
↓ torchvision.datasets.CIFAR10

Oppure dataset custom: Kaggle "Surface Defect Detection"
↓ https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – Transfer Learning ResNet (4h totali)

text
Ore 1: Carica modello pre‑addestrato


python
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
# ResNet18 pre‑addestrato
model = models.resnet18(pretrained=True)
print(model)

# Modifica ultimo layer per 10 classi (CIFAR)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

text
Ore 2: Data Augmentation + Training

python
# Transform con augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_data = datasets.CIFAR10('data', train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10('data', train=False, transform=test_transform)
Martedì/Sabato (2h/giorno) – Training + Fine‑Tuning (4h totali)

text
Ore 3: Training Loop Avanzato

python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss/len(loader), correct/total

text
Ore 4: Evaluation + Test

python
def test_epoch(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total
for epoch in range(10):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    test_acc = test_epoch(model, test_loader)
    print(f"Epoch {epoch}: Train {train_acc:.3f}, Test {test_acc:.3f}")
Mercoledì/Domenica (1h/giorno) – Progetto 2 Design (2h totali)

text
Ore 5: Scegli dataset

text
OPZIONI (scarica subito):
1. CIFAR‑10 (oggetti): torchvision.datasets.CIFAR10
2. Defect Detection: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

text
Ore 6: Setup Progetto 2

python
# Notebook "industrial_defect_detection.ipynb"
# Dataset: Casting Product Defects
# Obiettivo: Classificare "OK" vs "Defective"
# Baseline: ResNet18 transfer learning
Giovedì (2 ore) – Push + Documentazione

text
1. Completa training CIFAR‑10 (target >75% test accuracy)
2. Crea "cifar10_transfer_learning.ipynb"
3. Push:


bash
git add .
git commit -m "Week 6: PyTorch Transfer Learning ResNet CIFAR‑10"
git push


text
4. README con:
   - Accuracy finale
   - Screenshot loss curve
   - Link a Colab (se usi)

🎯 OUTPUT SETTIMANA 6

text
✅ ResNet18 accuracy CIFAR‑10 >75%
✅ Training loop con scheduler
✅ Data augmentation funzionante
✅ Secondo notebook DL su GitHub
✅ Capisci transfer learning

📋 CHECKLIST FINE SETTIMANA

text
☑️ models.resnet18(pretrained=True) → funziona
☑️ DataLoader con augmentation
☑️ Test accuracy >75%
☑️ Repo con 2 notebook DL
☑️ Loss diminuisce, accuracy sale


💡 TRUCCHI SETTIMANA 6

text
1. Usa Colab GPU (Runtime → Change → T4 GPU) → gratis!
2. LR=0.001 Adam = quasi sempre OK
3. Batch size 64 = buon equilibrio
4. Normalize sempre con ImageNet stats
5. Errore CUDA? Runtime → Restart
6. Screenshot loss curve epoch 1 vs 10 → mostra apprendimento!




ROADMAP ML: SETTIMANA 07 – Time Series + Sensori (Progetto Embedded)
Obiettivo: Time Series Analysis con focus su dati sensori (perfetto per il tuo background avionico/embedded). Tempo: 8 ore. Dataset: NASA Turbofan (predictive maintenance motori aerei).

📚 RISORSE GRATUITE
Time Series per Sensori (4 ore)

text
1. Dataset NASA Turbofan (GRATIS):
   ↓ https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan
   File: train_FD001.txt (sensori motore aereo → RUL prediction)

2. Video: "Time Series Forecasting" (StatQuest, 20min)
   ↓ https://www.youtube.com/watch?v=tepxdovS6UQ
   
3. Kaggle: "Time Series Course" Lessons 1‑2
   ↓ https://www.kaggle.com/learn/time-series
LSTM per Sequenze (2 ore)

text
1. PyTorch LSTM Tutorial:
   ↓ https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
   
2. Video: "LSTM Explained" (StatQuest, 15min)
   ↓ https://www.youtube.com/watch?v=YC8r8KNbgs8

📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – Time Series Basics (4h totali)

text
Ore 1: Carica NASA Turbofan

python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Scarica train_FD001.txt da NASA
columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_{i}' for i in range(1,22)]
df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=columns)
print(df.head())
print(df.shape)  # (20631, 26)

text
Ore 2: EDA Time Series

python
# Plot sensori per primo motore
engine1 = df[df['engine_id'] == 1]
plt.figure(figsize=(12, 8))
for i in range(1, 22):
    plt.subplot(5, 5, i)
    plt.plot(engine1['cycle'], engine1[f'sensor_{i}'])
    plt.title(f'Sensor {i}')
plt.tight_layout()
plt.show()
# RUL (Remaining Useful Life)
df['RUL'] = df.groupby('engine_id')['cycle'].transform(lambda x: len(x) - x)
Martedì/Sabato (2h/giorno) – Feature Engineering TS (4h totali)

text
Ore 3: Windowing + Features

python
def create_windows(df, window_size=20):
    features = []
    for engine in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine][['cycle'] + [f'sensor_{i}' for i in range(1,22)]]
        for i in range(len(engine_data) - window_size):
            window = engine_data.iloc[i:i+window_size].drop('cycle', axis=1).values
            features.append(np.concatenate([window.flatten(), [engine_data.iloc[i+window_size]['RUL']]]))
    return np.array(features)
X_windows = create_windows(df)
print(X_windows.shape)  # Es. (1000+, 420+1)

text
Ore 4: Normalizzazione + Split

python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_windows[:, :-1])
y = X_windows[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
Mercoledì/Domenica (1h/giorno) – LSTM Model (2h totali)

text
Ore 5: LSTM Architecture

python
import torch
import torch.nn as nn
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=21*20, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: [batch, seq_len=20, features=21]
        lstm_out, (h_n, c_n) = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Ultimo timestep
model = LSTMPredictor()
print(model)

text
Ore 6: Training Setup

python
# Reshape per LSTM: [batch, seq_len, features]
X_train_reshaped = X_train.reshape(-1, 20, 21)  # 20 timestep x 21 sensori
X_test_reshaped = X_test.reshape(-1, 20, 21)
# Tensor + Training loop (come settimana 5)
Giovedì (2 ore) – PROGETTO 2 + GitHub

text
1. Crea "turbofan_rul_prediction.ipynb":
   - EDA sensori + RUL
   - Windowing + LSTM
   - Training 20 epoche
   - RMSE su test set (<15 target)
2. Push:

bash
git add .
git commit -m "Week 7: LSTM Time Series NASA Turbofan RUL"
git push

text
3. README con:
   - Accuracy/RMSE finale
   - Plot predizioni vs reali
   - Screenshot modello LSTM

🎯 OUTPUT SETTIMANA 7

text
✅ LSTM RMSE <15 su Turbofan dataset
✅ Windowing per time series sensori
✅ Training loop LSTM funzionante
✅ Terzo notebook su GitHub (embedded‑relevant!)
✅ Portfolio con ML classico + DL

📋 CHECKLIST FINE SETTIMANA

text
☑️ NASA Turbofan dataset caricato
☑️ LSTM.forward() → funziona
☑️ RMSE calcolato su test set
☑️ Plot predizioni vs ground truth
☑️ Repo aggiornato


💡 TRUCCHI SETTIMANA 7

text
1. NASA Turbofan = dataset "embedded perfetto" (sensori reali)
2. Window size 20 = buon compromesso
3. LSTM hidden_size=64 = leggero ma efficace
4. RMSE <15 = buon risultato per baseline
5. Colab GPU per training veloce
6. Plot: plt.scatter(y_test, y_pred) → diagonale = perfetto!




ROADMAP ML: SETTIMANA 08 – MLOps Light + Progetto 1 Pipeline + GitHub Pro
Obiettivo: Pipeline completa (preprocessing + modello + API) + repo professionale. Tempo: 8 ore. Riprendiamo Titanic per creare un servizio predittivo deployabile.

📚 RISORSE GRATUITE
Pipeline + FastAPI (4 ore)

text
1. FastAPI Tutorial UFFICIALI (GRATIS):
   ↓ https://fastapi.tiangolo.com/tutorial/
   Prima 3 sezioni (30min)
   
2. Video: "Deploy ML Model with FastAPI + Docker" (Nicholas Renotte, 1h)
   ↓ https://www.youtube.com/watch?v=1vV07f1hH_Q
   
3. "ML Pipelines with scikit‑learn" (GRATIS):
   ↓ https://scikit-learn.org/stable/modules/compose.html#pipeline
Docker Base (2 ore)

text
1. Docker Tutorial UFFICIALE:
   ↓ https://docs.docker.com/get-started/
   Solo "Hello World" + "Containerizzare app"
   
2. Dockerfile per ML:
   ↓ https://towardsdatascience.com/docker-for-data-science-4d6e6e3d5f4a

📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – Pipeline Scikit‑Learn (4h totali)

text
Ore 1: Pipeline completa Titanic

python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
# Preprocessing
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Sex', 'Embarked']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
# Pipeline completa
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
model_pipeline.fit(X_train, y_train)
print("Pipeline Accuracy:", model_pipeline.score(X_test, y_test))

text
Ore 2: Save/Load Pipeline

python
import joblib
# Salva
joblib.dump(model_pipeline, 'titanic_pipeline.pkl')
# Carica e usa
loaded_pipeline = joblib.load('titanic_pipeline.pkl')
prediction = loaded_pipeline.predict([[3, 25, 0, 71.2833, 'female', 'C']])
print("Prediction:", prediction)  # [1] = sopravvive
Martedì/Sabato (2h/giorno) – FastAPI Service (4h totali)

text
Ore 3: FastAPI base

python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
app = FastAPI()
model = joblib.load('titanic_pipeline.pkl')
class Passenger(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex: str
    Embarked: str
@app.post("/predict")
def predict_survival(passenger: Passenger):
    data = np.array([[passenger.Pclass, passenger.Age, passenger.SibSp, 
                      passenger.Parch, passenger.Fare, passenger.Sex, passenger.Embarked]])
    prediction = model.predict(data)[0]
    return {"survived": bool(prediction), "probability": float(model.predict_proba(data)[0][1])}

text
Ore 4: Test API + uvicorn

bash
pip install fastapi uvicorn
uvicorn main:app --reload

text
Test: http://localhost:8000/docs → Prova endpoint!

Mercoledì/Domenica (1h/giorno) – Docker (2h totali)

text
Ore 5: Dockerfile

text
FROM python:3.11‑slim
WORKDIR /app
COPY requirements.txt .
RUN pip install ‑r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "‑‑host", "0.0.0.0", "‑‑port", "8000"]


text
Ore 6: Build e Run

bash
# requirements.txt
echo "fastapi\nuvicorn\nscikit‑learn\npandas\nnumpy\njoblib" > requirements.txt

docker build ‑t titanic‑api .
docker run ‑p 8000:8000 titanic‑api

Giovedì (2 ore) – Portfolio Update + README Pro

text
1. Crea "titanic_api_deployment.ipynb":
   - Pipeline salvata
   - Test API con requests
   - Screenshot FastAPI docs
   - Docker run screenshot
2. AGGIORNA README principale:

text
# ML Portfolio 2026 – Da Embedded a ML Engineer

## 📊 Progetto 1: Titanic Survival Prediction
✅ Pipeline completa (preprocessing + RF)
✅ FastAPI deployment
✅ Docker containerizzato
✅ Accuracy: 82.5%

**[Collegamento notebook](titanic_eda.ipynb)**  
**[API Live](http://localhost:8000/docs)**
![Pipeline Diagram](pipeline.png)

text
3. Screenshot + Push:

bash
git add .
git commit ‑m "Week 8: MLOps Pipeline + FastAPI + Docker"
git push


🎯 OUTPUT SETTIMANA 8

text
✅ Pipeline scikit‑learn completa e salvabile
✅ FastAPI endpoint funzionante
✅ Docker container che gira
✅ API deployata localhost
✅ Portfolio con MLOps reale

📋 CHECKLIST FINE SETTIMANA

text
☑️ joblib.dump(model_pipeline) → salvato
☑️ http://localhost:8000/docs → API funziona
☑️ docker run → container up
☑️ requests.post() → predizione OK
☑️ GitHub con demo screenshot


💡 TRUCCHI SETTIMANA 8

text
1. FastAPI docs automatiche = super pro!
2. Passenger(BaseModel) → validazione automatica
3. Docker slim Python = container piccolo
4. Screenshot API Swagger → perfetto per CV
5. Errore "port occupied"? Cambia porta 8080
6. Test con curl: curl ‑X POST http://localhost:8000/predict ‑H "Content‑Type: application/json" ‑d '{"Pclass":1,"Age":25,...}'




ROADMAP ML: SETTIMANA 09 – LLM Fundamentals + RAG
Obiettivo: Concetti base Large Language Models + Retrieval Augmented Generation (RAG) semplice. Tempo: 8 ore. Progetto: AI Manutentore che combina ML + testo generato.

📚 RISORSE GRATUITE
LLM Concepts (4 ore)

text
1. "LLM in a Nutshell" (GRATIS PDF):
   ↓ https://towardsdatascience.com/llm‑in‑a‑nutshell‑a‑guide‑to‑large‑language‑models‑9c5637b1b0a8
   
2. Video: "LLMs from Scratch" (Andrej Karpathy, 2h)
   ↓ https://www.youtube.com/watch?v=zjkBMFhNj_g
   Guarda prima ora (concetti, non codice)
3. OpenAI Cookbook RAG:
   ↓ https://cookbook.openai.com/examples/vector_databases/rag_with_openai
RAG + LangChain Lite (2 ore)

text
1. LangChain Quickstart:
   ↓ https://python.langchain.com/docs/get_started/quickstart
   
2. Video: "RAG Explained" (15min)
   ↓ https://www.youtube.com/watch?v=jzwFC3b1fq8

📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – LLM Concepts (4h totali)

text
Ore 1: Tokenizzazione + Embedding

python
import openai
from openai import OpenAI
client = OpenAI(api_key="your‑api‑key")  # O usa Grok/xAI

# Embedding
response = client.embeddings.create(
    input="Il motore mostra vibrazioni anomale nei sensori 3‑5",
    model="text‑embedding‑3‑small"
)
embedding = response.data[0].embedding
print(len(embedding))  # 1536 dimensioni


text
Ore 2: Prompt Engineering

python
def simple_chat(prompt):
    response = client.chat.completions.create(
        model="gpt‑4o‑mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

print(simple_chat("Spiega predictive maintenance in 3 frasi"))
Martedì/Sabato (2h/giorno) – RAG Base (4h totali)

text
Ore 3: Vector Store semplice (FAISS)

bash
pip install faiss‑cpu sentence‑transformers


python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
model = SentenceTransformer('all‑MiniLM‑L6‑v2')

# Documenti di "manutenzione" (tuoi testi)
docs = [
    "Vibrazioni alte nei sensori 3‑5 indicano usura cuscinetti",
    "Temperatura motore sopra 80°C richiede spegnimento",
    "Pressione olio bassa < 2.5 bar → rischio guasto pompa"
]

embeddings = model.encode(docs)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))

text
Ore 4: Retrieval + Generation

python
def rag_query(query, k=2):
    query_emb = model.encode([query])
    distances, indices = index.search(query_emb.astype('float32'), k)
    
    context = "\n".join([docs[i] for i in indices[0]])
    
    prompt = f"""Basandoti su questi dati tecnici:
{context}
Query: {query}
Rispondi come tecnico manutentore esperto:"""
    
    response = simple_chat(prompt)
    return response
print(rag_query("Il sensore 3 mostra valori alti, cosa controllare?"))
Mercoledì/Domenica (1h/giorno) – Integrazione ML + LLM (2h totali)

text
Ore 5: Carica tuo modello Turbofan

python
import joblib
rul_model = joblib.load('turbofan_lstm_model.pkl')  # Dal Progetto 2
def predict_rul(sensor_data):
    # Simula predizione RUL
    prediction = rul_model.predict(sensor_data.reshape(1, -1))[0]
    return prediction

text
Ore 6: AI Manutentore Completo

python
def ai_maintenance_assistant(sensor_readings, query):
    # 1. ML predizione RUL
    rul = predict_rul(sensor_readings)
    
    # 2. RAG con contesto + RUL
    rag_prompt = f"""Dati sensori attuali:
RUL stimato: {rul:.1f} cicli
    
Query tecnico: {query}
    
Rispondi:"""
    
    response = simple_chat(rag_prompt)
    return response
print(ai_maintenance_assistant([80, 75, 90, ...], "Cosa consigliare al pilota?"))
Giovedì (2 ore) – PROGETTO 3 + Portfolio

text
1. Crea "ai_maintenance_assistant.ipynb":
   - RAG system funzionante
   - Integrazione con tuo modello ML
   - 5 query di test con risposta
2. Push:

bash
git add .
git commit -m "Week 9: LLM RAG + AI Maintenance Assistant"
git push

text
3. README Master Update:

text
## 🤖 Progetto 3: AI Manutentore (ML + LLM)
✅ RAG con documenti tecnici
✅ Integrazione modello LSTM RUL
✅ API pronta per produzione

**Demo**: "Sensori anomali → RUL 150 cicli → Consiglio: controllare cuscinetti"


text

---
## 🎯 **OUTPUT SETTIMANA 9**

✅ RAG system funzionante
✅ Embedding + FAISS retrieval
✅ Integrazione ML + LLM
✅ AI Assistant per manutenzione
✅ Quarto progetto su GitHub

text

## 📋 **CHECKLIST FINE SETTIMANA**

☑️ client.chat.completions.create() → risposta LLM
☑️ faiss.search() → retrieval OK
☑️ ai_maintenance_assistant() → integra ML+LLM
☑️ 5 test query con risposta coerente
☑️ Portfolio con 4 progetti diversi

text

---
## 💡 **TRUCCHI SETTIMANA 9**

6. Usa API key gratuita (Grok, Anthropic, o OpenAI $5 credit)
7. SentenceTransformer 'all‑MiniLM‑L6‑v2' = veloce e gratis
8. 3‑5 documenti bastano per demo efficace
9. Screenshot chat LLM → super visivo
10. Errore API key? Registrati su platform.openai.com











ROADMAP ML: SETTIMANA 10 – MLOps Avanzato + Deploy Cloud Gratuito
Obiettivo: Docker + Cloud deployment del tuo AI Manutentore (ML + LLM) su servizio gratuito. Tempo: 8 ore.

📚 RISORSE GRATUITE
Docker + Cloud ML (4 ore)

text
1. Render.com (deploy GRATIS):
   ↓ https://render.com/docs/deploy-fastapi
   Piano free: 750 ore/mese → perfetto!

2. Video: "Deploy ML Model to Cloud FREE" (Mervin Praison, 20min)
   ↓ https://www.youtube.com/watch?v=7Q25‑S7jzgs
   
3. Railway.app (alternativa Render):
   ↓ https://railway.app/
Monitoring + Logging (2 ore)

text
1. MLflow Quickstart:
   ↓ https://mlflow.org/docs/latest/quickstart.html
   
2. Weights & Biases (wandb) Free:
   ↓ https://wandb.ai/site

📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – Docker Pro (4h totali)

text
Ore 1: Dockerfile ottimizzato

text
FROM python:3.11‑slim

WORKDIR /app
# Installa solo dipendenze necessarie
COPY requirements.txt .
RUN pip install --no‑cache‑dir ‑‑upgrade pip && \
    pip install --no‑cache‑dir ‑r requirements.txt

COPY . .
# MLflow per logging
EXPOSE 8000
CMD ["uvicorn", "main:app", "‑‑host", "0.0.0.0", "‑‑port", "8000"]


text
requirements.txt:

text
fastapi==0.115.0
uvicorn[standard]==0.31.0
scikit‑learn==1.5.2
joblib==1.4.2
pandas==2.2.3
numpy==2.1.1
sentence‑transformers==3.1.1
faiss‑cpu==1.8.0
openai==1.51.2
mlflow==2.16.2


text
Ore 2: Docker Compose + Multi‑container


text
# docker‑compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      ‑ "8000:8000"
    environment:
      ‑ OPENAI_API_KEY=${OPENAI_API_KEY}
  
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.2
    ports:
      ‑ "5000:5000"
    command: mlflow ui ‑‑backend‑store‑uri sqlite:///mlflow.db ‑‑default‑artifact‑root ./mlruns


text
docker compose up
Martedì/Sabato (2h/giorno) – MLflow Tracking (4h totali)

text
Ore 3: Integra MLflow nel training

python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Titanic Pipeline")
with mlflow.start_run():
    model_pipeline.fit(X_train, y_train)
    accuracy = model_pipeline.score(X_test, y_test)
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model_pipeline, "model")
    
    print(f"Run logged! Accuracy: {accuracy}")

text
Ore 4: MLflow UI + Reproducibility

text
Vai su http://localhost:5000 → vedi esperimenti!
Carica il migliore modello:
best_run = mlflow.get_run(mlflow.active_run().info.run_id)

Mercoledì/Domenica (1h/giorno) – Deploy Cloud (2h totali)

text
Ore 5: Render.com deployment

text
1. Crea account Render.com (gratuito)
2. New → Web Service → GitHub repo
3. Settings:
   Build: docker build .
   Start: docker compose up
   Env var: OPENAI_API_KEY=sk‑...


text
Ore 6: Test API live

bash
curl ‑X POST https://your‑app.onrender.com/predict \
  ‑H "Content‑Type: application/json" \
  ‑d '{"Pclass":1,"Age":25,"SibSp":0,"Parch":0,"Fare":71,"Sex":"female","Embarked":"C"}'

Giovedì (2 ore) – Portfolio Finale + Video Demo

text
1. Crea "deployment_guide.md":

text
# 🚀 AI Manutentore Deployed

## Live Demo
**[API Live](https://your‑app.onrender.com/docs)**

## Stack Tecnico
‑ FastAPI + Docker
‑ MLflow tracking
‑ Render.com (gratuito)
‑ ML (XGBoost) + LLM (GPT)

## Screenshot
![MLflow UI](mlflow.png)
![FastAPI Docs](api.png)

text
2. Registra 30s video demo (schermo + voce):
   ‑ Apri MLflow UI
   ‑ Chiama API live
   ‑ Mostra predizione

3. Push finale:

bash
git add .
git commit ‑m "Week 10: MLOps Complete + Cloud Deploy"
git push


🎯 OUTPUT SETTIMANA 10

text
✅ Docker multi‑container
✅ MLflow esperimenti tracciati
✅ API deployata su cloud GRATUITO
✅ Video demo 30s
✅ Portfolio production‑ready

📋 CHECKLIST FINE SETTIMANA

text
☑️ docker compose up → tutto funziona
☑️ MLflow UI → esperimenti visibili
☑️ https://your‑app.onrender.com → API live
☑️ curl POST → predizione corretta
☑️ Video demo registrato


💡 TRUCCHI SETTIMANA 10

text
1. Render.com free tier = 750 ore/mese → sempre on!
2. MLflow sqlite = zero configurazione
3. OPENAI_API_KEY env var → sicuro
4. Video: Loom o OBS Studio (gratis)
5. Errore Docker build? requirements.txt con pin versioni
6. API link nel CV → super impressionante!



ROADMAP ML: SETTIMANA 11 – Portfolio Polish + Colloqui Prep
Obiettivo: Rifinisci portfolio + preparati ai colloqui ML mid‑level + applica a 10 posizioni. Tempo: 8 ore.

📚 RISORSE GRATUITE
Portfolio + CV (4 ore)

text
1. "ML Engineer Portfolio Guide 2026"
   ↓ https://towardsdatascience.com/ml‑engineer‑portfolio‑2026
   
2. GitHub README Pro Tips:
   ↓ https://github.com/abhisheknaiidu/awesome‑github‑profile‑readme
   
3. CV Template ML Engineer:
   ↓ https://novoresume.com/career‑blog/machine‑learning‑engineer‑resume
Colloqui ML (4 ore)

text
1. "ML System Design Interview" (Alex Xu)
   ↓ https://www.youtube.com/watch?v=5‑C5d1‑0p8s
   
2. LeetCode ML-tagged:
   ↓ https://leetcode.com/problemset/all/?search=ml
   
3. "Cracking ML Interviews" GRATIS:
   ↓ https://github.com/ashishpatel26/ML‑Interview‑Prep

📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – Portfolio Master (4h totali)

text
Ore 1: README Principale

text
# 🚀 ML Engineer Portfolio – Da Embedded ad AI

[![Portfolio](https://img.shields.io/badge/Portfolio‑Live‑brightgreen)](https://your‑render‑app.onrender.com)
[![MLflow](https://img.shields.io/badge/MLflow‑Tracking‑blue)](http://localhost:5000)

## Progetti
### 🤖 **AI Manutentore (ML + LLM)**
‑ XGBoost pipeline + RAG
‑ FastAPI + Docker + Render
‑ **Live Demo**: [API](https://your‑app.onrender.com/docs)
‑ Accuracy: 83.2%

### 🛫 **NASA Turbofan RUL Prediction**
‑ LSTM su dati reali sensori
‑ Windowing + feature engineering
‑ RMSE: 12.8

### 👗 **Fashion MNIST CNN**
‑ Transfer Learning ResNet18
‑ Accuracy: 89.5%

### 🚢 **Titanic Survival (Pipeline)**
‑ End‑to‑end MLOps
‑ Scikit‑learn + FastAPI

## Tech Stack
![Python](https://img.shields.io/badge/Python‑3.11‑3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch‑2.3‑EE4C2C)
![FastAPI](https://img.shields.io/badge/FastAPI‑0.115‑005571)
![Docker](https://img.shields.io/badge/Docker‑27‑2496ED)


text
Ore 2: Screenshot + Badge

text
1. Screenshot ogni progetto → cartella images/
2. Badge generator: https://shields.io
3. Aggiungi a ogni notebook: ![Badge](badge.png)
Martedì/Sabato (2h/giorno) – CV + LinkedIn (4h totali)

text
Ore 3: CV ML Engineer

text
[Nome Cognome]
Embedded → ML Engineer | 3+ anni esperienza sistemi real‑time

✈️ **NASA Turbofan RUL Prediction** (LSTM)
‑ Predizione vita residua motori aerei da 21 sensori
‑ Windowing + LSTM: RMSE 12.8 su dati NASA reali
‑ [GitHub](link)

🤖 **AI Manutentore Deployed** (ML + LLM RAG)
‑ Pipeline XGBoost + FastAPI + Docker + Render.com
‑ Integrazione ML predittivo con LLM per report tecnici
‑ Live API: [your‑app.onrender.com](link)

👨‍💻 **Stack**: Python, PyTorch, scikit‑learn, FastAPI, Docker, MLflow

💼 **Esperienza**: 3 anni Embedded C/C++ Linux avionica


text
Ore 4: LinkedIn Profile

text
1. Headline: "Embedded → ML Engineer | Predictive Maintenance | LLM RAG | FastAPI"
2. About: 3 paragrafi (passato embedded → transizione ML → progetti)
3. Featured: Link GitHub + API live
4. Experience: aggiungi "ML Engineer Self‑taught" con progetti

Mercoledì/Domenica (1h/giorno) – Colloqui Prep (2h totali)

text
Ore 5: Domande ML Base

text
Domande da sapere a memoria:
1. "Spiega overfitting/underfitting"
2. "Cross‑validation vs train/test split?"
3. "XGBoost vs Random Forest?"
4. "Pipeline scikit‑learn?"


text
Ore 6: Domande System Design

text
"Esempio: Predictive maintenance per flotte aeree"
‑ Dati: sensori real‑time
‑ Modello: LSTM + feature engineering
‑ Deploy: FastAPI + Docker + Kubernetes
‑ Monitoring: MLflow + Prometheus

Giovedì (2 ore) – APPLICAZIONI + Mock Interview

text
1. Cerca su LinkedIn/Indeed:
"ML Engineer" OR "Data Scientist" OR "AI Engineer" Milano 0‑3 anni

text
2. Candidati a **10 posizioni** con:
‑ CV customizzato
‑ Cover letter: "La mia transizione embedded→ML + progetti live"

3. Mock interview (registra 5min):
‑ "Parlami del tuo progetto AI Manutentore"
‑ Rispondi spiegando architettura + demo live


🎯 OUTPUT SETTIMANA 11

text
✅ README portfolio professionale con badge
✅ CV ML Engineer pronto
✅ LinkedIn ottimizzato con progetti live
✅ 10 candidature inviate
✅ Video mock interview 5min registrato

📋 CHECKLIST FINE SETTIMANA

text
☑️ GitHub README con live links
☑️ CV con 3 progetti principali
☑️ LinkedIn "Open to Work" + Featured
☑️ 10 application inviate
☑️ Risposte pronte a 5 domande ML base


💡 TRUCCHI SETTIMANA 11

text
1. Badge Shields.io = super pro
2. CV 1 pagina, progetti in alto
3. LinkedIn: pubblica "Ho deployato il mio primo ML+LLM system!"
4. Candidature: personalizza "Ho visto che fate predictive maintenance..."
5. Mock: registrati, riguardati, migliora






ROADMAP ML: SETTIMANA 12 – Colloqui Simulati + Networking + Applicazioni Avanzate
Obiettivo: Preparazione colloqui ML mid‑level + networking strategico + candidature mirate. Tempo: 8 ore.

📚 RISORSE GRATUITE
Colloqui ML Engineer (4 ore)

text
1. "Machine Learning Interviews" (Sean Kernon):
   ↓ https://www.mlinterviewsbook.com/ (anteprima GRATIS)
   
2. LeetCode ML Problems:
   ↓ https://leetcode.com/problemset/?search=ml
   
3. Pramp (mock interviews GRATIS):
   ↓ https://www.pramp.com/
Networking + Applicazioni (2 ore)

text
1. LinkedIn Messaging Templates:
   ↓ https://github.com/amitness/ml‑engineer‑roadmap
   
2. Italian ML Community:
   ↓ Telegram: t.me/ML_Italia
   ↓ Discord: Italian Data Scientists


📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – Technical Interview Prep (4h totali)

text
Ore 1: Domande ML Theory (prepara risposte 2‑3min)


text
Domande da sapere:
1️⃣ **"Overfitting vs Underfitting?"**
"Overfitting: modello troppo complesso, memorizza train ma generalizza male → alta varianza. 
Underfitting: modello troppo semplice → errore alto anche su train → bias alto.
Soluzioni: regolarizzazione, più dati, cross‑validation."

2️⃣ **"Feature Engineering nel tuo progetto?"**
"Titanic: ho creato 'FamilySize = SibSp + Parch' (+5% accuracy), 
Age_Group categorica, imputazione mediana Age. 
Turbofan: windowing 20 timestep sui sensori."

3️⃣ **"Pipeline end‑to‑end?"**
"Preprocessing → Model → API → Docker → Cloud. 
Esempio: Titanic pipeline con ColumnTransformer + XGBoost → FastAPI → Render.com live."


text
Ore 2: Coding Interview (LeetCode style)

text
Problemi facili ML (30min ciascuno):
1. Implementa gradient descent da zero (NumPy)
2. Confusion matrix + F1 score manuali
3. Data preprocessing function (missing + scaling)

python
def gradient_descent(X, y, lr=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(epochs):
        y_pred = X.dot(w) + b
        dw = X.T.dot(y_pred - y) / len(y)
        db = np.sum(y_pred - y) / len(y)
        w -= lr * dw
        b -= lr * db
    return w, b
Martedì/Sabato (2h/giorno) – System Design + Mock (4h totali)

text
Ore 3: ML System Design (prepara 5min)

text
Caso: "Predictive maintenance per flotte aeree"
**Architettura**:
Dati sensori → Kafka → Preprocessing (Docker) →
LSTM Model (MLflow) → RUL Prediction →
LLM RAG → Report tecnico → Slack/Email

text
**Componenti**:
‑ Ingestion: Kafka per streaming
‑ Storage: S3/PostgreSQL
‑ Model serving: Triton/FastAPI
‑ Monitoring: Prometheus + Grafana


text
Ore 4: Mock Interview Completa (registra 15min)

text
Struttura:
1. "Presentati" (2min) → embedded → ML projects
2. "Parlami del tuo progetto preferito" (5min) → AI Manutentore
3. "Domanda tecnica" (3min) → overfitting
4. "System design" (3min) → pred maint
5. "Domande per noi?" (2min)

Mercoledì/Domenica (1h/giorno) – Networking (2h totali)

text
Ore 5: LinkedIn Outreach (20 messaggi)

text
Template 1 – Recruiter:

"Ciao [Nome],
sono un Embedded Engineer (3 anni avionica) in transizione verso ML.
Ho deployato AI Manutentore (ML+LLM) live su cloud.
Cerco ruoli ML Engineer mid.
Il vostro team [Azienda] fa progetti edge computing?
Portfolio: [GitHub]
Grazie!
[Nome]"

text
undefined
Template 2 – ML Engineer in azienda target:

text
"Ciao [Nome],
il tuo post su MLOps mi ha colpito! 
Sto sviluppando predictive maintenance con LSTM + RAG per sensori embedded.
Hai consigli per chi viene da real‑time systems?
Portfolio: [GitHub]

Grazie,
[Nome]"

text
Ore 6: Community Italiane

text
1. Telegram ML_Italia → presentati + link portfolio
2. LinkedIn Groups: "Data Science Italia", "AI Italy"
3. Evento gratuito: cerca "meetup ML Milano" febbraio

Giovedì (2 ore) – 15 CANDIDATURE MIRATE

text
Obiettivo: 15 applicazioni mirate (non spam)
Target aziende Milano:
‑ Skytechnology (embedded ML)
‑ VisLab (computer vision aerospace)
‑ Sidel (industrial ML)
‑ Thales (avionics ML)
‑ AYES Consulting (R&D ML)
‑ Startup autonomous systems

Per ogni candidatura:
1. CV personalizzato (1 bullet progetto rilevante)
2. Cover letter breve: "Il mio background avionico + progetti ML = perfect fit"

🎯 OUTPUT SETTIMANA 12

text
✅ Risposte scritte a 10 domande ML
✅ Mock interview 15min registrata
✅ 20 messaggi LinkedIn inviati
✅ 15 candidature mirate
✅ Contatti community stabilite

📋 CHECKLIST FINE SETTIMANA

text
☑️ 10 domande ML con risposta 2min
☑️ System design pred maint (schema)
☑️ Mock video caricato (privato OK)
☑️ 20 LinkedIn messages inviati
☑️ 15 application con cover personalizzata
☑️ Almeno 1 risposta community


💡 TRUCCHI SETTIMANA 12

text
1. Mock: registrati, riguardati, ripeti 3 volte
2. LinkedIn: 10 messages/giorno max (non spam)
3. Candidature: qualità > quantità
4. Community: "Ciao, nuovo nel gruppo, feedback portfolio?"
5. Rispondi recruiter in <2h
6. Portfolio link OVUNQUE (firma email, CV, LinkedIn)







ROADMAP ML: SETTIMANA 13 – Follow‑up + Advanced Topics + Seconda Onda Candidature
Obiettivo: Gestire risposte + apprendere MLOps avanzato + seconda ondata 20 candidature. Tempo: 8 ore.

📚 RISORSE GRATUITE
Follow‑up + Communication (2 ore)

text
1. "Email Templates per Tech Jobs"
   ↓ https://github.com/AndrewStark69/Tech‑Job‑Hunt‑Templates
   
2. Video: "How to Follow Up After Interviews" (Don Georgevich)
   ↓ https://www.youtube.com/watch?v=5‑yq‑W8z8z0
MLOps Avanzato (4 ore)

text
1. MLflow Advanced:
   ↓ https://mlflow.org/docs/latest/tracking.html#experiments‑and‑runs
   
2. DVC (Data Versioning):
   ↓ https://dvc.org/doc/start

📅 PIANO SETTIMANALE (8 ore)
Lunedì/Venerdì (2h/giorno) – Follow‑up Strategico (4h totali)

text
Ore 1: Analizza risposte ricevute

text
Per ogni recruiter/azienda:
1. **No risposta** (7+ giorni): Invia follow‑up


text
Subject: Follow‑up [Nome Azienda] ML Engineer

Ciao [Nome],
spero tutto bene. Ti scrivo per un gentile follow‑up sulla mia candidatura del [data].

Nel frattempo ho completato un nuovo progetto: predictive maintenance con LSTM su dati NASA Turbofan (RMSE 12.8).
Resto disponibile per un colloquio.
Grazie,
[Nome]
[GitHub] [LinkedIn]

text
Ore 2: Thank You + Next Steps

text
Per chi ti ha risposto:

text
Subject: Grazie per l’intervista [Nome Azienda]

Ciao [Nome],
grazie per il tempo dedicato oggi. 
Ho apprezzato molto la discussione su [argomento specifico].
Ho approfondito [domanda ricevuta] e ti allego note.
Resto a disposizione per chiarimenti.
Cordiali saluti,
[Nome]

text

### **Martedì/Sabato (2h/giorno) – MLOps: MLflow + DVC (4h totali)**

Ore 3: MLflow Model Registry

text
```python
import mlflow
import mlflow.sklearn
# Registra il migliore modello
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Titanic_Production")
with mlflow.start_run():
    model_pipeline.fit(X_train, y_train)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("test_accuracy", model_pipeline.score(X_test, y_test))
    
    # Registra come Production
    mlflow.sklearn.log_model(model_pipeline, "model")
    mlflow.register_model("runs:/"+mlflow.active_run().info.run_id+"/model", "TitanicModel")

text
Ore 4: DVC per Data Versioning

bash
pip install dvc
dvc init
echo "data/raw/titanic.csv" >> .dvcignore
dvc add data/raw/titanic.csv
git add data/raw/titanic.csv.dvc
git commit -m "Version data con DVC"
dvc push
Mercoledì/Domenica (1h/giorno) – Advanced Topics (2h totali)

text
Ore 5: Model Monitoring

python
# Drift detection semplice
def detect_drift(old_data, new_data, threshold=0.1):
    from scipy.stats import ks_2samp
    stat, p_value = ks_2samp(old_data.flatten(), new_data.flatten())
    return p_value < threshold
# Esempio su sensori
print("Drift detected:", detect_drift(X_train[:100], X_test[:100]))

text
Ore 6: Explainability (SHAP)

bash
pip install shap

python
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
Giovedì (2 ore) – Seconda Onda 20 Candidature

text
1. Target specifico ML mid‑level Milano:
   ‑ "ML Engineer" 1‑4 anni
   ‑ "Data Scientist" autonomous/industrial
   ‑ "AI Engineer" edge computing
   
2. Personalizza per settore:
   ‑ Aerospace: VisLab, Leonardo, Thales
   ‑ Industrial: Sidel, E80 Group
   ‑ Consulting: AYES, BCG Gamma

3. Cover letter template:

text
La mia esperienza 3 anni embedded avionica + progetti ML (LSTM NASA Turbofan, AI Manutentore deployato) mi rende candidato ideale per [ruolo].
Portfolio live: [GitHub]
Disponibile colloquio immediato.
Grazie,

text

***
## 🎯 **OUTPUT SETTIMANA 13**

✅ 10 follow‑up inviati
✅ MLflow Model Registry con versione production
✅ DVC data versioning setup
✅ SHAP explainability su un modello
✅ 20 nuove candidature mirate
✅ Risposte pronte per 2a ondata colloqui

text

## 📋 **CHECKLIST FINE SETTIMANA**

☑️ mlflow.register_model() → versione registrata
☑️ dvc push → data versionato
☑️ SHAP summary plot generato
☑️ 10 follow‑up + 20 nuove candidature
☑️ Template system design pronti

text

***
## 💡 **TRUCCHI SETTIMANA 13**

11. Follow‑up dopo 7 giorni esatti = professionale
12. MLflow UI → screenshot "Production model ready"
13. SHAP plot = super impressionante nei colloqui
14. Candidature: 5/giorno max (qualità)
15. Rispondi recruiter entro 2h
16. "Ho appena deployato X" = freschezza conta!
