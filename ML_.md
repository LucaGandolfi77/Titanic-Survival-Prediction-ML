ROADMAP ML: SETTIMANA 1 – Riassunto e Setup Completo
Obiettivo settimana: Essere a tuo agio con Python per data science e iniziare a giocare con dati reali. Tempo totale: 8 ore.

📚 SETUP Ambiente (30 minuti – FARE ORA)
1. Installa Anaconda/Miniconda (gestisce tutto)
text
https://docs.conda.io/projects/miniconda/en/latest/
Scarica Miniconda Windows/Linux (più leggero).

Apri Anaconda Prompt → conda create -n ml_env python=3.11

conda activate ml_env

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
text
1. "Python Data Science Handbook" (GRATIS PDF)
   ↓ Scarica: https://jakevdp.github.io/PythonDataScienceHandbook/
   Capitoli 2 (NumPy) + 3 (Pandas) = 80 pagine totali
   
2. Video: "Python for Data Science - Full Course" (freeCodeCamp, 6h)
   ↓ https://www.youtube.com/watch?v=ua-CiD60rO4
   Guarda 1h: 0:00‑1:00 (Python basics + NumPy/Pandas)

3. Notebook interattivo GRATIS:
   ↓ https://www.kaggle.com/learn/python
Plotting (2 ore)
text
1. "Python Data Science Handbook" Cap. 4 (Matplotlib)
2. Video: "Matplotlib Tutorial" (Corey Schafer, 30min)
   ↓ https://www.youtube.com/watch?v=DAQNHzq6IcY
Mini‑Progetto (2 ore)
text
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