# AICH HS-23 Gruppe 2 Competition Repository

Dieses Repository beinhaltet den Python Code der Gruppe 2 für die Kaggle Competition von Child Mind Institute Names: **Detect Sleep States**. 

Competition Link: [Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states)

## Erste Schritte

Folge diesen Schritten, um den Code auszuführen:

1. Klone das Repository auf deine lokale Maschine:
   ```bash
   git clone https://github.com/cedrege/AICH.git
   ```
2. Lade die Wettbewerbsdaten von [hier](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data) herunter und platziere sie im Verzeichnis ./data des geklonten Repositories.
3. Führe das Feature-Engineering-Notebook unter `./code/XXX` aus. Dieses Notebook erstellt die für das Training des Modells benötigten Datensätze aus den Competition-Daten.
4. Trainiere die verschiedenen Modelle mithilfe des Notebooks unter `./code/YYY`.
   * Das Notebook enthält Anweisungen dazu, entweder das beste Modell oder alle Modelle zu trainieren.
5. Abschliessend, um Ereignisse mithilfe der lokalen Scoring-Funktion der Competition zu extrahieren, führe das Notebook `./code/submission.ipynb` aus.
