# UMA-projekt
## Autorzy
Andrzej Tokajuk i Aleksander Szymczyk
## Zmodyfikowany las losowy w zadaniu klasyfikacji.
### Treść zadania
Podczas budowy każdego z drzew testy (z pełnego zbioru testów) wybieramy za pomocą turnieju: https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/turniejRuletka.pdf, czyli wybieramy losowo 2 testy, liczymy ich jakość, stosujemy ten z wyższą jakością.
Do wstępnych testów polecam zbiór danych z zadaniem klasyfikacji grzybów: https://archive.ics.uci.edu/dataset/73/mushroom, na którym powinno dać się uzyskać wyniki w okolicy 100%.
Do dalszych testów należy znaleźć i pobrać inne zbiory danych (na wskazanej stronie lub w Kaggle).
Przed rozpoczęciem realizacji projektu proszę zapoznać się z zawartością: https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/index.html
### Struktura projektu
W folderze `src` umieszczono wszystkie pliki zawierające kod źródłowy projektu. Struktura folderu i funkcjonalności poszczególnych plików są następujące:
- W pliku `random_forest.py` - implementacja algorytmu lasu losowego wykorzystująca drzewa typu CART.
- W pliku `cart_tree.py` - implementacja drzewa CART z selekcją turniejową.
- W pliku `tree_node.py` - znajduje się definicja pojedynczej gałęzi drzewa.
- W pliku `baseline.py` - biblioteczna wersja algorytmu lasu losowego z biblioteki Scikit-learn.
- W pliku `test_implementation.py` - funkcje sprawdzające poprawność algorytmu na prostym zbiorze danych **Mushroom**.
- W pliku `run_experiments.py` - cały pipeline przytowujący dane oraz przeprowadzający eksperymenty.
- W pliku `datasets_config.json` - konfiguracja ścieżek do plików oraz parametrów ładowania zbiorów danych.
- W plikach `analyze_results.py` oraz `analyze_tuning_results.py` - funkcje pomocnicze do generowania wykresów na podstawie wyników eksperymentów.

W podfolderze `results` przechowywane są pliki `.json` z wynikami przeprowadzonych eksperymentów dla poszczególnych zbiorów danych.

W podfolderze `plots` znajdują się wykresy stworzone do analizy wyników eksperymentów.
