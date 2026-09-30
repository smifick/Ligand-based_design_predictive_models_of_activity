# Ligand-based дизайн: предиктивная модель активности
Проектная деятельность СНО НИЯУ МИФИ
Таблица для того, чтобы отслеживать активность: https://docs.google.com/spreadsheets/d/1qLNoeubCFvYyJ_7GTCXLU9o1kCiip3jgTxjwJ7ndlTs/edit?usp=sharing

Что должна делать модель:
СТРУКТУРА МОЛЕКУЛЫ  →  [ МОДЕЛЬ ]  →  ЧИСЛО (сила ингибирования EGFR)
Сначала пишется парсер main.py, где каждая строка выводит все параметры для белка egfr, что эквивалентно  https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL203&standard_type=IC50&limit=100&offset=0
Затем в файле audit.raw.py формируется отчет по полученным данным с процентным соотношением из main.py


Версия датасета chembl 37


