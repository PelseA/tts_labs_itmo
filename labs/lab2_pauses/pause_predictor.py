"""Pause predictor — skeleton for lab 2.

Run as a script to score the predictor on the prepared data::

    python pause_predictor.py

Precision, recall and F1 are computed for `is_pause_after`, and MAE for `pause_duration`
on true positives only. The last word of every utterance is excluded.
"""
import csv
import numpy as np
import tqdm
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error

PAUSE_PREDICTOR_DATA = '../../data/RUSLAN_pause_metadata_byPelse.csv'

class PausePredictor():
    """Предсказывает, в каких местах предложения возникают паузы и какова их длительность.

    Входные данные представляют собой одно предложение в виде последовательности токенов 
    label_raw — слов вместе с их конечными знаками препинания, расположенных по порядку:
    
    ["Я", "вышел", "из", "дома,", "когда", "стемнело."]
    """

    def _get_grammar_group(self, token: str) -> str | None:
        """Определяет грамматическую группу окончания слова для проверки согласования.
        
        Возвращает строковый идентификатор группы или None, если слово не определительное.
        """
        word = token.lower().strip(".,;:!?-\"")
        
        # 1. МУЖСКОЙ РОД, ЕДИНСТВЕННОЕ ЧИСЛО
        if word.endswith(('ый', 'ой', 'ий')):
            return 'singul_mail_nom_acc'  # Именительный / Винительный (неодуш.)
        if word.endswith(('ого', 'его')):
            return 'singul_mail_gen_acc'  # Родительный / Винительный (одуш.)
        if word.endswith(('ому', 'ему')):
            return 'singul_mail_dat'      # Дательный
        if word.endswith(('ым', 'им')):
            return 'singul_mail_inst'     # Творительный
        if word.endswith(('ом', 'ем')):
            return 'singul_mail_prep'     # Предложный

        # 2. ЖЕНСКИЙ РОД, ЕДИНСТВЕННОЕ ЧИСЛО
        if word.endswith(('ая', 'яя')):
            return 'singul_fem_nom'       # Именительный
        if word.endswith(('ую', 'юю')):
            return 'singul_fem_gen_acc'   # Родительный / Винительный
        if word.endswith(('ой', 'ей')):
            return 'singul_fem_dat_inst_prep' # Дательный / Творительный / Предложный

        # 3. СРЕДНИЙ РОД, ЕДИНСТВЕННОЕ ЧИСЛО
        if word.endswith(('ое', 'ее')):
            return 'singul_neuter_nom'    # Именительный / Винительный
        # Остальные падежи среднего рода совпадают с мужским, 
        # поэтому они автоматически обработались блоком мужского рода выше.

        # 4. МНОЖЕСТВЕННОЕ ЧИСЛО
        if word.endswith(('ые', 'ие')):
            return 'plur_nom'             # Именительный
        if word.endswith(('ых', 'их')):
            return 'plur_gen_acc_prep'    # Родительный / Винительный / Предложный
        if word.endswith(('ым', 'им')):
            # Внимание: 'ым'/'им' есть и в ед.ч. мужского рода (Тв.п.), и во мн.ч. (Дат.п.).
            # Возвращаем общую группу для совпадения с таким же соседним словом.
            return 'singul_mail_inst_or_plur_dat' 
        if word.endswith(('ыми', 'ими')):
            return 'plur_inst'            # Творительный

        return None

    def _is_infinitive(self, token: str) -> bool:
        """Вспомогательный метод для Правила 5: проверка, является ли слово начальной формой глагола."""
        # Добавили многоточие в strip, чтобы корректно отсекать его на конце глаголов
        word = token.lower().strip(".,;:!?-\"…")
        return word.endswith(('ть', 'ти', 'чь'))

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Определяет для каждого токена, следует ли за ним пауза, и её длительность.

        Аргументы:
            tokens: Токены (слова со знаками препинания) одного предложения.

        Возвращает:
            is_pause (int, 1 если за токеном следует пауза) и pause_duration
            (float, секунды, 0.0 если паузы нет), оба массива длины len(tokens).
        """
        is_pause = np.zeros(len(tokens), int)
        pause_duration = np.zeros(len(tokens), float)

        # Множество индексов глаголов, после которых нужно поставить паузу по Правилу 5
        infinitive_pause_indices = set()

        # --- СНАЧАЛА ОПРЕДЕЛЯЕМ ИНДЕКСЫ ДЛЯ ПРАВИЛА 5 ("чтобы" + [не] + [любое слово] + инфинитив) ---
        for idx, token in enumerate(tokens):
            clean_word = token.lower().strip(".,;:!?-\"…")
            
            if clean_word == "чтобы":
                search_idx = idx + 1
                
                # Ограничиваем окно поиска четырьмя словами вперед
                while search_idx < len(tokens) and (search_idx - idx) <= 4:
                    next_token = tokens[search_idx]
                    next_clean = next_token.lower().strip(".,;:!?-\"…")
                    
                    # 1. Если встретили частицу "не" — просто пропускаем её и идем дальше
                    if next_clean == "не":
                        search_idx += 1
                        continue
                        
                    # 2. Если встретили инфинитив — цель достигнута, фиксируем его индекс и выходим из поиска
                    if self._is_infinitive(next_token):
                        infinitive_pause_indices.add(search_idx)
                        break
                        
                    # 3. Если это НЕ инфинитив (любое другое слово: существительное, наречие, местоимение),
                    # мы его пропускаем и продолжаем искать инфинитив дальше
                    search_idx += 1

        # Набор русских гласных букв в нижнем регистре
        vowels = set('аоуыэяеёюи')

        # --- ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ПРАВИЛ ---
        for idx, token in enumerate(tokens):
            clean_word = token.strip(".,;:!?-\"…")
            clean_word_lower = clean_word.lower()
            
            # --- Правило 6: Пауза после слова "говорит" с исключениями ---
            if clean_word_lower == "говорит":
                # По умолчанию предполагаем, что пауза нужна
                is_p_govorit = True
                
                # Заглядываем вперед для проверки исключений ["мне", "нам", "ей", "ему"]
                if idx < len(tokens) - 1:
                    next_token_clean = tokens[idx + 1].lower().strip(".,;:!?-\"…")
                    if next_token_clean in ["мне", "нам", "ей", "ему"]:
                        is_p_govorit = False  # Отменяем паузу для исключений
                
                if is_p_govorit:
                    is_pause[idx] = 1
                    pause_duration[idx] = 0.15  # Интонационная пауза перед цитатой/прямой речью
                    continue
                else:
                    # Если сработал запрет на паузу ("говорит мне"), принудительно переходим к следующему слову
                    continue

            # --- Правило 8: Пауза ПЕРЕД словом "авторов" (после предшествующего слова) ---
            if idx < len(tokens) - 1:
                next_token_clean = tokens[idx + 1].lower().strip(".,;:!?-\"…")
                # Если следующее слово — "авторов", ставим паузу после текущего слова
                if next_token_clean == "авторов":
                    is_pause[idx] = 1
                    pause_duration[idx] = 0.08  # Легкая выделительная пауза перед ключевым словом
                    continue

            # --- Правило 1: Любой знак препинания с исключениями ---
            has_punctuation = any(char in token for char in [',', '.', ':', ';', '!', '?', '…']) or token == '-'
            
            if has_punctuation:
                # ИСКЛЮЧЕНИЕ 1: сложный союз "потому, что" -> ПАУЗУ СТРОГО НЕ СТАВИМ
                if clean_word_lower == "потому" and "," in token and idx < len(tokens) - 1:
                    next_token_clean = tokens[idx + 1].lower().strip(".,;:!?-\"…")
                    if next_token_clean == "что":
                        continue
                
                # ИСКЛЮЧЕНИЕ 2: вводные слова, после которых в речи редко бывает пауза
                # Проверяем, что в токене именно ЗАПЯТАЯ, и слово входит в список
                if "," in token and clean_word_lower in ["очевидно", "наверное", "видимо", "возможно"]:
                    continue  # Пропускаем постановку паузы и переходим к следующему слову
                
                # Если исключения не сработали, ставим стандартную паузу по знаку препинания
                is_pause[idx] = 1
                if any(strong_char in token for strong_char in ['.', '!', '?', ':', ';', '…']):
                    pause_duration[idx] = 0.22
                else:
                    pause_duration[idx] = 0.12
                continue

            # --- Правило 5: Пауза после инфинитива в конструкции с "чтобы" ---
            if idx in infinitive_pause_indices:
                is_pause[idx] = 1
                pause_duration[idx] = 0.10
                continue

            # --- Правило 7: Пауза ПЕРЕД союзом "и" (после слов от n букв без знаков препинания) ---
            if idx < len(tokens) - 1:
                # Слово должно быть длиннее 4 букв
                if len(clean_word_lower) > 5:
                    next_token_clean = tokens[idx + 1].lower().strip(".,;:!?-\"…")
                    # Проверяем, что следующее слово — строго союз "и"
                    if next_token_clean == "и":
                        is_pause[idx] = 1
                        pause_duration[idx] = 0.08
                        continue

            # --- Уточненное Правило 2: Согласованные определительные подряд (пауза после каждого кроме последнего) ---
            if idx < len(tokens) - 1:
                current_group = self._get_grammar_group(token)
                next_group = self._get_grammar_group(tokens[idx + 1])
                
                if current_group is not None and current_group == next_group:
                    is_pause[idx] = 1
                    pause_duration[idx] = 0.08
                    continue

            # --- Правило 3: Слово-повтор после знака препинания ---
            if idx > 0:
                prev_token = tokens[idx - 1]
                prev_has_punct = any(char in prev_token for char in [',', '.', ':', ';', '!', '?', '…']) or prev_token == '-'
                if prev_has_punct:
                    prev_clean = prev_token.strip(".,;:!?-\"…")
                    if clean_word_lower == prev_clean.lower():
                        is_pause[idx] = 1
                        pause_duration[idx] = 0.06
                        continue

            # --- Правило 4: Длинное слово с высокой плотностью согласных ---
            if len(clean_word_lower) >= 12:
                
                # Считаем количество гласных букв в очищенном слове
                vowel_count = sum(1 for char in clean_word_lower if char in vowels)
                
                # Защита от деления на ноль, если слово вдруг пустое
                if len(clean_word_lower) > 0:
                    vowel_ratio = vowel_count / len(clean_word_lower)
                    
                    # Если доля гласных строго менее 33%, то предсказываем паузу
                    if vowel_ratio < 0.33:
                        is_pause[idx] = 1
                        pause_duration[idx] = 0.08
                        continue


        return is_pause, pause_duration

    def predict_durations(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Вставляет предсказанные паузы в последовательность токенов.

        В таком формате данные передаются в акустическую модель в лабораторных работах 4 и 5.

        Аргументы:
            tokens: Токены (слова со знаками препинания) одного предложения.

        Возвращает:
            Кортеж из двух массивов:
            - Массив токенов, где после каждой предсказанной паузы вставлена метка "<SIL>".
            - Массив длительностей (дробные числа) для каждого выходного токена: 
              значение в секундах для пауз "<SIL>" и "-1.0" для обычных слов.
        """
        is_pause, durations = self.predict(tokens)
        tokens_w_pauses = []
        durations_w_pauses = []
        
        for token, is_p, dur in zip(tokens, is_pause, durations):
            if bool(is_p):
                tokens_w_pauses.extend([token, '<SIL>'])
                durations_w_pauses.extend([-1.0, float(dur)])
            else:
                tokens_w_pauses.append(token)
                durations_w_pauses.append(-1.0)
        
        return np.array(tokens_w_pauses), np.array(durations_w_pauses, dtype=np.float32)

def calc_metrics(df: pd.DataFrame) -> None:
    """Print precision, recall and F1 for pause placement, and MAE for pause duration.

    MAE counts only rows where both the reference and the prediction have a pause.
    """
    # zero_division=0 защищает от варнингов sklearn, если предсказаний нет
    rec = recall_score(df.is_pause_after, df.is_pause_hat, zero_division=0)
    prc = precision_score(df.is_pause_after, df.is_pause_hat, zero_division=0)
    f1 = f1_score(df.is_pause_after, df.is_pause_hat, zero_division=0)

    # Защита от падения: MAE считаем ТОЛЬКО если есть хотя бы одно True Positive совпадение
    tp_mask = (df.is_pause_after == 1) & (df.is_pause_hat == 1)
    if tp_mask.sum() > 0:
        mae = mean_absolute_error(df[tp_mask].pause_duration, df[tp_mask].pause_duration_hat)
        mae_str = f"{mae:.4f}"
    else:
        mae_str = "Undefined (No True Positives yet)"

    print(f'PRC: {prc:.4f}, REC: {rec:.4f}, F1: {f1:.4f}; MAE: {mae_str};')

def test_pause_predictor() -> None:
    """Run the predictor on every sentence and print train and test metrics.

    Expects the layout written by `prepare_training_data.py`: rows grouped by utterance
    in order, each utterance ending with its `is_last_word` row.
    """
    pause_df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)
    pp = PausePredictor()

    # Словари для безопасного сбора предсказаний по каждому id предложения
    is_pause_after_hat = {}
    pause_duration_hat = {}

    print("Running predictions...")
    # Безопасный обход через groupby защищает от сдвига индексов idx
    for name, group in tqdm.tqdm(pause_df.groupby('id', sort=False)):
        is_pause_hat, pause_dur_hat = pp.predict(group.label_raw.values)
        is_pause_after_hat[name] = list(is_pause_hat)
        pause_duration_hat[name] = list(pause_dur_hat)
    
    # Переносим предсказания обратно в датафрейм, сохраняя структуру строк
    pause_df['is_pause_hat'] = pause_df.groupby('id', sort=False)['id'].transform(lambda x: is_pause_after_hat[x.name])
    pause_df['pause_duration_hat'] = pause_df.groupby('id', sort=False)['id'].transform(lambda x: pause_duration_hat[x.name])
    
    print('Calculate metrics, training fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df.set=='train') & (pause_df.is_last_word==0)])

    print('\nCalculate metrics, testing fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df.set=='test') & (pause_df.is_last_word==0)])
    
if __name__=='__main__':
    test_pause_predictor()