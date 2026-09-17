"""Normalized / non-normalized classifier — skeleton for lab 1.

Run as a script to score yourself on the development set::

    python text_filter.py
"""

import csv
import re
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DEV_SET_PATH = "data/dev_sentences.csv"
#DEV_SET_PATH = "data/only_normalized.csv"

# с помощью этого класса можно определить, является ли строка пригодной для передачи в TTS.
class TextFilter:
    """Decides whether an utterance is usable as a training example.

    Example:
        >>> textfilter = TextFilter()
        >>> textfilter.filter("Я вышел из дома.")
        1
        >>> textfilter.filter("Александрову Г. П.")
        0
    """

    # Часть числительного прописью
    DIGIT_ROOTS = [
        "ноль", "ноля", "нуль", "нуля", "нулем", "нулём",
        "один", "одно", "одна", "одни", "одну",
        "два", "две", "двух", "двум", "двое",
        "три", "трой", "трое", "троих", "троим", "трем", "трём", "трех", "трёх",
        "четыр",
        "пять", "пяти",
        "шесть", "шести",
        "семь", "семи",
        "восем", "восьм",
        "девят", "десят",
        "надцат", "дцат", "сорок", "девянос", "сто", "ста",
        "тысяч", "миллион", "миллиард", "триллион", "квадриллион"
    ]

    # Меры - сокращения
    MEASUREMENTS_LIST = [
        # Длина / Расстояние
        "мм", "см", "дм", "м", "км",
        "mm", "cm", "dm", "m", "km",
        # Масса
        "мг", "г", "кг", "т", "ц",
        # Время
        "с", "сек", "мин", "ч", "д", "дн", "сут", "нед", "дек", "мес", "г", "деслет",
        "s", "min", "h", "d", "WEE", "DAD", "MON", "a", "DEC",
        # Объём 
        "мм^3", "мм3", "куб. мм", "куб мм", "MMQ",
        "см^3", "см3", "куб. см", "куб см", "CMQ",
        "дм^3", "дм3", "куб. дм", "куб дм", "DMQ",
        "м^3", "м3", "куб. м", "куб м", "MTQ",
        "мл", "ml", "MLT", "л", "I", "L", "LTR",
        "дл", "dl", "DLT", "гл", "hl", "HLT", "Мл", "Ml", "MAL",
        "дюйм^3", "дюйм3", "куб. дюймов", "куб. дюйма", "куб дюймов", "куб дюйма", "in3", "INQ",
        "фут^3", "фут3", "куб. футов", "куб. фута", "куб футов", "куб фута", "ft3", "FTQ",
        "ярд^3", "ярд3", "куб. ярдов", "куб. ярда", "куб ярдов", "куб ярда", "yd3", "YDQ",
        # Площадь
        "мм2", "мм^2", "mm2", "mm^2", "ММК",
        "см2","см^2","cm2","cm^2","СМК",
        "дм2","дм^2","dm2","dm^2","DMК",
        "м2","м^2","m2","m^2","МТК",
        "10^3 м^2", "daa", "ТЫС М2", "ТЫС М^2", 
        "га", "ha", "HAR",
        "км2", "км^2", "km2", "km^2", "КМК", 
        "дюйм2", "дюйм^2", "in2", "in^2", "INK", 
        "фут2", "фут^2", "ft2", "ft^2", "FTK",
        "ярд2", "ярд^2", "yd2", "yd^2", "YDК",
        "Ар", "а", "a", "ARE",
        # Скорость
        "Бк", "Bq", "BQL", "вБ", "Wb", "WEB", 
        "зуз", "kn", "УЗ", "KNT",
        "м/с",  "м в сек.", "m/s", "м/сек", "MTS",
        "об/с", "об в с", "r/s", "RPS", "об в сек.", 
        "об/мин", "об в мин", "r/min", "RPM",
        "км/ч", "км в ч",  "km/h", "KMH",  
        "м/с2", "m/s2", "MSK",
        # Электричество и технич.
        "Вт", "w", "WTT", "кВт", "kW", "KWT", 
        "МВт", "MW", "МЕГАВТ", "MAW",
        "В", "V", "VLT", "кВ", "kV", "KVT", 
        "кВ·А", "kV·A", "KVA", "МВ·А", "MV·A", "МЕГАВ·А", "MVA",
        "квар", "kVAR", "KVR", "Вт·ч", "W·h", "ВЧ·Ч", "WHR",
        "кВт·ч", "kW·h", "KWH", "МВт·ч", "MW·h", "МЕГАВТ·Ч", "MWH",
        "ГВт·ч", "GW·h", "ГИГАВТ·Ч", "GWH", 
        "А", "A", "AMP", "А·ч", "A·h", "АМН", "ТАН",
        "Кл", "С", "соu", "Дж", "J", "JOU",
        "кДж", "kJ", "KJO",
        "Ом", "Ω", "OHM", "Гр", "Gy",
        "мкГо", "μGy", "МКГР", "MKGY",
        "мГр", "mGy", "МЛГР", "MGY",
        "кГр", "kGy", "КИЛОГР", "KGY",
        "°С", "°C", "ГРАД ЦЕЛЬС", "CEL",
        "°F", "°F", "ГРАД ФАРЕНГ", "FAN",
        "кд", "cd", "CDL", "лк",  "lx", "LUX",
        "лм", "lm", "LUM", "К",  "К", "KEL",
        "Н", "N", "NEW", "Гц",  "Hz", "HTZ",
        "кГц", "kHz", "МГц", "MHz",  "МЕГАГЦ",
        "ГГц", "GHz", "ГИГАГЦ", "GHZ",
        "Па", "Pa", "PAL", "ТГц",  "THz", "ТЕРАГЦ"
    ]

    HOMOGRAPHS = [
      "атлас",
      "белки",
      "виски",
      "дорогой",
      "замок",
      "ирис",
      "мука",
      "орган",
      "парить",
      "плачу",
      "полки",
      "пропасть",
      "сорок",
      "стрелки",
      "хлопок"
    ]

    def __init__(self):
        """Prepare the classifier's resources.

        Compiled regular expressions, abbreviation and contraction dictionaries, a
        trained model — anything that should not be rebuilt for every utterance.
        """
        # 0. Регулярное выражение для поиска латиницы
        self._latinica_pattern = re.compile(r'[A-Za-z]')
        
        # 1. Регулярное выражение для поиска сокращений с точками (инициалы, т.д., одна буква с точкой)
        # Ищет <перенос/пробел + заглавную/строчную букву + точка>
        self._abbreviations_pattern = re.compile(r'(^|\s)[A-Za-zА-Яа-яЁё]\.')

        # 2. Регулярное выражение для поиска капса / аббревиатур (кроме одиночных предлогов/союзов вроде Я, А, И)
        # Ищет слова из 2 и более заглавных букв подряд.
        self._acronyms_pattern = re.compile(r'\b[A-ЯЁA-Z]{2,}\b')

        # 3. Регулярное выражение для поиска цифр
        self._digits_pattern = re.compile(r'\d')

        # 4. Регулярное выражение для спецсимволов, которые могут читаться словами (%, $, №, +, =)
        self._special_chars_pattern = re.compile(r'[%$№+=&@\)\(#\\/\^\*“”]')
        
        # 5. Регулярное выражение для: <цифра + пробелы + сокращение меры>
        # Например: "5 кг", "10  м"
        measures_alternatives = '|'.join([re.escape(m) for m in self.MEASUREMENTS_LIST])
        self._digits_with_measures_pattern = re.compile(rf'\d+\s*({measures_alternatives})\b', re.IGNORECASE)

        # 6. Регулярка для поиска 3 и более подряд идущих гласных, отделенных границами от остального текста (ОАО "Рубеж")
        self._several_vowels_pattern = re.compile(r'(^|\s|[\(\'"«])[аеёиоуыэюя]{3,}(\s+|[\.!?»\'"\)]|$)', re.IGNORECASE)

        # 7. Регулярка для поиска 3 и более подряд идущих СОгласных, отделенных границами от остального текста (ОАО "Рубеж")
        self._several_consonants_pattern = re.compile(r'(^|\s|[\(\'"«])[бвгджзйклмнпрстфхцчшщ]{3,}(\s+|[\.!?»\'"\)]|$)', re.IGNORECASE)

        # 8. Регулярка для поиска <словао + пробел + знак препинания> - знак препинания должен прилипать к слову
        self._space_before_punct = re.compile(r'[а-яёА-ЯЁ]+\s+[\.!?,;:]')

        # 9. Регулярка для <два и более подряд ОДИНАКОВЫХ знака пунктуации>
        # ВНИМАНИЕ троеточие можно, две точки - нет, разные знаки можно, но не все - СМОТРЕТЬ ПРИМЕРЫ
        self._double_punct = re.compile(r'(^|[а-яёА-ЯЁ]|\s)(\?\?+|!!+|\.\.|\.\.\.\.+|\?\.|!\.)($|\s|[а-яёА-ЯЁ])')

        # 10. Шаблон для поиска пробела нулевой ширины
        self._zero_space_pattern = re.compile(r'[\u200b\u200c\u200d\ufeff\u200e\u200f]')

        # 11. Шаблон для поиска сокращений для сущностей адреса - в населенном пункте
        self._address_entity_pattern = re.compile(r'\b(ул|пер|пр|просп|б-р|бульв|ал|алл|наб|пл|пл-дь|ш|туп|пр-д|пер-к|скв|кв-л|кв|к|корп|стр|вл|дор|р-н|авт|тер|ст|лит|л|мкрн)\.', re.IGNORECASE)

        # 12. Шаблон для поиска сокращений для сущностей адреса - нас пункты и субъекты
        self._city_pattern = re.compile(
            r'\b(п|пгт|пос|д|дер|г|гор|обл|респ|р|кр|окр|ст-(ца[миеу]?|цы|цей|цею|ниц)|с|сп|рп)\.', 
            re.IGNORECASE
        )

        # 13. После знака препинания перед текстом нет пробела
        self._no_space_after_punct = re.compile(r'[\.!?,;:][а-яёА-ЯЁ]')

        # 14. Для междометий
        self._interjection_pattern = re.compile(
            r'(\s|\.|^)((ха|хе|хи|хо|гы)(-?(ха|хе|хи|хо|гы))+\+?|мда|хм+|гм+|ы+|м+)\b', 
            re.IGNORECASE
        )

        # 15. Для социальных статусов
        self._soc_status_pattern = re.compile(
            '(\s|\.|^)(г-н(у|е|а|ом)?|г-ж(а|е|и|у|ой)|г-да|гг|ув|глубокоув|мн\.ув|гр|гражд|тов|тт|проф|доц|акад|докт|д-р|канд|вр|св|еп)(\s|\.|$)', 
            re.IGNORECASE
        )

        # n. Регулярка для римских цифр
        #self._roman_numerals_pattern = re.compile(r'\s([IVXLCDM]|[IVXLCDM][IVXLCDM]{1,8})(\s+|\.)')

        # 16. Для омографов
        self._homograph_pattern = re.compile(
            rf"(\s|\.|^)({'|'.join(self.HOMOGRAPHS)})(\s|\.|$)", 
            re.IGNORECASE
        )

    def _has_homograph(self, text: str) -> bool:
        """Проверяет, содержит ли входящая строка слово-омограф из списка.
        Возвращает True, если омограф найден как отдельное слово, иначе False.
        """
        if self._homograph_pattern.search(text):
            return True
        return False

    def _has_measurements_issue(self, text: str) -> bool:
        """Внутренний метод для проверки правила №5"""
        text_lower = text.lower()
        
        # Шаг А: Проверяем паттерн "цифра + мера" (например, "5 кг")
        if self._digits_with_measures_pattern.search(text_lower):
            return True
            
        # Шаг Б: Проверяем "числительное прописью + сокращение меры"
        # Перебираем все возможные сокращения мер
        for measure in self.MEASUREMENTS_LIST:
            measure_lower = measure.lower()
            
            # Ищем меру в тексте. Важно убедиться, что это отдельное слово-мера (используем \b в поиске)
            # Шаблон ищет: <какой-то текст><мера как отдельное слово>
            for match in re.finditer(rf'\b{re.escape(measure_lower)}\b', text_lower):
                # Берем фрагмент текста СЛЕВА от найденной меры (например, последние 12 символов)
                start_index = max(0, match.start() - 12)
                left_context = text_lower[start_index:match.start()].strip()
                
                # Проверяем, содержится ли какой-то из корней числительных в этом левом контексте
                for root in self.DIGIT_ROOTS:
                    if root in left_context:
                        # Находим индекс последнего вхождения корня в левом контексте
                        root_idx_in_context = left_context.rfind(root)
                    
                        # Вычисляем, где корень ЗАКАНЧИВАЕТСЯ относительно левого контекста
                        root_end_in_context = root_idx_in_context + len(root)
                    
                        # Извлекаем фрагмент, который находится СТРОГО между корнем и началом меры
                        between_fragment = left_context[root_end_in_context:]
                    
                        # Проверяем, что между ними один пробел или два
                        if between_fragment == " " or between_fragment == "  ":
                            return True
                        
        return False

    def _has_quotes_issue(self, text: str) -> bool:
        """Проверяет баланс открывающих и закрывающих кавычек-ёлочек.
        Возвращает True, если есть ошибка (нарушен баланс), 
        и False, если ошибки нет (все кавычки парные).
        """
        stack = []
        
        for char in text:
            if char == '«':
                # Запоминаем открывающую кавычку в стек
                stack.append('«')
            elif char == '»':
                # Если пришла закрывающая, а открывающей не было — это ОШИБКА
                if not stack:
                    return True
                # Если всё ок, закрываем пару
                stack.pop()
                
        # Если строка закончилась, но какая-то кавычка осталась незакрытой — это ОШИБКА
        if stack:
            return True
            
        # Ошибок не обнаружено
        return False

    def _has_interjection(self, text: str) -> bool:
        """Внутренний метод для поиска междометий типа ха-ха, гы, мда и т.д."""
        # Используем .search(), так как объект регулярного выражения не является callable
        if self._interjection_pattern.search(text):
            return True
        return False

    def filter(self, text: str) -> int:
        """
        Определяет, пригодна ли строка для передачи в TTS.
        Classify a single utterance.
        Args:
            text: Utterance text, already passed through :class:`TextNormalizer`.

        Returns:
            ``1`` if the text is normalized and the utterance can be used for
            training; 
            ``0`` if it contains something the speaker pronounced
            differently from how it is written, and the utterance should be dropped.
        """
        if not text or not text.strip():
            return 0

        if self._abbreviations_pattern.search(text):
            return 0

        if self._latinica_pattern.search(text):
            return 0

        if self._acronyms_pattern.search(text):
            return 0

        if self._digits_pattern.search(text):
            return 0

        if self._special_chars_pattern.search(text):
            return 0

        if self._several_vowels_pattern.search(text):
            return 0

        if self._soc_status_pattern.search(text):
            return 0

        if self._several_consonants_pattern.search(text):
            return 0

        # Проверка составного правила (цифра/числительное прописью + мера)
        if self._has_measurements_issue(text):
            return 0

        if self._has_homograph(text):
            return 0

        if self._has_interjection(text):
            return 0
        
        # Поиск смайликов
        if self._has_smiley(text):
            return 0

        if self._zero_space_pattern.search(text):
            return 0

        if self._city_pattern.search(text):
            return 0

        if self._address_entity_pattern.search(text):
            return 0

        if self._no_space_after_punct.search(text):
            return 0

        if self._space_before_punct.search(text):
            return 0

        if self._double_punct.search(text):
            return 0

        if self._has_quotes_issue(text):
            return 0

        return 1

    def _has_smiley(self, text: str) -> bool:
        # Список смайлов
        smiles = [
            # латиница, кириллица
            r'[8BВ]-\)+',
            r'[OoOо]:-\)+',        
            r'>\:-\)',        
        
            r':-@',
            r'\>:-\(',

            # одинокие скобки не берем, т.к. могут быть частью пунктуации
            r'[:;=][-~]?[\(d]+',
            r'\(\(+',
            r':,\(+',
            r'[ТT].[ТT]',
        
            # латиница, кириллица, ноль
            r'[:=]-?[ОO0]',

            # одинокие скобки не берем, т.к. могут быть частью пунктуации
            r'[:=][-~]?[\)D]+',
            r'\)\)+',
            r'[XxХх][DДд]+',
            r':[3З]+',
        
            r';-?[\)D]+',
        
            r':-\/',
            r':-\|',
            r':-\\',
        
            r':-[PpЪ]',
        
            r':[-~]?\*',
            r':-{}'

            r'\>.\<',
        
            r'(\>\<|&gt;&lt;)',

            r'(\^_\^)'
        ]
        combined_pattern = '|'.join(smiles)
        
        # Ищем хотя бы одно совпадение в тексте
        if re.search(combined_pattern, text):
            return True
        return False

if __name__ == "__main__":
    textfilter = TextFilter()

    dev_files = pd.read_csv(
        DEV_SET_PATH, sep="|", encoding="utf-8", quoting=csv.QUOTE_NONE, header=0
    )

    dev_files["predicted"] = dev_files["text"].apply(textfilter.filter)

    # --- НАЧАЛО БЛОКА ДЛЯ ПОИСКА ОШИБОК ---
    # Создаем датафрейм, где реальный класс не равен предсказанному
    errors = dev_files[dev_files["is_normalized"] != dev_files["predicted"]]
    
    print(f"\nВсего ошибок предсказания: {len(errors)}")
    print("Первые 10 ошибочных строк:")
    # Выводим индекс, исходный текст, правильный ответ и то, что предсказала модель
    print(errors[["text", "is_normalized", "predicted"]].head(10))
    
    # Сохраняем все ошибки в файл, чтобы удобно открыть в Excel/Notepad++
    errors.to_csv("mismatched_predictions.csv", sep="|", encoding="utf-8", index=False)
    # --- КОНЕЦ БЛОКА ДЛЯ ПОИСКА ОШИБОК ---
    
    prc = precision_score(dev_files["is_normalized"], dev_files["predicted"])
    rec = recall_score(dev_files["is_normalized"], dev_files["predicted"])
    f1 = f1_score(dev_files["is_normalized"], dev_files["predicted"])
    print(f"F1 Score is {f1:.6f}, Precision is {prc:.6f}, Recall is {rec:.6f}")
