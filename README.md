# legal-rag-starter-kit

Reusable RAG core for source-grounded vertical AI assistants.  
Supports hosted OpenAI-compatible APIs, local Ollama runtimes, source-aware answers, insufficient-basis mode, retrieval quality controls, and reusable evaluation workflows.

---

## О проекте

`legal-rag-starter-kit` — это переиспользуемая инфраструктурная основа для построения вертикальных AI-ассистентов с Retrieval-Augmented Generation.

Проект позиционируется как **starter kit / technical core**, а не как готовый конечный бот под один узкий сценарий.  
Ключевая идея: отделить стабильное domain-agnostic ядро (retrieval, prompting, cache, runtime orchestration) от domain-specific слоев (корпус, продуктовая логика, UX, шаблоны ответов, экспорт).

---

## Позиционирование

Проект предназначен для команд и индивидуальных разработчиков, которым нужна надежная база для запуска вертикальных ассистентов с source-grounded ответами.

Это foundation-уровень, на котором строятся специализированные продукты:

- ассистенты по юридическим документам;
- ассистенты по комплаенсу;
- ассистенты по финансам;
- ассистенты по персональным базам знаний;
- другие вертикальные RAG-приложения с собственными корпусами и требованиями.

---

## Ключевые возможности ядра

- **Provider-agnostic LLM client**
  - Нейтральные переменные `LLM_API_KEY` и `LLM_BASE_URL` с backward-compatible fallback.
- **Hosted + Local режимы**
  - Поддержка OpenAI-compatible hosted APIs.
  - Поддержка локальных рантаймов через Ollama.
- **Source-grounded generation**
  - Ответы формируются на основе retrieved context с ссылочностью на источники.
- **Insufficient-basis mode**
  - Явная фиксация недостаточности данных вместо уверенных неподтвержденных выводов.
  - Если ни один фрагмент не прошёл отбор, LLM не вызывается: возвращается детерминированный ответ о недостаточности данных.
- **Language-aware prompting**
  - Контроль языка и формата ответа в зависимости от языка запроса.
- **Retrieval quality pass**
  - Отбор контекста одной функцией: проверка distance + cutoff + dedup по содержимому, без добивки слабыми результатами.
- **Кеширование**
  - Кеш запросов/ответов и версия корпуса для управляемой инвалидации.
- **Переиспользуемая оценка качества**
  - Подходит для системной оценки retrieval/generation в разных вертикалях.
- **Экспортно-ориентированная архитектура**
  - Ядро отделено от слоев экспорта (включая PDF-пайплайны).
- **Web-first подход**
  - Core не зависит от UI; web-слой остается легкой интеграцией поверх ядра.

---

## Retrieval Quality (runtime)

В рантайме используется один quality pass — функция `select_context` (`app_core/retrieval/selection.py`). Её же вызывают пайплайн и диагностические скрипты. Порядок результатов поиска сохраняется:

1. **Raw retrieval.** Получение расширенного набора кандидатов (`RAG_RAW_TOP_K`, по умолчанию `10`).

2. **Проверка distance и cutoff.** Остаются только фрагменты с корректным (числовым, конечным) расстоянием не выше `RAG_MAX_DISTANCE` (по умолчанию `0.44`). Отсутствующий, `None`, NaN и нечисловой distance не проходят.

3. **Dedup по содержимому.** Удаляются только точные дубли текста (после нормализации Unicode NFC и пробелов); остаётся фрагмент с лучшим рангом. Разные фрагменты одного источника или раздела не схлопываются. Пустые фрагменты отбрасываются.

4. **Final context clamp.** Не более `RAG_FINAL_TOP_K` (по умолчанию `5`) фрагментов. Слабые результаты не добавляются, чтобы добить это число: подходящих фрагментов может быть меньше или не быть вовсе.

Если подходящих фрагментов нет, LLM не вызывается, а пользователь получает детерминированный ответ о недостаточности данных в подключенных источниках (без ссылок на источники; такой ответ не кешируется).

Фрагменты передаются в prompt внутри тегов `<retrieved_fragment>`, а system prompt и инструкции требуют считать их недоверенными справочными данными и не выполнять содержащиеся в них указания. Если фрагменты противоречат друг другу, модель должна указать на противоречие, а не выбирать один из них молча.

---

## Структура проекта

```text
app_core/                 # переиспользуемое ядро
  config/                 # конфигурация core-уровня
  cache/                  # кеш и хранилище кеша
  evaluation/             # reusable evaluation-логика
  generation/             # prompt builder и generation helpers
  llm/                    # provider-agnostic клиентский слой
  retrieval/              # векторное хранилище и retrieval-логика
  schemas/                # схемы и структуры данных

web/                      # web-интерфейс
  templates/              # html-шаблоны
  static/                 # статические ресурсы

examples/                 # примеры запуска и конфигураций
sample_corpus/            # крошечный синтетический корпус для проверки запуска
raw_sources/              # исходные документы
knowledge_base/           # подготовленные/нормализованные материалы
scripts/                  # служебные и диагностические скрипты
runtime/                  # локальные runtime-артефакты (chroma, cache)
exporters/                # слой экспорта (например, PDF)

tests/                    # тесты
```

---

## Примеры vertical-проектов поверх starter kit

- **Cross-Border Contract Risk Assistant**
- **Trade Finance Legal Reference Assistant**

Каждый vertical-проект может иметь собственные:

- corpus/config слой;
- prompt profile;
- eval datasets;
- product UX;
- экспортные шаблоны и отчеты.

При этом runtime-ядро остается общим и переиспользуемым.

---

## Локальный запуск

Каноническая версия Python — **3.11** (проверено на 3.11.9). Другие версии не тестировались и не заявляются как поддерживаемые.

### 1) Создание и активация виртуального окружения

**Windows (PowerShell):**

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2) Установка зависимостей

Зависимости разделены по файлам; `constraints-py311.txt` фиксирует протестированные версии для Python 3.11.

```bash
# Рантайм + тесты (рекомендуется для разработки):
pip install -r requirements-dev.txt -c constraints-py311.txt

# Только рантайм (CLI + web):
pip install -r requirements.txt -c constraints-py311.txt

# Опционально: стек RAGAS для evaluate_ragas.py (поверх рантайма):
pip install -r requirements-eval.txt -c constraints-py311.txt
```

### 3) Настройка окружения

Выберите профиль и скопируйте значения в файл `.env` **в корне репозитория**:

- `.env.local.example` — локальный Ollama-режим;
- `.env.hosted.example` — hosted OpenAI-compatible режим (по умолчанию `https://api.openai.com/v1`);
- `.env.example` — объединенный reference-шаблон.

`.env` читается только из корня репозитория (родительские каталоги не просматриваются). Переменные, уже заданные в окружении ОС, имеют приоритет над значениями из `.env`. `.env` и `.env.*` игнорируются git, кроме шаблонов `*.example`.

### 4) Корпус

Корпус задаётся явно — JSON-манифестом, путь к которому указан в `RAG_CORPUS_CONFIG` (путь от корня репозитория):

```env
RAG_CORPUS_CONFIG=sample_corpus/corpus.json
```

`sample_corpus/` — крошечный синтетический пример (вымышленная компания), чтобы проверить запуск без приватных данных; он уже прописан в `.env.*.example`. Для собственных документов создайте манифест того же формата (см. `sample_corpus/README.md`) и укажите его в `RAG_CORPUS_CONFIG`. Приватные материалы кладите в `raw_sources/`, `knowledge_base/` или `data/` — их содержимое игнорируется git. Если корпус не настроен, а индекс пуст, приложение сообщает об этом явной ошибкой.

Индексация корпуса при первом запуске вызывает embeddings-API выбранного провайдера (hosted или локальный).

### 5) Запуск приложения

Консольное приложение:

```bash
python app.py
```

Web-интерфейс (http://127.0.0.1:8010):

```bash
uvicorn web.app:app --port 8010
```

### Тесты

```bash
pytest -q
```

Тесты работают офлайн: ключи API и сеть не нужны, провайдеры подменены заглушками.

### Оценка качества (опционально)

`evaluate_ragas.py` — отдельный опциональный скрипт: требует `requirements-eval.txt`, вызывает платный OpenAI API (`OPENAI_API_KEY`), а его вопросы и эталоны написаны под корпус `examples/independent_guarantees/corpus.json` (локальные файлы `data/*.txt` в репозитории отсутствуют). В основной рантайм и CI он не входит.

---

## Режимы работы

### Local Ollama mode

Пример конфигурации:

```env
LLM_API_KEY=local-placeholder
LLM_BASE_URL=http://localhost:11434/v1
# Fast local mode:
RAG_CHAT_MODEL=gemma3:4b

# Quality local mode:
# RAG_CHAT_MODEL=deepseek-r1:8b
RAG_EMBEDDING_MODEL=bge-m3
RAG_CHROMA_PATH=runtime/chroma_db_local_bge_m3
RAG_CORPUS_VERSION=local-bge-m3-v1
```

Рекомендации:

- разделяйте локальные и hosted векторные базы;
- при смене embedding model/chunking обновляйте индекс и `RAG_CORPUS_VERSION`.

### Hosted OpenAI-compatible mode

Пример конфигурации:

```env
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
RAG_CHAT_MODEL=gpt-4o-mini
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_CHROMA_PATH=runtime/chroma_db_hosted_text_embedding_3_small
RAG_CORPUS_VERSION=hosted-text-embedding-3-small-v1
```

---

## Основные runtime-параметры

Retrieval quality controls:

```env
RAG_RAW_TOP_K=10
RAG_MAX_DISTANCE=0.44
RAG_FINAL_TOP_K=5

# Legacy fallback:
# RAG_TOP_K=5
```

Дополнительно:

- `RAG_MAX_TOKENS`
- `RAG_TEMPERATURE`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_MIN_CHUNK_LEN`
- `RAG_EMBED_BATCH_SIZE`

---

## Масштабирование и дальнейшее развитие

При развитии starter kit для production-verticals логично добавить:

- reranking поверх базового vector retrieval;
- расширенные evaluation datasets и регрессионные quality-gates;
- более точный source scoring и confidence сигналы;
- улучшения web UI для прозрачности retrieval/source traceability;
- генерацию PDF-чеклистов и структурированных отчетов;
- multi-corpus конфигурации;
- domain packs (корпус + prompts + eval profile) для быстрого запуска новых вертикалей.

---

## Принцип архитектуры

Сохраняйте ядро нейтральным и переиспользуемым.  
Предметную специализацию выносите в отдельные vertical-слои.

Именно это делает проект масштабируемым как инфраструктурный RAG starter kit.

---

## Статус проекта

Проект находится в активной разработке.  
Текущий фокус: стабилизация reusable core, quality controls и подготовка vertical demos поверх ядра.
