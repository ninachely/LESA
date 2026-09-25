# Первый запуск LESA в ML Space

Схема соответствует `RF-Solver-Edit/FLUX_Image_Edit/jobs/reinforce_train_smoke_job.yaml`:
CPU Jupyter для подготовки, отдельное окружение на общем диске, `mls job submit`,
образ `cuda12.1-torch2-py311:0.0.36`, одна `a100plus.1gpu.80vG.12C.96G`.
Окружение RF-Solver не изменяется. GPU-задача использует PyTorch 2.6.0 cu124 из
отдельного окружения LESA, а не встроенный PyTorch образа. CUDA проверяется внутри job.

## Что уже есть у авторов

В `LESA_FLUX.1-dev_FLUX.1-schnell/src/train.py` есть подготовка признаков и две стадии
обучения (`GT-Guided Training`, `CL-AR Training`). В `src/sample.py` есть инференс,
но в исходном коммите он неверно вызывал `denoise_cache`; вызов исправлен.
В исходном репозитории нет файлов весов LESA. Для первого запуска обучаем маленький
предиктор сами. Большой FLUX не обучается. Тренировочный скрипт запускаем в один
процесс: в авторском коде нет синхронизации градиентов предиктора между GPU.

## 1. Подготовка на CPU Jupyter

Открыть Terminal в папке LESA на общем NFS того же региона, что и GPU jobs.
Должны быть доступны `conda`, `git`, `mls`, интернет и не менее примерно 120 ГиБ
свободного места для первого запуска (модели, окружение, pip cache, признаки).
Проверьте также квоту NFS: `df` может показывать свободное место всего хранилища.

```bash
git pull --ff-only
pwd
df -h .
command -v conda
command -v mls
bash scripts/mlspace/setup.sh
source scripts/mlspace/env.sh
```

Окружение устанавливается в `.runtime/env`, веса и HF cache — в `.runtime/huggingface`.
GPU на этапе установки не требуется. Используем исходники через PYTHONPATH, поскольку
в поставке авторов метаданные `pip install .` ссылаются на отсутствующие README/LICENSE.

Перед скачиванием открыть https://huggingface.co/black-forest-labs/FLUX.1-dev
и получить доступ аккаунтом Hugging Face. Ввести read token интерактивно:

```bash
"$LESA_STORAGE/env/bin/hf" auth login
"$LESA_STORAGE/env/bin/python" scripts/mlspace/download_models.py
```

При запросе сохранения HF-токена в Git credentials можно ответить `n`.
Токен не передаётся в job YAML. GPU job читает уже скачанные веса в offline-режиме.
Оригинальный T5-файл занимает около 44.5 GB, FLUX — около 24 GB.
Скачивается только один формат весов для каждого компонента.

## 2. Создать YAML и отправить job

Определить путь к **этой же копии LESA внутри GPU job**. В RF-Solver используется
`/workspace/nyuchelysheva/...`; в Jupyter тот же NFS может быть доступен как
`/home/jovyan/nyuchelysheva/...`. Не переносите пути между разными NFS/регионами
простым переименованием. Если LESA лежит в `lesa-work/LESA`, сохраните этот суффикс.

Например, для Jupyter `/home/jovyan/nyuchelysheva/LESA` и соответствующего ему
worker `/workspace/nyuchelysheva/LESA`:

```bash
source scripts/mlspace/env.sh
"$LESA_STORAGE/env/bin/python" scripts/mlspace/make_job.py \
  --worker-repo /workspace/nyuchelysheva/LESA
cat .runtime/flux_smoke_job.yaml
mls job submit -c .runtime/flux_smoke_job.yaml
```

`make_job.py` только пишет YAML, не отправляет задачу. Все пути в YAML проверяем
перед отправкой. При обновлении кода дождаться завершения текущей job.
Статус и stdout/stderr доступны в интерфейсе ML Space. Перезапуск создаёт новый
каталог результатов и начинает обучение заново; это не resume.

## Что выполняет job

1. Проверка CUDA, BF16 и GPU >=70 GiB (расчёт на A100 80GB).
2. Сохранение полной траектории FLUX для 8 обучающих промптов, 1024×1024, 50 шагов.
3. Одна итерация по промптам GT-Guided, затем одна CL-AR с загрузкой предыдущих весов
   и состояния оптимизатора, как предусмотрено авторами. `N=7`, `E=3`, guidance=3.5.
4. Сравнение на двух других промптах DrawBench: full и LESA используют одинаковый
   начальный шум и одинаковую сетку из 50 шагов. Прогрев каждого режима исключён
   из замеров; порядок full/LESA чередуется. CUDA синхронизируется на границах замера.

Измеряется **только денойзинг**, а не полная latency: загрузка модели, текстовые
энкодеры, декодирование, сохранение и обучение в это время не входят. На двух
промптах и одном замере на пару оценка ускорения предварительная.
Изображения PNG без watermark; PSNR считается в float32, SSIM — по RGB.
LPIPS, HPS, ImageReward и доверительные интервалы в этот smoke test не входят.

## Результаты

В конце лога должны быть `LESA_SMOKE=PASS` и путь `RESULTS=.../summary.json`.
PASS означает успешное выполнение и конечные значения тензоров/весов, а не
достижение качества или ускорения из статьи. В `runs/flux-smoke-.../`:

- `comparison_0.png`, `comparison_1.png` — full слева, LESA справа;
- `full/`, `lesa/` — исходные 1024×1024 PNG;
- `summary.json` — парные метрики, времена, фактические полные шаги;
- `config.json`, `requirements.freeze.txt` — параметры, commit, GPU и зависимости;
- `weights/predictor_flux_N7_E3.pt` — итоговый предиктор;
- `gt-guided.pt` — копия предиктора после первой стадии;
- `data/`, `log/` — признаки, шум и авторские логи обучения.

```bash
ls -td runs/flux-smoke-*
```

Код проверен локально на синтаксис и небольших CPU-тестах. Успешный A100-прогон
нужно подтвердить логом вашей job: доступа к ML Space из локальной сессии нет.

Документация формата job: https://cloud.ru/docs/aicloud/mlspace/concepts/client-lib__job
