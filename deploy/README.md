# Запуск PT Network на одном сервере

Целевая конфигурация Ubuntu, Nginx, rootless Podman и PostgreSQL описана в [PRODUCTION.md](PRODUCTION.md). Ниже приведён существующий сценарий Docker Compose для локальной разработки и совместимости.

Для локальной проверки с Docker Compose:

```bash
docker compose up -d --build --wait
docker compose exec site python -m app.operator --data-dir /data --add-user analyst
```

Сайт доступен на `http://127.0.0.1:8845/`, вход в лабораторию — на `/lab/`.
Для другого локального порта задайте одновременно `PT_PORT` и `PT_PUBLIC_ORIGIN`.

## Публичный запуск

Укажите домен с DNS-записью на сервер и разрешите входящий TCP на 80 и 443.
Caddy выпустит и обновит сертификат. FastAPI доступен Caddy через внутреннюю сеть
Compose. В production наружу публикуются только порты Caddy.

```bash
export PT_DOMAIN=pt.example.com
docker compose -f compose.yaml -f compose.production.yaml up -d --build --wait
docker compose -f compose.yaml -f compose.production.yaml exec site \
  python -m app.operator --data-dir /data --add-user analyst
```

Пароль вводится в терминале. Данные пользователей и SQLite находятся в томе
`workspace`, а данные TLS — в `tls`. Перед обновлением сохраните резервную копию тома
`workspace` при остановленном `site`. Храните архив вне сервера с ограниченным доступом.

```bash
umask 077
docker compose -f compose.yaml -f compose.production.yaml stop site
docker compose -f compose.yaml -f compose.production.yaml run --rm --no-deps -T \
  --entrypoint tar site -czf - -C /data . > "workspace-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
docker compose -f compose.yaml -f compose.production.yaml start site
```

Для отката соберите предыдущую проверенную ревизию с тем же именем Compose-проекта.
Если нужно восстановить резервную копию, остановите приложение и подключите новый
пустой том как `workspace`. Разверните архив в него, сохранив числовые UID и GID,
и проверьте данные перед переключением. Архив содержит данные на момент копирования.

## Инференс multiday_codebook_plus

Для подключения модели подготовьте доверенный образ рантайма с кодом
предобработки, сборкой векторов, реализацией дополнительной головы и
зафиксированными зависимостями. Рантайм должен прочитать полный набор весов из
`PT_MODEL_DIR`, проверить контрольные суммы перед загрузкой и открыть
`POST /v1/predict` на `0.0.0.0:8001`. Запускайте одну копию модели в контейнере.
В образе задайте HEALTHCHECK, который проверяет готовность после загрузки весов.
Контракт запроса, ответа и контрольный пример берутся из пакета модели.
Включение слоя:

```bash
export PT_MODEL_DIR=/srv/pt/models/multiday_codebook_plus
export PT_INFERENCE_IMAGE='registry.example.com/pt/inference:plus-v1'
docker compose -f compose.yaml -f compose.production.yaml -f compose.inference.yaml \
  up -d --build --wait
```

`POST /api/inference` доступен из сессии CorgCode на том же домене. Он передаёт
JSON во внутренний рантайм и возвращает его JSON-ответ. Сайт ограничивает размер
запроса и ответа до 8 МиБ. Весам доступно только чтение, порт модели остаётся
в закрытой сети. Показ результатов в интерфейсе зависит от схемы ответа модели.

В репозитории есть описание и результаты Plus. По [реестру артефактов](../docs/ARTIFACTS.md)
подтверждён только checkpoint основы. Для запуска Plus ещё нужны полный пакет и
рабочий рантайм. Перед выбором сервера замерьте время холодного старта, пиковую
RAM и задержку на типичных окнах при одном запросе и под ожидаемой нагрузкой.
