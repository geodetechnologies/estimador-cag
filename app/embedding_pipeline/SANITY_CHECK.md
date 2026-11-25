# Sanity check de embeddings

Ejecución real del 2026-11-25 con `text-embedding-3-small` (1536 dimensiones), lanzando
`python scripts/compare.py --text-a "..." --text-b "..."` desde el host.

| Pareja | Texto 1 | Texto 2 | Expectativa | Similitud coseno |
|---|---|---|---|---|
| A — cercanos | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Authorization service using JSON Web Tokens for a banking application | > 0.6 | **0.5957** |
| B — no relacionados | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Database migration from MySQL to PostgreSQL with zero downtime | < 0.4 | **0.1920** |
| C — genéricos | Backend services | API development | sin expectativa fija | **0.5407** |

## Comentario

El modelo discrimina bien entre textos cercanos y lejanos: entre A y B hay unos 0.40 puntos de
diferencia. Aun así, A queda justo por debajo del umbral orientativo de 0.6. Las dos frases comparten
el concepto (autenticación con JWT en banca o fintech) pero casi ningún término literal
(OAuth/fintech/mobile frente a JSON Web Tokens/banking). Además, `text-embedding-3-small` tiende a
comprimir las similitudes en un rango más bajo que otros modelos. Lo más llamativo es C: dos frases de
dos palabras, vagas y sin relación clara, puntúan 0.54, casi tanto como A. Los textos cortos y
genéricos caen en una zona densa del espacio («software de backend») y esta es la razón de no hacer
chunks demasiado pequeños ni pobres en contexto. Conclusión: el umbral absoluto depende del modelo; es
mejor comparar rankings relativos que fijar un corte universal.

## Ingesta del sample

Hice una llamada real a `POST /embeddings/ingest` con `data/budgets_sample.json`. Respondió 200 en
~0.9 s. Devolvió 15 presupuestos y 52 chunks (uno por componente), todos de 1536 dimensiones y con
`chunk_id` único. Se procesaron 4126 tokens, con un coste estimado de 0.0000825 USD a 0.02 USD por
millón de tokens.
