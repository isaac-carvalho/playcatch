# Playcatch — DJ Musical com IA

Sistema inteligente de recomendacao musical que analisa o sentimento das letras e sugere faixas que combinam com o humor do usuario, atraves de um chatbot interativo.

## Arquitetura

```
[Usuario] <-> [Chatbot Gradio] <-> [Recomendador] <-> [Analisador de Sentimentos]
                                        |
                                [Base de letras + sentimentos]
```

### Modulos

| Modulo | Arquivo | Descricao |
|--------|---------|-----------|
| Analisador | `src/sentiment_analyzer.py` | Analise de sentimentos com BERT multilingual (nlptown) |
| Recomendador | `src/recommender.py` | Filtragem por humor + sistema de feedback |
| Chatbot | `src/chatbot.py` | Deteccao de intencao + contexto de sessao |
| Interface | `src/app.py` | Gradio com abas: Conversa, Explorar, Estatisticas |

### Pipeline de IA

1. **Analise de sentimentos** — modelo `nlptown/bert-base-multilingual-uncased-sentiment` classifica letras em 5 categorias (triste, melancolico, neutro, feliz, energetico)
2. **Recomendacao** — filtra musicas por humor desejado, ordena por score de confianca + feedback acumulado
3. **Chatbot** — detecta intencao por keywords + classificacao zero-shot (`facebook/bart-large-mnli`), mantém contexto de sessao
4. **Interface** — Gradio com 3 abas integradas

## Como executar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Executar analise de sentimentos (gera data/lyrics_sentiment.json)
python -m src.sentiment_analyzer

# Iniciar interface Gradio
python -m src.app
```

A interface abre em `http://localhost:7860`.

## Estrutura

```
playcatch/
├── data/
│   ├── lyrics.csv                # 20 letras de musicas brasileiras
│   └── lyrics_sentiment.json     # resultados da analise (gerado)
├── src/
│   ├── __init__.py
│   ├── sentiment_analyzer.py     # Etapa 1
│   ├── recommender.py            # Etapa 2
│   ├── chatbot.py                # Etapa 3
│   └── app.py                    # Etapa 4
├── requirements.txt
└── README.md
```

## Stack

- Python 3.10+
- PyTorch
- HuggingFace Transformers
- Gradio
- Modelos: `nlptown/bert-base-multilingual-uncased-sentiment`, `facebook/bart-large-mnli`

## Decisoes de design

- **BERT multilingual** escolhido por suportar portugues nativamente
- **Mapeamento 5 estrelas -> 5 humores** simplifica a UX sem perder granularidade
- **Keywords antes de modelo** para deteccao de intencao: rapido na maioria dos casos, modelo como fallback
- **Feedback com likes/skips** reordena recomendacoes sem retreinar modelo
- **Lazy loading** do classificador zero-shot para nao impactar tempo de inicializacao
