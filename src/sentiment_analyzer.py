"""
Etapa 1 — Analise de Sentimentos das Letras
Playcatch: plataforma de streaming musical inteligente

Carrega letras de musicas, aplica modelo de analise de sentimentos
(nlptown/bert-base-multilingual-uncased-sentiment) e salva resultados
em JSON estruturado para uso pelo recomendador.
"""

import csv
import json
import logging
import re
from pathlib import Path

from transformers import pipeline

log = logging.getLogger(__name__)


# Mapeamento de estrelas (1-5) para categorias de humor
STAR_TO_MOOD = {
    1: "triste",
    2: "melancolico",
    3: "neutro",
    4: "feliz",
    5: "energetico",
}


def load_lyrics(csv_path: str) -> list[dict]:
    """Carrega letras do CSV e retorna lista de dicts."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Arquivo de letras nao encontrado: {csv_path}")
    lyrics = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lyrics.append({
                "titulo": row["titulo"].strip(),
                "artista": row["artista"].strip(),
                "letra": row["letra"].strip(),
            })
    return lyrics


def clean_text(text: str) -> str:
    """Normaliza texto: lowercase, remove pontuacao excessiva, whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúâêîôûãõçà]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_sentiments(lyrics: list[dict], model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment") -> list[dict]:
    """
    Aplica analise de sentimentos a cada letra.
    O modelo retorna estrelas de 1 a 5, mapeadas para categorias de humor.
    Trunca letras longas para respeitar limite de 512 tokens do BERT.
    """
    log.info("Carregando modelo: %s", model_name)
    classifier = pipeline("sentiment-analysis", model=model_name, truncation=True, max_length=512)
    log.info("Modelo carregado com sucesso!")

    results = []
    for item in lyrics:
        cleaned = clean_text(item["letra"])

        prediction = classifier(cleaned[:512])[0]
        # Label vem como "1 star", "2 stars", etc.
        stars = int(prediction["label"].split()[0])
        score = round(prediction["score"], 4)
        mood = STAR_TO_MOOD.get(stars, "neutro")

        result = {
            "titulo": item["titulo"],
            "artista": item["artista"],
            "letra": item["letra"],
            "estrelas": stars,
            "sentimento": mood,
            "score": score,
        }
        results.append(result)
        log.info("  %s | %d estrelas | %s | score: %s", item['titulo'], stars, mood, score)

    return results


def save_results(results: list[dict], output_path: str) -> None:
    """Salva resultados em JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("Resultados salvos em: %s", output_path)


def main():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "data" / "lyrics.csv"
    output_path = base_dir / "data" / "lyrics_sentiment.json"

    print("=" * 60)
    print("PLAYCATCH — Analise de Sentimentos das Letras")
    print("=" * 60)

    lyrics = load_lyrics(str(csv_path))
    print(f"\n{len(lyrics)} letras carregadas.\n")

    results = analyze_sentiments(lyrics)

    save_results(results, str(output_path))

    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    from collections import Counter
    mood_counts = Counter(r["sentimento"] for r in results)
    for mood, count in mood_counts.most_common():
        print(f"  {mood:15s}: {count} musicas")


if __name__ == "__main__":
    main()
