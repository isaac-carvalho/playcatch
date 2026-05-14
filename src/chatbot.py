"""
Etapa 3 — Chatbot Interativo
Playcatch: plataforma de streaming musical inteligente

Chatbot que conversa com usuarios, detecta humor na mensagem
e recomenda musicas usando o modulo recomendador.
Usa classificacao zero-shot para identificar intencao e sentimento.
"""

import re
from pathlib import Path

from transformers import pipeline

from src.recommender import MusicRecommender, SENTIMENTOS_VALIDOS


# Palavras-chave para deteccao de intencao (fallback rapido)
MOOD_KEYWORDS = {
    "feliz": ["feliz", "alegre", "contente", "bem", "otimo", "maravilhoso", "legal"],
    "energetico": ["animado", "energia", "empolgado", "agitado", "festa", "dançar", "correr", "malhar"],
    "triste": ["triste", "chateado", "deprimido", "mal", "chorar", "sozinho", "dor"],
    "melancolico": ["saudade", "nostalgia", "melancolico", "lembrar", "passado", "distante"],
    "neutro": ["calmo", "tranquilo", "relaxar", "paz", "suave", "leve", "descansar"],
}

# Intencoes do chatbot
INTENCOES = [
    "buscar musica por humor",
    "pedir mais recomendacoes",
    "dar feedback positivo",
    "dar feedback negativo",
    "saudacao",
    "despedida",
    "pergunta geral",
]


class PlaycatchChatbot:
    """Chatbot musical com deteccao de intencao e recomendacao."""

    def __init__(self, recommender: MusicRecommender):
        self.recommender = recommender
        self.classifier = None
        self.last_mood = None
        self.last_recommendations = []
        self.shown_titles = []

    def _load_classifier(self):
        """Carrega classificador zero-shot (lazy)."""
        if self.classifier is None:
            print("Carregando classificador de intencao...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                truncation=True,
            )
        return self.classifier

    def detect_mood_keywords(self, text: str) -> str | None:
        """Detecta humor por palavras-chave (rapido, sem modelo)."""
        text_lower = text.lower()
        for mood, keywords in MOOD_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return mood
        return None

    def detect_mood_model(self, text: str) -> str:
        """Detecta humor usando classificacao zero-shot."""
        clf = self._load_classifier()
        labels = ["feliz e animado", "triste e melancolico", "calmo e tranquilo", "energetico e empolgado", "nostalgico e saudoso"]
        result = clf(text, labels)
        top_label = result["labels"][0]

        mapping = {
            "feliz e animado": "feliz",
            "triste e melancolico": "triste",
            "calmo e tranquilo": "neutro",
            "energetico e empolgado": "energetico",
            "nostalgico e saudoso": "melancolico",
        }
        return mapping.get(top_label, "neutro")

    def detect_intent(self, text: str) -> str:
        """Detecta intencao da mensagem do usuario."""
        text_lower = text.lower().strip()

        # Regras rapidas antes do modelo
        if any(w in text_lower for w in ["oi", "ola", "hey", "eai", "bom dia", "boa tarde", "boa noite"]):
            return "saudacao"
        if any(w in text_lower for w in ["tchau", "bye", "ate", "falou", "valeu"]):
            return "despedida"
        # Negativo ANTES do positivo ("nao gostei" contém "gostei")
        if any(w in text_lower for w in ["nao gostei", "pulei", "pula", "nao curti", "fraca"]):
            return "dar feedback negativo"
        if any(w in text_lower for w in ["gostei", "curti", "adorei", "boa", "show", "top"]):
            return "dar feedback positivo"
        # Word boundary para "mais" evitar falso positivo em "demais"
        if re.search(r'\bmais\b', text_lower) or any(w in text_lower for w in ["outra", "outras", "proxima", "mais dessas"]):
            return "pedir mais recomendacoes"

        # Se tem palavras de humor, e busca
        if self.detect_mood_keywords(text):
            return "buscar musica por humor"

        # Fallback: assume busca por humor
        return "buscar musica por humor"

    def format_recommendations(self, recs: list[dict]) -> str:
        """Formata lista de recomendacoes para exibicao."""
        if not recs:
            return "Nao encontrei musicas para esse humor na minha base. Tente outro!"
        lines = []
        for i, r in enumerate(recs, 1):
            stars = "⭐" * r.get("estrelas", 3)
            lines.append(f"🎵 {i}. **{r['titulo']}** — {r['artista']} {stars}")
        return "\n".join(lines)

    def respond(self, message: str, history: list | None = None) -> str:
        """Processa mensagem do usuario e retorna resposta."""
        if not message or not message.strip():
            return "Oi! Me diz como voce ta se sentindo e eu recomendo musicas."
        intent = self.detect_intent(message)

        if intent == "saudacao":
            moods = self.recommender.get_available_moods()
            return (
                f"Oi! Sou o DJ da Playcatch 🎧\n\n"
                f"Me conta como voce ta se sentindo e eu recomendo musicas pra combinar!\n\n"
                f"Humores disponiveis: {', '.join(moods)}\n\n"
                f"Exemplos: \"to animado\", \"quero algo calmo\", \"to com saudade\""
            )

        if intent == "despedida":
            return "Ate mais! Espero que curta as musicas 🎶"

        if intent == "dar feedback positivo":
            if self.last_recommendations:
                song = self.last_recommendations[0]
                self.recommender.register_feedback(song["titulo"], song["artista"], "like")
                return f"Que bom que curtiu **{song['titulo']}**! Vou lembrar disso nas proximas recomendacoes 👍"
            return "Fico feliz! Me diz um humor e eu recomendo mais musicas."

        if intent == "dar feedback negativo":
            if self.last_recommendations:
                song = self.last_recommendations[0]
                self.recommender.register_feedback(song["titulo"], song["artista"], "skip")
                return f"Entendi, **{song['titulo']}** nao e sua praia. Quer tentar outro humor?"
            return "Sem problema! Me diz o que voce quer ouvir."

        if intent == "pedir mais recomendacoes":
            if self.last_mood:
                recs = self.recommender.recommend(self.last_mood, n=3, exclude=self.shown_titles)
                if not recs:
                    return f"Ja mostrei todas as musicas de humor '{self.last_mood}'. Quer experimentar outro?"
                self.last_recommendations = recs
                self.shown_titles.extend([r["titulo"] for r in recs])
                return f"Mais musicas com humor **{self.last_mood}**:\n\n{self.format_recommendations(recs)}"
            return "Ainda nao sei seu humor! Me conta como voce ta."

        # buscar musica por humor
        mood = self.detect_mood_keywords(message)
        if mood is None:
            mood = self.detect_mood_model(message)

        self.last_mood = mood
        self.shown_titles = []
        recs = self.recommender.recommend(mood, n=3)
        self.last_recommendations = recs
        self.shown_titles = [r["titulo"] for r in recs]

        header = f"Entendi! Voce ta com humor **{mood}**. Olha o que separei pra voce:\n\n"
        return header + self.format_recommendations(recs)


def main():
    """Teste interativo no terminal."""
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "lyrics_sentiment.json"

    rec = MusicRecommender(str(data_path))
    bot = PlaycatchChatbot(rec)

    print("=" * 60)
    print("PLAYCATCH — Chatbot Musical (terminal)")
    print("Digite 'sair' para encerrar")
    print("=" * 60)

    while True:
        msg = input("\nVoce: ").strip()
        if msg.lower() in ("sair", "exit", "quit"):
            print("\nAte mais! 🎶")
            break
        if not msg:
            continue
        response = bot.respond(msg)
        print(f"\nDJ Playcatch: {response}")


if __name__ == "__main__":
    main()
