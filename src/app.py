"""
Etapa 4 — Integracao e Interface Unificada
Playcatch: plataforma de streaming musical inteligente

Interface Gradio com duas abas:
1. Conversa: chatbot interativo para descoberta musical
2. Explorar: busca por humor com cards e botoes de feedback
"""

import json
from pathlib import Path

import gradio as gr

from src.recommender import MusicRecommender, SENTIMENTOS_VALIDOS
from src.chatbot import PlaycatchChatbot


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "lyrics_sentiment.json"

# Inicializa componentes
recommender = MusicRecommender(str(DATA_PATH))
chatbot = PlaycatchChatbot(recommender)


# ===================== ABA 1: CONVERSA =====================

def chat_respond(message: str, history: list[dict]) -> tuple[list[dict], str]:
    """Processa mensagem no chatbot e retorna historico atualizado."""
    if not message.strip():
        return history, ""

    response = chatbot.respond(message, history)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return history, ""


# ===================== ABA 2: EXPLORAR =====================

def explore_by_mood(mood: str) -> str:
    """Retorna recomendacoes formatadas para o humor selecionado."""
    recs = recommender.recommend(mood, n=5)
    if not recs:
        return f"Nenhuma musica encontrada para humor: {mood}"

    lines = [f"## Musicas com humor: {mood.upper()}\n"]
    for i, r in enumerate(recs, 1):
        stars = "⭐" * r.get("estrelas", 3)
        lines.append(f"### {i}. {r['titulo']} — {r['artista']}")
        lines.append(f"- Sentimento: **{r['sentimento']}** | Score: {r['score']} | {stars}")
        lines.append(f"- Trecho: _{r['letra'][:100]}..._\n")

    return "\n".join(lines)


def give_feedback(titulo: str, action: str) -> str:
    """Registra feedback do usuario."""
    song = recommender.get_song_by_title(titulo)
    if not song:
        return f"Musica '{titulo}' nao encontrada."
    success = recommender.register_feedback(song["titulo"], song["artista"], action)
    if success:
        emoji = "👍" if action == "like" else "👎"
        return f"{emoji} Feedback registrado para **{song['titulo']}**!"
    return "Erro ao registrar feedback."


def get_song_titles() -> list[str]:
    """Retorna lista de titulos para o dropdown."""
    return [s["titulo"] for s in recommender.songs]


def get_stats() -> str:
    """Retorna estatisticas da base."""
    from collections import Counter
    moods = Counter(s["sentimento"] for s in recommender.songs)
    lines = ["## Estatisticas da Base\n"]
    lines.append(f"**Total de musicas:** {len(recommender.songs)}\n")
    for mood, count in moods.most_common():
        bar = "█" * count
        lines.append(f"- **{mood}**: {count} musicas {bar}")
    return "\n".join(lines)


# ===================== INTERFACE =====================

def build_interface() -> gr.Blocks:
    """Constroi interface Gradio unificada."""

    with gr.Blocks(
        title="Playcatch — DJ Musical com IA",
    ) as app:

        gr.Markdown(
            "# 🎧 Playcatch — DJ Musical com IA\n"
            "Descubra musicas que combinam com seu humor usando inteligencia artificial."
        )

        with gr.Tab("💬 Conversa"):
            gr.Markdown("Converse com o DJ Playcatch! Diga como voce ta se sentindo.")

            chatbot_ui = gr.Chatbot(
                value=[{"role": "assistant", "content": "Oi! Sou o DJ da Playcatch 🎧\nMe conta como voce ta se sentindo e eu recomendo musicas!"}],
                height=400,
            )
            msg_input = gr.Textbox(
                placeholder="Ex: to animado, quero algo calmo, to com saudade...",
                label="Sua mensagem",
                scale=4,
            )
            send_btn = gr.Button("Enviar", variant="primary")

            send_btn.click(chat_respond, [msg_input, chatbot_ui], [chatbot_ui, msg_input])
            msg_input.submit(chat_respond, [msg_input, chatbot_ui], [chatbot_ui, msg_input])

        with gr.Tab("🔍 Explorar por Humor"):
            gr.Markdown("Escolha um humor e veja as recomendacoes!")

            with gr.Row():
                mood_dropdown = gr.Dropdown(
                    choices=recommender.get_available_moods(),
                    label="Selecione o humor",
                    value=recommender.get_available_moods()[0] if recommender.get_available_moods() else None,
                )
                explore_btn = gr.Button("Buscar", variant="primary")

            explore_output = gr.Markdown()
            explore_btn.click(explore_by_mood, [mood_dropdown], [explore_output])

            gr.Markdown("---\n### Feedback")
            with gr.Row():
                title_dropdown = gr.Dropdown(
                    choices=get_song_titles(),
                    label="Selecione a musica",
                )
                like_btn = gr.Button("👍 Gostei", variant="secondary")
                skip_btn = gr.Button("👎 Pular", variant="secondary")

            feedback_output = gr.Markdown()
            like_btn.click(lambda t: give_feedback(t, "like"), [title_dropdown], [feedback_output])
            skip_btn.click(lambda t: give_feedback(t, "skip"), [title_dropdown], [feedback_output])

        with gr.Tab("📊 Estatisticas"):
            stats_output = gr.Markdown(value=get_stats())
            refresh_btn = gr.Button("Atualizar")
            refresh_btn.click(get_stats, [], [stats_output])

    return app


def main():
    app = build_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
