import streamlit as st
import pandas as pd
import altair as alt

# ----------------------------
# TÍTULO PRINCIPAL
# ----------------------------
def titulo_principal():
    st.markdown("""
    <div style="text-align:center; margin-bottom:20px;">
        <h1>⚽ Análise Futebol - RedScore</h1>
        <p style="font-size:18px; color:gray;">Probabilidades, Over/Under, BTTS e Escanteios</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# ESTILO INTERVALO DE JOGOS
# ----------------------------
def configurar_estilo_intervalo_jogos():
    st.markdown("""
    <style>
    div[role='radiogroup'] > label {
        background-color: #262730;
        color: white;
        margin-top: 5px;
        border-radius: 12px;
        padding: 4px 12px;
        margin-right: 8px;
        cursor: pointer;
        border: 1px solid transparent;
        transition: all 0.2s ease-in-out;
    }
    div[role='radiogroup'] > label:hover {
        background-color: #ff4b4b;
    }
    div[role='radiogroup'] > label[data-selected="true"] {
        background-color: #ff4b4b;
        border-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# CARD DE PLACAR PROVÁVEL
# ----------------------------
def card_placar(placar: str, prob: float):
    """
    Cria um card estilizado para exibir um placar e a sua probabilidade.
    """
    st.markdown(f"""
    <div style="background-color:#1f2937; padding:15px; border-radius:8px; text-align:center; color:white; height: 100%;margin-bottom:10px;">
        <h3 style="margin:0; font-size: 24px;">{placar}</h3>
        <p style="font-size:18px; margin:0; color: #9CA3AF;">{prob:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# CARD DE VENCEDOR
# ----------------------------
def card_vencedor(vencedor_nome: str, home_team_nome: str, away_team_nome: str):
    st.markdown("### 🏆 Vencedor da Partida")
    # Lógica para definir a cor com base no vencedor
    if vencedor_nome == home_team_nome:
        cor = "#4CAF50"  # Verde para o time da casa
    elif vencedor_nome == away_team_nome:
        cor = "#F44336"  # Vermelho para o time visitante
    else:
        cor = "#607D8B"  # Cinza para o empate

    st.markdown(
        f"""
        <div style='background-color:{cor};padding:10px;border-radius:8px'>
            <h3 style='color:white;text-align:center'>{vencedor_nome}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

def mostrar_cards_media_gols(
    home_team: str,
    away_team: str,
    media_home_gols_marcados: float,
    media_home_gols_sofridos: float,
    media_away_gols_marcados: float,
    media_away_gols_sofridos: float
):
    """Mostra os cards estilizados com médias de gols."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="background-color:#1f77b4; padding:15px; border-radius:8px; 
                    text-align:center; color:white; margin-bottom:10px;">
            <h3>🏠 {home_team}</h3>
            <p style="font-size:18px;">⚽ Média de Gols Marcados: 
                <strong>{media_home_gols_marcados:.2f}</strong></p>
            <p style="font-size:18px;">🛡️ Média de Gols Sofridos: 
                <strong>{media_home_gols_sofridos:.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color:#d62728; padding:15px; border-radius:8px; 
                    text-align:center; color:white; margin-bottom:10px;">
            <h3>✈️ {away_team}</h3>
            <p style="font-size:18px;">⚽ Média de Gols Marcados: 
                <strong>{media_away_gols_marcados:.2f}</strong></p>
            <p style="font-size:18px;">🛡️ Média de Gols Sofridos: 
                <strong>{media_away_gols_sofridos:.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# GRÁFICO DE MERCADOS
# ----------------------------
def grafico_mercados(df: pd.DataFrame, titulo: str = "Probabilidades por Mercado"):
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Mercado', sort=None),
        y=alt.Y('Probabilidade (%)', title='Probabilidade (%)'),
        color='Mercado',
        tooltip=['Mercado', 'Probabilidade (%)', 'Odd Justa']
    )
    st.subheader(titulo)
    st.altair_chart(chart, use_container_width=True)

# ----------------------------
# TABELA DE JOGOS HOME E AWAY
# ----------------------------
def mostrar_tabela_jogos(df: pd.DataFrame, team: str, tipo: str):
    """Mostra a tabela de últimos jogos de um time (Home/Away)."""
    def auto_height(df, base=35, header=40, max_height=500):
        return min(len(df) * base + header, max_height)

    cols_to_show = [c for c in df.columns if c not in ["Pais", "resultado"]]

    st.markdown(f"### {tipo} Últimos {len(df)} jogos do **{team}**")
    st.dataframe(
        df[cols_to_show].reset_index(drop=True),
        use_container_width=True,
        height=auto_height(df),
        hide_index=True
    )

