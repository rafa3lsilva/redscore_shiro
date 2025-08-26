import streamlit as st
import pandas as pd
import numpy as np
import data as dt
import sidebar as sb
import services as sv
import views as vw
import logging

# ----------------------------
# CONFIGURAÇÕES INICIAIS
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if "saved_analyses" not in st.session_state:
    st.session_state.saved_analyses = []
if "dados_jogos" not in st.session_state:
    st.session_state.dados_jogos = None
if "df_jogos" not in st.session_state:
    st.session_state.df_jogos = pd.DataFrame()
if "data_loaded_successfully" not in st.session_state:
    st.session_state.data_loaded_successfully = False

st.set_page_config(
    page_title="Análise Futebol",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inicialização de estados
for key, default in {
    "saved_analyses": [],
    "dados_jogos": None,
    "df_jogos": pd.DataFrame(),
    "data_loaded_successfully": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------
# FUNÇÃO AUXILIAR → Value Bet
# ----------------------------
def mostrar_value_bet(label, prob, odd_justa, col):
    """
    Exibe probabilidade, odd justa e permite inserir odd do mercado para detectar valor.
    """
    with col:
        st.metric(label=label, value=f"{prob}%", delta=f"Odd Justa: {odd_justa}")
        odd_mercado = st.number_input(
            f"Odd Mercado para {label}",
            min_value=1.00,
            value=float(odd_justa),
            step=0.01,
            format="%.2f",
            key=f"odd_mercado_{label}"
        )
        if odd_mercado > odd_justa:
            valor_ev = (odd_mercado / odd_justa - 1) * 100
            st.success(f"✅ Valor Encontrado: +{valor_ev:.2f}%")
        else:
            st.warning("Sem valor aparente.")


def odd_justa_from_pct(pct):
    # evita divisão por zero e odds infinitas
    p = max(min(float(pct), 99.999), 0.001) / 100.0
    return round(1.0 / p, 2)

# ----------------------------
# INTERFACE
# ----------------------------
vw.titulo_principal()
sb.sidebar()

# ----------------------------
# CARREGAMENTO DE DADOS
# ----------------------------
# st.sidebar.markdown("### 🔎 Filtros da Análise")
with st.spinner("⏳ Carregando dados do GitHub..."):
    df_jogos, df_proximos_jogos, dia_br, dia_iso = sv.carregar_dados(dia)

st.session_state.df_proximos_jogos = df_proximos_jogos
st.session_state.df_jogos = sv.carregar_base_historica()
df, df_proximos = st.session_state.df_jogos, st.session_state.df_proximos_jogos

vw.mostrar_status_carregamento(df_proximos_jogos, dia_br, dia_iso)
vw.configurar_estilo_intervalo_jogos()

# ----------------------------
# SELEÇÃO DE JOGO
# ----------------------------
if not df.empty and not df_proximos.empty:
    # Filtros sequenciais (hora → liga → confronto)
    selected_time = st.sidebar.selectbox(
        "Selecione o Horário:", sorted(df_proximos['hora'].unique()))
    jogos_filtrado_hora = df_proximos[df_proximos['hora'] == selected_time]

    selected_league = st.sidebar.selectbox(
        "Selecione a Liga:", sorted(jogos_filtrado_hora['liga'].unique()))
    jogos_filtrado_liga = jogos_filtrado_hora[jogos_filtrado_hora['liga']
                                              == selected_league]

    selected_game = st.sidebar.selectbox(
        "Escolha o Jogo:", sorted(jogos_filtrado_liga['confronto'].unique()))
    selected_game_data = jogos_filtrado_liga[jogos_filtrado_liga['confronto'] == selected_game]

    if selected_game_data.empty:
        st.warning("Por favor, selecione um jogo válido para iniciar a análise.")
        st.stop()

    home_team, away_team = selected_game_data[['home', 'away']].iloc[0]

    # Cenário
    selected_scenario = st.sidebar.selectbox(
        "Cenário de Análise:",
        ["Geral", "Casa/Fora"],
        help="Geral: todos os jogos. Casa/Fora: só casa do mandante e fora do visitante."
    )

    # Define bases de dados de acordo com cenário
    if selected_scenario == 'Geral':
        df_home_base = df[(df['Home'].str.lower() == home_team.lower()) |
                          (df['Away'].str.lower() == home_team.lower())].copy()
        df_away_base = df[(df['Home'].str.lower() == away_team.lower()) |
                          (df['Away'].str.lower() == away_team.lower())].copy()
    else:
        df_home_base = df[df['Home'].str.lower() == home_team.lower()].copy()
        df_away_base = df[df['Away'].str.lower() == away_team.lower()].copy()

    # Ordenar jogos mais recentes
    df_home_base, df_away_base = df_home_base.sort_values(
        by='Data', ascending=False), df_away_base.sort_values(by='Data', ascending=False)

    # ----------------------------
    # INTERVALO DE JOGOS
    # ----------------------------
    st.markdown("### 📅 Intervalo de Jogos")
    intervalo = st.radio("", options=["Últimos 5 jogos", "Últimos 6 jogos",
                         "Últimos 8 jogos", "Últimos 10 jogos"], index=1, horizontal=True)
    num_jogos_selecionado = int(intervalo.split()[1])
    df_home, df_away = df_home_base.head(
        num_jogos_selecionado), df_away_base.head(num_jogos_selecionado)

    # Ajusta o número de jogos se o usuário pedir mais do que o disponível
    num_jogos_home = min(num_jogos_selecionado, len(df_home_base))
    num_jogos_away = min(num_jogos_selecionado, len(df_away_base))

    # Pega os N primeiros jogos (os mais recentes, pois já ordenámos no início) para a análise final
    df_home = df_home_base.head(num_jogos_home)
    df_away = df_away_base.head(num_jogos_away)
    st.markdown("---")

    # ----------------------------
    # AJUSTE DE PESOS
    # ----------------------------
    with st.sidebar.expander("⚙️ Ajustar Pesos do Modelo"):
        limite_consistente = st.slider(
            "Nível 'Consistente' (DP ≤)", 0.1, 2.0, 0.8, 0.1)
        limite_imprevisivel = st.slider(
            "Nível 'Imprevisível' (DP >)", 0.1, 2.0, 1.2, 0.1)

    # ----------------------------
    # ANÁLISE PRINCIPAL DO CENÁRIO
    # ----------------------------
    analise = dt.analisar_cenario_partida(
        home_team, away_team, df_jogos,
        num_jogos=num_jogos_selecionado,
        scenario=selected_scenario,
        linha_gols=2.5
    )
    if "erro" in analise:
        # Se sim, exibe o aviso e para a execução
        st.warning(f"⚠️ {analise['erro']}")
        st.stop()

    # Resultado 1X2
    st.markdown(f"#### 📊 Cenário da Partida ({analise['cenario_usado']})")
    col1, col2, col3 = st.columns(3)
    col1.metric("🏠 Vitória " + home_team, f"{analise['prob_home']}%")
    col2.metric("🤝 Empate", f"{analise['prob_draw']}%")
    col3.metric("✈️ Vitória " + away_team, f"{analise['prob_away']}%")

    # Over/Under + BTTS
    col1, col2 = st.columns(2)
    col1.markdown(f"🔼 Over {analise['over_under']['linha']} gols: **{analise['over_under']['p_over']}%**")
    col2.markdown(f"🔽 Under {analise['over_under']['linha']} gols: **{analise['over_under']['p_under']}%**")
    col1, col2 = st.columns(2)
    col1.markdown(f"✅ BTTS Sim: **{analise['btts']['p_btts_sim']}%**")
    col2.markdown(f"❌ BTTS Não: **{analise['btts']['p_btts_nao']}%**")

    # ----------------------------
    # CARD DE VENCEDOR
    # ----------------------------

    # Calcula previsões
    resultados = dt.prever_gols(home_team, away_team, df_jogos,
                                num_jogos=num_jogos_selecionado,
                                min_jogos=3,
                                scenario=selected_scenario)
    
    # converte para %
    prob_home = round(resultados["p_home"] * 100, 2)
    prob_draw = round(resultados["p_draw"] * 100, 2)
    prob_away = round(resultados["p_away"] * 100, 2)

    # odds justas (sem margem de bookmaker)
    odd_home = round(1 / max(resultados["p_home"], 1e-6), 2)
    odd_draw = round(1 / max(resultados["p_draw"], 1e-6), 2)
    odd_away = round(1 / max(resultados["p_away"], 1e-6), 2)

    # define provável vencedor
    if prob_home > prob_away and prob_home > prob_draw:
        vencedor = home_team
    elif prob_away > prob_home and prob_away > prob_draw:
        vencedor = away_team
    else:
        vencedor = "Empate"

    vw.card_vencedor(vencedor_nome=vencedor,
                     home_team_nome=home_team, away_team_nome=away_team)

    #-------------------------
    # Mercados Complementares
    #-------------------------
    # Exibir Top 5 Placares, Gol no 1º Tempo(Poisson + histórico), Mercado de Gols(comparador EV + gráfico)
    # Escanteios(NegBin + médias históricas)
    st.markdown("### Análise para mercados complementares")
    # Placares prováveis
    st.markdown("### 🔮 Top 5 Placares Mais Prováveis")
    # Cria as colunas para os 5 placares
    cols = st.columns(5)

    # Itera sobre os resultados e chama a função para cada um
    if analise.get("placares_top"):
        cols = st.columns(min(5, len(analise['placares_top'])))
        for idx, p in enumerate(analise['placares_top']):
            with cols[idx]:
                vw.card_placar(placar=p['placar'], prob=p['prob'])
    else:
        st.warning("⚠️ Sem dados suficientes para estimar placares prováveis.")

    # Analise de gol HT
    st.markdown("## 🕐 Gol no 1º Tempo")
    # 🎯 Modelo probabilístico (Poisson)
    ht = dt.prever_gol_ht(
        home_team, away_team, df_jogos,
        num_jogos=num_jogos_selecionado,
        scenario=selected_scenario
    )
    st.markdown(f"### Probabilidades com base no Modelo Poisson")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- Over 0.5 HT: **{ht['p_gol_ht']}%**")
    with col2:
        st.markdown(f"- Exatamente 1 gol no HT: **{ht['p_exato1_ht']}%**")

    # Histórico de apoio
    analise_ht_hist = dt.analise_gol_ht(df_home, df_away)
    with st.expander("📋 Estatística de apoio para gols no 1º tempo, usando médias históricas com base nos últimos jogos"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Média Over 0.5 HT",
                      f"{analise_ht_hist['media_05ht']:.1f}%")
        with col2:
            st.metric("Média Over 1.5 FT",
                      f"{analise_ht_hist['media_15ft']:.1f}%")
        with col3:
            st.metric("Média Over 2.5 FT",
                      f"{analise_ht_hist['media_25ft']:.1f}%")

        # 1. Junta os dataframes de casa e fora para uma análise combinada
        df_total_ht = pd.concat([df_home, df_away], ignore_index=True)

        # 2. Chama a nova função que criámos em data.py
        desvio_padrao_ht = dt.analisar_consistencia_gols_ht(df_total_ht)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Desvio Padrão dos Gols HT", f"{desvio_padrao_ht:.2f}")

        with col2:
            if desvio_padrao_ht == 0.0:
                interpretacao = "ℹ️ Dados insuficientes."
            elif desvio_padrao_ht <= limite_consistente:
                interpretacao = "✅ **Cenário Consistente:** A quantidade de gols no HT nos jogos destas equipas tende a ser muito previsível."
            elif desvio_padrao_ht <= limite_imprevisivel:
                interpretacao = "⚠️ **Cenário Moderado:** Há alguma variação na quantidade de gols no HT, mas ainda com alguma previsibilidade."
            else:
                interpretacao = "🚨 **Cenário Imprevisível:** A quantidade de gols no HT varia muito de jogo para jogo. É um cenário de 'altos e baixos'."

            st.info(interpretacao)

        # Análise de Gol no Primeiro Tempo (HT)
        analise_ht = dt.analisar_gol_ht_home_away(df_home, df_away)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Análise de {home_team}:**")
            st.info(
                f"Marcou gol no HT em **{analise_ht['home_marca']:.1f}%** dos seus jogos.")
            st.warning(
                f"Sofreu gol no HT em **{analise_ht['home_sofre']:.1f}%** dos seus jogos.")

        with col2:
            st.markdown(f"**Análise de {away_team}:**")
            st.info(
                f"Marcou gol no HT em **{analise_ht['away_marca']:.1f}%** dos seus jogos.")
            st.warning(
                f"Sofreu gol no HT em **{analise_ht['away_sofre']:.1f}%** dos seus jogos.")

    # --- MERCADO DE GOLS ---
    st.markdown("## 🎯 Mercado de Gols (FT)")

    # Probabilidades principais (Poisson)
    st.sidebar.markdown(
        "<h3 style='text-align: center;'>🎯 Linhas de Gols Over/Under</h3>", unsafe_allow_html=True)
    linha_gols = st.sidebar.selectbox(
        "Linha de Gols - Over/Under:",
        [1.5, 2.5, 3.5],
        index=1
    )
    over_under = dt.calcular_over_under(resultados, linha=linha_gols)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- 🔼 Over {linha_gols}: **{over_under['p_over']}%**")
    with col2:
        st.markdown(f"- 🔽 Under {linha_gols}: **{over_under['p_under']}%**")

    # --- Probabilidades por Mercado (Poisson) ---
    linha_over15 = dt.calcular_over_under(resultados, linha=1.5)
    linha_over25 = dt.calcular_over_under(resultados, linha=2.5)
    linha_over35 = dt.calcular_over_under(resultados, linha=3.5)
    btts = dt.calcular_btts(resultados) if resultados else {
        "p_btts_sim": 0, "p_btts_nao": 0}


    # Monta DataFrame direto do modelo Poisson
    df_resultado_mercados = pd.DataFrame([
        {"Mercado": "Over 1.5", "Probabilidade (%)": linha_over15['p_over'], "Odd Justa": round(
            100/linha_over15['p_over'], 2)},
        {"Mercado": "Over 2.5", "Probabilidade (%)": linha_over25['p_over'], "Odd Justa": round(
            100/linha_over25['p_over'], 2)},
        {"Mercado": "Over 3.5", "Probabilidade (%)": linha_over35['p_over'], "Odd Justa": round(
            100/linha_over35['p_over'], 2)},
        {"Mercado": "BTTS", "Probabilidade (%)": btts['p_btts_sim'], "Odd Justa": round(
            100/btts['p_btts_sim'], 2)},
    ])

    st.subheader(
        "Probabilidades por Mercado com Poisson e Comparador de Valor")

    cols = st.columns(len(df_resultado_mercados))
    for i, col in enumerate(cols):
        with col:
            mercado = df_resultado_mercados.iloc[i]
            odd_justa_safe = mercado['Odd Justa'] if np.isfinite(
                mercado['Odd Justa']) else 1.0
            st.metric(
                label=mercado["Mercado"],
                value=f'{mercado["Probabilidade (%)"]}%',
                delta=f'Odd Justa: {mercado["Odd Justa"]}'
            )

            odd_mercado = st.number_input(
                f"Odd Mercado para {mercado['Mercado']}",
                min_value=1.00,
                value=odd_justa_safe,
                step=0.01,
                format="%.2f",
                key=f"odd_mercado_{mercado['Mercado']}"
            )

            if odd_mercado > mercado['Odd Justa']:
                valor_ev = (odd_mercado / mercado['Odd Justa'] - 1) * 100
                st.success(f"✅ Valor Encontrado: +{valor_ev:.2f}%")
            else:
                st.warning("Sem valor aparente.")
 
    # Gráfico de barras para as probabilidades por mercado
    with st.expander("📊 Gráfico de Probabilidades por Mercado"):
        vw.grafico_mercados(df_resultado_mercados,
                            titulo="Probabilidades (Poisson + BTTS)")

    # Linhas Over/Under de Escanteios na sidebar
    # A nova linha, com o texto centralizado
    st.sidebar.markdown(
        "<h3 style='text-align: center;'>📊 Linhas de Escanteios Over/Under</h3>", unsafe_allow_html=True)
    linha_escanteios = st.sidebar.selectbox(
        "Selecione a linha de escanteios:",
        [6.5, 7.5, 8.5, 9.5, 10.5, 11.5],
        index=3
    )
    st.session_state.linha_escanteios = linha_escanteios

    # Calcula probabilidades de escanteios
    cantos = dt.prever_escanteios_nb(home_team, away_team, df_jogos,
                                     num_jogos=num_jogos_selecionado, scenario=selected_scenario)

    # Probabilidades Over/Under
    st.session_state.over_under_cantos = dt.calcular_over_under_cantos(
        cantos, st.session_state.linha_escanteios)

    # Quem tem mais cantos
    mais_cantos = dt.prob_home_mais_cantos(cantos)

    # --- ESCANTEIOS ---
    st.markdown("## 🟦 Estimativa de Escanteios")

    # 🎯 Modelo principal (NegBin)
    cantos = dt.prever_escanteios_nb(
        home_team, away_team, df_jogos,
        num_jogos=num_jogos_selecionado,
        scenario=selected_scenario
    )
    st.session_state.over_under_cantos = dt.calcular_over_under_cantos(
        cantos, st.session_state.linha_escanteios
    )
    mais_cantos = dt.prob_home_mais_cantos(cantos)

    st.markdown("### Probabilidades (Modelo NegBin)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"- Over {st.session_state.linha_escanteios}: **{st.session_state.over_under_cantos['p_over']}%**")
    with col2:
        st.markdown(
            f"- Under {st.session_state.linha_escanteios}: **{st.session_state.over_under_cantos['p_under']}%**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"- 🏠 Home mais cantos: **{mais_cantos['home_mais']}%**")
    with col2:
        st.markdown(f"- 🤝 Empate em cantos: **{mais_cantos['empate']}%**")
    with col3:
        st.markdown(f"- ✈️ Away mais cantos: **{mais_cantos['away_mais']}%**")

    # 📊 Apoio: médias históricas
    resultado_escanteios = dt.estimar_linha_escanteios(
        df_home, df_away, home_team, away_team)
    with st.expander("📋 Estatísticas Históricas de Escanteios"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Média Cantos Mandante",
                      f"{resultado_escanteios['Escanteios Mandante']:.2f}")
        with col2:
            st.metric("Média Cantos Visitante",
                      f"{resultado_escanteios['Escanteios Visitante']:.2f}")
        with col3:
            st.metric("Média Total de Cantos",
                      f"{resultado_escanteios['Escanteios Totais Ajustados']:.2f}")
            
    # ----------------------------------------------------
    # Apoio Estatístico com médias históricas estatísticas
    # ----------------------------------------------------
    st.markdown("### Apoio Estatístico")

    # Exibe as médias de gols
    media_home_gols_marcados = dt.media_gols_marcados(df_home, home_team)
    media_home_gols_sofridos = dt.media_gols_sofridos(df_home, home_team)
    media_away_gols_marcados = dt.media_gols_marcados(df_away, away_team)
    media_away_gols_sofridos = dt.media_gols_sofridos(df_away, away_team)

    # Exibe as médias de gols
    st.markdown("### 📋 Médias de Gols Home e Away", unsafe_allow_html=True)
    vw.mostrar_cards_media_gols(
        home_team,
        away_team,
        media_home_gols_marcados,
        media_home_gols_sofridos,
        media_away_gols_marcados,
        media_away_gols_sofridos
    )
 
    # Tabela de Jogos home e away
    with st.expander("📋 Ver Últimos Jogos Analisados"):
        vw.mostrar_tabela_jogos(df_home, home_team, "🏠")
        vw.mostrar_tabela_jogos(df_away, away_team, "✈️")

    # Botão para salvar análise atual
if st.sidebar.button("💾 Salvar Análise Atual"):
    # 1. Extrai os dados dos mercados e escanteios
    prob_over_1_5 = df_resultado_mercados.loc[
        df_resultado_mercados['Mercado'] == 'Over 1.5', 'Probabilidade (%)'
    ].iloc[0]
    prob_over_2_5 = df_resultado_mercados.loc[
        df_resultado_mercados['Mercado'] == 'Over 2.5', 'Probabilidade (%)'
    ].iloc[0]
    prob_btts = df_resultado_mercados.loc[
        df_resultado_mercados['Mercado'] == 'BTTS', 'Probabilidade (%)'
    ].iloc[0]

    df_escanteios = pd.DataFrame(
        resultado_escanteios['Probabilidades por Mercado'])
    linha_mais_provavel = df_escanteios.loc[df_escanteios['Probabilidade (%)'].idxmax(
    )]
    linha_escanteio_str = f"{linha_mais_provavel['Mercado']} ({linha_mais_provavel['Probabilidade (%)']:.1f}%)"

    # 2. Monta dicionário da análise
    current_analysis = {
        "Liga": selected_league,
        "Home": home_team,
        "Away": away_team,
        "Cenário": selected_scenario,
        "Jogos Analisados": f"{len(df_home)} vs {len(df_away)}",
        "Prob. Casa (%)": prob_home,
        "Prob. Empate (%)": prob_draw,
        "Prob. Visitante (%)": prob_away,
        "Prob. Gol HT (%)": round(ht['p_gol_ht'], 2),
        "Prob. Over 1.5 (%)": prob_over_1_5,
        "Prob. Over 2.5 (%)": prob_over_2_5,
        "Prob. BTTS (%)": prob_btts,
        "Linha Escanteios": linha_escanteio_str,
    }

    # 3. Salva no relatório
    st.session_state.saved_analyses.append(current_analysis)
    st.toast(f"✅ Análise de '{home_team} vs {away_team}' salva no relatório!")

# --- Relatório de análises salvas ---
if st.session_state.saved_analyses:
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Relatório de Análises")

    num_saved = len(st.session_state.saved_analyses)
    st.sidebar.info(f"Você tem **{num_saved}** análise(s) salva(s).")

    df_report = pd.DataFrame(st.session_state.saved_analyses)

    # Função para converter para Excel
    @st.cache_data
    def to_excel(df):
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Analises')
        return output.getvalue()

    excel_data = to_excel(df_report)

    # Botão de download
    st.sidebar.download_button(
        label="📥 Baixar Relatório (Excel)",
        data=excel_data,
        file_name="relatorio_analises.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Botão para limpar
    if st.sidebar.button("🗑️ Limpar Análises Salvas"):
        st.session_state.saved_analyses = []
        st.rerun()

    # Expansor para ver as análises
    with st.sidebar.expander("Ver análises salvas"):
        st.dataframe(df_report)
