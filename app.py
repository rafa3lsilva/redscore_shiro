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

st.set_page_config(
    page_title="Análise Futebol",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "saved_analyses" not in st.session_state:
    st.session_state.saved_analyses = []

if "ultimo_confronto_notificado" not in st.session_state:
    st.session_state.ultimo_confronto_notificado = None

# ----------------------------
# INTERFACE
# ----------------------------
vw.titulo_principal()
sb.sidebar()
vw.configurar_estilo_intervalo_jogos()

# ----------------------------
# CARREGAMENTO DE DADOS E PROCESSAMENTO
# ----------------------------
texto_colado = sb.entrada_de_dados_principal()

if texto_colado:

    with st.spinner("🕵️ Identificando times e processando dados..."):
        # CORREÇÃO 1: Receber os 4 itens que a função retorna.
        # As variáveis df_home_base e df_away_base já contêm os jogos corretos e separados.
        home_team, away_team, df_home_base, df_away_base = sv.processar_dados_e_identificar_times(
            texto_colado)

    # A análise só começa DEPOIS que confirmamos que tudo foi extraído corretamente
    if home_team and away_team and not df_home_base.empty and not df_away_base.empty:

        # CORREÇÃO 2: A exibição do cabeçalho e da notificação vem AQUI DENTRO.
        confronto_atual = f"{home_team}-vs-{away_team}"
        if st.session_state.get("ultimo_confronto_notificado") != confronto_atual:
            st.toast(
                f"Análise carregada: 🏠 {home_team} vs ✈️ {away_team}", icon="📊")
            st.session_state.ultimo_confronto_notificado = confronto_atual

        # Exibe o cabeçalho do confronto
        col1, col_vs, col2 = st.columns([5, 1, 5])
        with col1:
            st.markdown(f"""
            <div style="background-color: #1f77b4; border-radius: 10px; padding: 25px; text-align: center; color: white; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
                <h3 style="margin: 0;">🏠 {home_team}</h3>
            </div>
            """, unsafe_allow_html=True)
        with col_vs:
            st.markdown(f"""
            <div style="text-align: center; padding-top: 30px;">
                <p style="font-size: 28px; font-weight: bold; color: #888;">VS</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background-color: #d62728; border-radius: 10px; padding: 25px; text-align: center; color: white; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
                <h3 style="margin: 0;">{away_team} ✈️</h3>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Junta os dois DataFrames para as análises que precisam da base completa
        df_jogos = pd.concat([df_home_base, df_away_base],
                             ignore_index=True).drop_duplicates()

        # ----------------------------
        # DEFINIÇÃO DOS PARÂMETROS DE ANÁLISE (agora na ordem correta)
        # ----------------------------
        st.sidebar.markdown("### ⚙️ Parâmetros da Análise")
        selected_scenario = st.sidebar.selectbox(
            "Cenário de Análise:",
            ["Geral", "Casa/Fora"],
            help="Geral: todos os jogos. Casa/Fora: só jogos em casa do mandante e fora do visitante."
        )

        st.markdown("### 📅 Intervalo de Jogos")
        intervalo = st.radio("", options=["Últimos 5 jogos", "Últimos 6 jogos",
                                          "Últimos 8 jogos", "Últimos 10 jogos"], index=1, horizontal=True)
        num_jogos_selecionado = int(intervalo.split()[1])

        # CORREÇÃO 3: Lógica de filtragem simplificada que usa os DFs corretos
        temp_df_home = df_home_base.copy()
        temp_df_away = df_away_base.copy()

        if selected_scenario == 'Casa/Fora':
            # Filtra os jogos do time da casa para incluir apenas os que ele jogou em casa
            temp_df_home = temp_df_home[temp_df_home['Home'].str.contains(
                home_team.split()[0])]
            # Filtra os jogos do time visitante para incluir apenas os que ele jogou fora
            temp_df_away = temp_df_away[temp_df_away['Away'].str.contains(
                away_team.split()[0])]

        # Pega os N primeiros jogos para a análise final
        df_home = temp_df_home.head(num_jogos_selecionado)
        df_away = temp_df_away.head(num_jogos_selecionado)
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
        col1.markdown(
            f"🔼 Over {analise['over_under']['linha']} gols: **{analise['over_under']['p_over']}%**")
        col2.markdown(
            f"🔽 Under {analise['over_under']['linha']} gols: **{analise['over_under']['p_under']}%**")
        col1, col2 = st.columns(2)
        col1.markdown(f"✅ BTTS Sim: **{analise['btts']['p_btts_sim']}%**")
        col2.markdown(f"❌ BTTS Não: **{analise['btts']['p_btts_nao']}%**")

        # CARD DE VENCEDOR 

        resultados = dt.prever_gols(home_team, away_team, df_jogos,
                                    num_jogos=num_jogos_selecionado,
                                    min_jogos=3,
                                    scenario=selected_scenario)

        prob_home = round(resultados["p_home"] * 100, 2)
        prob_draw = round(resultados["p_draw"] * 100, 2)
        prob_away = round(resultados["p_away"] * 100, 2)

        if prob_home > prob_away and prob_home > prob_draw:
            vencedor = home_team
        elif prob_away > prob_home and prob_away > prob_draw:
            vencedor = away_team
        else:
            vencedor = "Empate"

        vw.card_vencedor(vencedor_nome=vencedor,
                         home_team_nome=home_team, away_team_nome=away_team)

        st.markdown("### Análise para mercados complementares")
        st.markdown("### 🔮 Top 5 Placares Mais Prováveis")
        if analise.get("placares_top"):
            cols = st.columns(min(5, len(analise['placares_top'])))
            for idx, p in enumerate(analise['placares_top']):
                with cols[idx]:
                    vw.card_placar(placar=p['placar'], prob=p['prob'])
        else:
            st.warning(
                "⚠️ Sem dados suficientes para estimar placares prováveis.")

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
        # --- CÁLCULO E EXIBIÇÃO DE TENDÊNCIAS ---
        st.markdown("### 📈 Tendências baseado nas Médias")

        # 1. Chamar as novas funções que você adicionou em data.py
        status_btts = dt.btts_status(
            media_home_gols_marcados, media_away_gols_sofridos, media_away_gols_marcados, media_home_gols_sofridos)
        status_over = dt.over_status(
            media_home_gols_marcados, media_away_gols_sofridos, media_away_gols_marcados, media_home_gols_sofridos)

        # 2. Exibir os resultados em colunas estilizadas
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style='background-color:#262730; padding:15px; border-radius:8px; text-align:center; color:white; height: 100%;'>
                <h4>Ambas Marcam (BTTS)</h4>
                <p style='font-size: 24px; font-weight: bold;'>{status_btts}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background-color:#262730; padding:15px; border-radius:8px; text-align:center; color:white; height: 100%;'>
                <h4>Over 2.5 Gols</h4>
                <p style='font-size: 24px; font-weight: bold;'>{status_over}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

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

    