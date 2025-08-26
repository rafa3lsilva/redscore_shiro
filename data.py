import pandas as pd
from scipy.stats import poisson, nbinom
import numpy as np


def drop_reset_index(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.index += 1
    return df


def media_gols_marcados(df: pd.DataFrame, team_name: str) -> float:
    """Calcula a média de gols MARCADOS por um time específico,
    independentemente de ser mandante ou visitante."""
    if df.empty:
        return 0.0

    # Soma os gols marcados em casa (H_Gols_FT) e fora (A_Gols_FT)
    gols_marcados = pd.concat([
        df.loc[df['Home'] == team_name, 'H_Gols_FT'],
        df.loc[df['Away'] == team_name, 'A_Gols_FT']
    ])

    return gols_marcados.mean()


def media_gols_sofridos(df, team_name):
    """Calcula a média de gols SOFRIDOS por um time específico,
    independentemente de ser mandante ou visitante."""
    if df.empty:
        return 0.0

    # Soma os gols sofridos em casa (A_Gols_FT) e fora (H_Gols_FT)
    gols_sofridos = pd.concat([
        df.loc[df['Home'] == team_name, 'A_Gols_FT'],
        df.loc[df['Away'] == team_name, 'H_Gols_FT']
    ])

    return gols_sofridos.mean()


def contar_frequencia_gols_HT_home(df):
    total_jogos = df.shape[0]
    if total_jogos == 0:
        return 0.0
    jogos_com_gols = df[df["H_Gols_HT"] > 0].shape[0]
    return jogos_com_gols / total_jogos


def contar_frequencia_gols_HT_away(df):
    total_jogos = df.shape[0]
    if total_jogos == 0:
        return 0.0
    jogos_com_gols = df[df["A_Gols_HT"] > 0].shape[0]
    return jogos_com_gols / total_jogos


def analisar_gol_ht_home_away(df_home, df_away):
    # 1. Calcula todas as frequências
    freq_home_marca = contar_frequencia_gols_HT_home(df_home)
    freq_home_sofre = contar_frequencia_gols_HT_away(df_home)

    freq_away_marca = contar_frequencia_gols_HT_away(df_away)
    freq_away_sofre = contar_frequencia_gols_HT_home(df_away)

    return {
        "home_marca": freq_home_marca * 100,
        "home_sofre": freq_home_sofre * 100,
        "away_marca": freq_away_marca * 100,
        "away_sofre": freq_away_sofre * 100,
    }


def analise_gol_ht(df_home, df_away, suavizar=True):
    df_total = pd.concat([df_home, df_away], ignore_index=True)
    total_jogos = df_total.shape[0]

    def contar_prob(sucessos, total):
        if total == 0:
            return 0.0
        return (sucessos + 1) / (total + 2) if suavizar else sucessos / total

    over_05ht_sucessos = df_total[(
        df_total['H_Gols_HT'] + df_total['A_Gols_HT']) > 0].shape[0]
    over_15ft_sucessos = df_total[(
        df_total['H_Gols_FT'] + df_total['A_Gols_FT']) > 1].shape[0]
    over_25ft_sucessos = df_total[(
        df_total['H_Gols_FT'] + df_total['A_Gols_FT']) > 2].shape[0]

    prob_05ht = contar_prob(over_05ht_sucessos, total_jogos)
    prob_15ft = contar_prob(over_15ft_sucessos, total_jogos)
    prob_25ft = contar_prob(over_25ft_sucessos, total_jogos)

    prob_final_estimada = (prob_05ht + prob_15ft + prob_25ft) / 3

    conclusao = ""
    odd_justa = 0

    if prob_final_estimada >= 0.70:
        conclusao = "✅ Probabilidade Alta de Gol HT"
    elif prob_final_estimada <= 0.60:
        conclusao = "⚠️ Probabilidade Baixa de Gol HT"
    else:
        conclusao = "🔎 Probabilidade Moderada de Gol HT"

    if prob_final_estimada > 0:
        odd_justa = 1 / prob_final_estimada

    return {
        "conclusao": conclusao,
        "probabilidade": prob_final_estimada * 100,
        "odd_justa": odd_justa,
        "media_05ht": prob_05ht * 100,
        "media_15ft": prob_15ft * 100,
        "media_25ft": prob_25ft * 100,
    }


def calc_stats_team(df, team_name):
    esc_feitos = pd.concat([
        df.loc[df['Home'] == team_name, 'H_Escanteios'],
        df.loc[df['Away'] == team_name, 'A_Escanteios']
    ])
    esc_sofridos = pd.concat([
        df.loc[df['Home'] == team_name, 'A_Escanteios'],
        df.loc[df['Away'] == team_name, 'H_Escanteios']
    ])
    finalizacoes = pd.concat([
        df.loc[df['Home'] == team_name, 'H_Chute'],
        df.loc[df['Away'] == team_name, 'A_Chute']
    ])
    ataques = pd.concat([
        df.loc[df['Home'] == team_name, 'H_Ataques'],
        df.loc[df['Away'] == team_name, 'A_Ataques']
    ])
    return {
        'esc_feitos_mean': esc_feitos.mean(),
        'esc_sofridos_mean': esc_sofridos.mean(),
        'esc_feitos_std': esc_feitos.std(),
        'finalizacoes_mean': finalizacoes.mean(),
        'ataques_mean': ataques.mean()
    }


def probabilidade_poisson_over(media_esperada, linha_str):
    try:
        parts = linha_str.split()
        tipo_linha = parts[0].lower()
        linha_num = float(parts[1])

        if tipo_linha == 'over':
            linha_int = int(linha_num) + 1
            prob = 1 - poisson.cdf(linha_int - 1, media_esperada)
        elif tipo_linha == 'under':
            linha_int = int(linha_num)
            prob = poisson.cdf(linha_int, media_esperada)
        else:
            return 0.0
        return round(prob, 4)
    except:
        return 0.0


def estimar_linha_escanteios(df_home, df_away, home_team_name, away_team_name):
    stats_home = calc_stats_team(df_home, home_team_name)
    stats_away = calc_stats_team(df_away, away_team_name)

    esc_home = (stats_home['esc_feitos_mean'] +
                stats_away['esc_sofridos_mean']) / 2
    esc_away = (stats_away['esc_feitos_mean'] +
                stats_home['esc_sofridos_mean']) / 2
    fator_ofensivo = (stats_home['finalizacoes_mean'] + stats_away['finalizacoes_mean'] +
                      stats_home['ataques_mean'] + stats_away['ataques_mean']) / 600
    esc_total_ajustado = (esc_home + esc_away) * (1 + fator_ofensivo)

    linhas_de_mercado = ['Over 8.5', 'Over 9.5', 'Over 10.5', 'Over 11.5',
                         'Under 8.5',  'Under 9.5', 'Under 10.5', 'Under 11.5']

    resultados_mercado = []
    for linha in linhas_de_mercado:
        prob = probabilidade_poisson_over(esc_total_ajustado, linha)
        odd_justa = round(1 / prob, 2) if prob > 0 else None
        resultados_mercado.append({
            'Mercado': linha,
            'Probabilidade (%)': round(prob * 100, 2),
            'Odd Justa': odd_justa
        })
    return {
        'Escanteios Mandante': round(esc_home, 2),
        'Escanteios Visitante': round(esc_away, 2),
        'Escanteios Totais Ajustados': round(esc_total_ajustado, 2),
        'Probabilidades por Mercado': resultados_mercado
    }


def analisar_consistencia_gols_ht(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    gols_ht_por_jogo = df['H_Gols_HT'] + df['A_Gols_HT']
    desvio_padrao_ht = gols_ht_por_jogo.std()
    return desvio_padrao_ht


def calcular_forca_times(df: pd.DataFrame, min_jogos: int = 3):
    media_gols_casa = df["H_Gols_FT"].mean()
    media_gols_fora = df["A_Gols_FT"].mean()
    ataque = {}
    defesa = {}
    times = pd.concat([df["Home"], df["Away"]]).unique()
    for time in times:
        jogos_casa = df[df["Home"] == time]
        jogos_fora = df[df["Away"] == time]
        n_casa = len(jogos_casa)
        n_fora = len(jogos_fora)
        ataque_casa = (jogos_casa["H_Gols_FT"].mean(
        ) / media_gols_casa) if n_casa >= min_jogos else 1
        defesa_casa = (jogos_casa["A_Gols_FT"].mean(
        ) / media_gols_fora) if n_casa >= min_jogos else 1
        ataque_fora = (jogos_fora["A_Gols_FT"].mean(
        ) / media_gols_fora) if n_fora >= min_jogos else 1
        defesa_fora = (jogos_fora["H_Gols_FT"].mean(
        ) / media_gols_casa) if n_fora >= min_jogos else 1
        ataque[time] = {"casa": ataque_casa, "fora": ataque_fora}
        defesa[time] = {"casa": defesa_casa, "fora": defesa_fora}
    return ataque, defesa, media_gols_casa, media_gols_fora


def prever_gols(home: str, away: str, df: pd.DataFrame, num_jogos: int = 6,
                min_jogos: int = 3, max_gols: int = 5, scenario: str = "Casa/Fora"):
    """
    Previsão de gols com Poisson ajustada.
    """

    if scenario == "Casa/Fora":
        df_home = df[df["Home"] == home].head(num_jogos)
        df_away = df[df["Away"] == away].head(num_jogos)
    else:  # Geral
        df_home = df[(df["Home"] == home) | (
            df["Away"] == home)].head(num_jogos)
        df_away = df[(df["Home"] == away) | (
            df["Away"] == away)].head(num_jogos)

    df_filtrado = pd.concat([df_home, df_away])
    ataque, defesa, media_gols_casa, media_gols_fora = calcular_forca_times(
        df_filtrado, min_jogos=min_jogos)

    # Verifica se os times existem nos dicionários antes de acessá-los
    if home not in ataque or away not in defesa or away not in ataque or home not in defesa:
        return {
            "lambda_home": 0, "lambda_away": 0, "matriz": np.zeros((max_gols + 1, max_gols + 1)),
            "p_home": 0, "p_draw": 0, "p_away": 0,
            "jogos_home_considerados": 0, "jogos_away_considerados": 0,
            "erro": "Dados insuficientes para um dos times no cenário selecionado."
        }

    lambda_home = ataque[home]["casa"] * defesa[away]["fora"] * media_gols_casa
    lambda_away = ataque[away]["fora"] * defesa[home]["casa"] * media_gols_fora

    probs_home = [poisson.pmf(i, lambda_home) for i in range(max_gols+1)]
    probs_away = [poisson.pmf(i, lambda_away) for i in range(max_gols+1)]
    matriz = np.outer(probs_home, probs_away)
    soma_total_matriz = matriz.sum()
    if soma_total_matriz > 0:
        matriz = matriz / soma_total_matriz

    p_home = np.tril(matriz, -1).sum()
    p_away = np.triu(matriz, 1).sum()
    p_draw = np.trace(matriz)

    return {
        "lambda_home": lambda_home, "lambda_away": lambda_away, "matriz": matriz,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "jogos_home_considerados": len(df_home), "jogos_away_considerados": len(df_away),
    }


def calcular_over_under(resultados: dict, linha: float = 2.5):
    matriz = resultados["matriz"]
    max_gols = matriz.shape[0] - 1
    p_over = 0
    p_under = 0
    for i in range(max_gols+1):
        for j in range(max_gols+1):
            total_gols = i + j
            if total_gols > linha:
                p_over += matriz[i, j]
            else:
                p_under += matriz[i, j]
    return {"linha": linha, "p_over": round(p_over * 100, 2), "p_under": round(p_under * 100, 2)}


def calcular_btts(resultados: dict):
    matriz = resultados["matriz"]
    max_gols = matriz.shape[0] - 1
    p_btts_sim = 0
    p_btts_nao = 0
    for i in range(max_gols+1):
        for j in range(max_gols+1):
            if i > 0 and j > 0:
                p_btts_sim += matriz[i, j]
            else:
                p_btts_nao += matriz[i, j]
    return {"p_btts_sim": round(p_btts_sim * 100, 2), "p_btts_nao": round(p_btts_nao * 100, 2)}


def analisar_cenario_partida(
    home: str, away: str, df: pd.DataFrame, num_jogos: int = 6,
    min_jogos: int = 3, max_gols: int = 5, scenario: str = "Casa/Fora", linha_gols: float = 2.5
):
    times_historicos = pd.unique(df[['Home', 'Away']].values.ravel('K'))
    if home not in times_historicos:
        return {"erro": f"Não há dados históricos suficientes para a equipa: {home}"}
    if away not in times_historicos:
        return {"erro": f"Não há dados históricos suficientes para a equipa: {away}"}

    resultados = prever_gols(
        home, away, df, num_jogos=num_jogos,
        min_jogos=min_jogos, max_gols=max_gols, scenario=scenario
    )

    if "erro" in resultados:
        return {"erro": resultados["erro"]}

    matriz = resultados["matriz"]
    p_home = resultados["p_home"] * 100
    p_draw = resultados["p_draw"] * 100
    p_away = resultados["p_away"] * 100

    over_under = calcular_over_under(resultados, linha=linha_gols)
    btts = calcular_btts(resultados)

    flat_probs = matriz.flatten()
    top_indices = flat_probs.argsort()[-5:][::-1]
    placares_provaveis = []
    for idx in top_indices:
        i, j = divmod(idx, matriz.shape[1])
        placares_provaveis.append({
            "placar": f"{i} x {j}", "prob": round(flat_probs[idx] * 100, 2)
        })

    return {
        "cenario_usado": scenario, "lambda_home": resultados["lambda_home"], "lambda_away": resultados["lambda_away"],
        "prob_home": round(p_home, 2), "prob_draw": round(p_draw, 2), "prob_away": round(p_away, 2),
        "placares_top": placares_provaveis, "over_under": over_under, "btts": btts,
    }


def _stats_ht(df, time, min_jogos, liga_ht_home, liga_ht_away, scenario, num_jogos):
    if scenario == "Casa/Fora":
        jogos_casa = df[df["Home"] == time]
        jogos_fora = df[df["Away"] == time]
    else:
        jogos_casa = df[(df["Home"] == time) | (df["Away"] == time)]
        jogos_fora = jogos_casa

    n_c, n_f = len(jogos_casa), len(jogos_fora)
    atk_c = (jogos_casa["H_Gols_HT"].mean() /
             liga_ht_home) if n_c >= min_jogos else 1.0
    def_c = (jogos_casa["A_Gols_HT"].mean() /
             liga_ht_away) if n_c >= min_jogos else 1.0
    atk_f = (jogos_fora["A_Gols_HT"].mean() /
             liga_ht_away) if n_f >= min_jogos else 1.0
    def_f = (jogos_fora["H_Gols_HT"].mean() /
             liga_ht_home) if n_f >= min_jogos else 1.0
    return {"atk_c": atk_c, "def_c": def_c, "atk_f": atk_f, "def_f": def_f}


def prever_gol_ht(
    home: str, away: str, df: pd.DataFrame, num_jogos: int = 6,
    min_jogos: int = 3, scenario: str = "Casa/Fora", max_gols_ht: int = 3,
):

    if scenario == "Casa/Fora":
        df_home = df[df["Home"] == home].head(num_jogos)
        df_away = df[df["Away"] == away].head(num_jogos)
    else:
        df_home = df[(df["Home"] == home) | (
            df["Away"] == home)].head(num_jogos)
        df_away = df[(df["Home"] == away) | (
            df["Away"] == away)].head(num_jogos)

    liga_ht_home = df["H_Gols_HT"].mean()
    liga_ht_away = df["A_Gols_HT"].mean()

    s_home = _stats_ht(df, home, min_jogos, liga_ht_home,
                       liga_ht_away, scenario, num_jogos)
    s_away = _stats_ht(df, away, min_jogos, liga_ht_home,
                       liga_ht_away, scenario, num_jogos)

    lam_home_ht = s_home["atk_c"] * s_away["def_f"] * liga_ht_home
    lam_away_ht = s_away["atk_f"] * s_home["def_c"] * liga_ht_away

    probs_h = [poisson.pmf(i, lam_home_ht) for i in range(max_gols_ht + 1)]
    probs_a = [poisson.pmf(i, lam_away_ht) for i in range(max_gols_ht + 1)]
    matriz_ht = np.outer(probs_h, probs_a)

    lam_total_ht = lam_home_ht + lam_away_ht
    p_gol_ht = 1 - np.exp(-lam_total_ht)
    p_exato1_ht = lam_total_ht * np.exp(-lam_total_ht)

    return {
        "lambda_home_ht": lam_home_ht, "lambda_away_ht": lam_away_ht, "lambda_total_ht": lam_total_ht,
        "p_gol_ht": round(p_gol_ht * 100, 2), "p_exato1_ht": round(p_exato1_ht * 100, 2), "matriz_ht": matriz_ht,
        "jogos_home_considerados": len(df_home), "jogos_away_considerados": len(df_away), "cenario_usado": scenario,
    }


def _fit_nb_params(mu, var, eps=1e-9):
    if np.isnan(mu) or np.isnan(var) or var <= mu + eps:
        return None
    r = (mu * mu) / (var - mu)
    p = r / (r + mu)
    return r, p


def _pmf_nb_or_poisson(k_max, mu, var):
    params = _fit_nb_params(mu, var)
    if params is None:
        return [poisson.pmf(k, mu) for k in range(k_max + 1)]
    r, p = params
    return [nbinom.pmf(k, r, p) for k in range(k_max + 1)]


def prever_escanteios_nb(
    home: str, away: str, df: pd.DataFrame, num_jogos: int = 6,
    scenario: str = "Casa/Fora", max_cantos: int = 20,
):

    if scenario == "Casa/Fora":
        h = df[df["Home"] == home].head(num_jogos)
        a = df[df["Away"] == away].head(num_jogos)
        mu_h = h["H_Escanteios"].mean()
        var_h = h["H_Escanteios"].var(ddof=1)
        mu_a = a["A_Escanteios"].mean()
        var_a = a["A_Escanteios"].var(ddof=1)
    else:
        h_all = df[(df["Home"] == home) | (df["Away"] == home)].head(num_jogos)
        a_all = df[(df["Home"] == away) | (df["Away"] == away)].head(num_jogos)
        vals_h = np.where(h_all["Home"] == home,
                          h_all["H_Escanteios"], h_all["A_Escanteios"])
        vals_a = np.where(a_all["Away"] == away,
                          a_all["A_Escanteios"], a_all["H_Escanteios"])
        mu_h = np.mean(vals_h)
        var_h = np.var(vals_h, ddof=1)
        mu_a = np.mean(vals_a)
        var_a = np.var(vals_a, ddof=1)

    probs_h = _pmf_nb_or_poisson(max_cantos, mu_h, var_h)
    probs_a = _pmf_nb_or_poisson(max_cantos, mu_a, var_a)
    matriz = np.outer(probs_h, probs_a)

    return {
        "mu_home_cantos": mu_h, "mu_away_cantos": mu_a, "matriz_cantos": matriz,
        "cenario_usado": scenario, "jogos_home_considerados": len(h) if scenario == "Casa/Fora" else len(h_all),
        "jogos_away_considerados": len(a) if scenario == "Casa/Fora" else len(a_all),
    }


def calcular_over_under_cantos(resultados_cantos: dict, linha_total: float = 10.5):
    M = resultados_cantos["matriz_cantos"]
    kmax_h, kmax_a = M.shape[0]-1, M.shape[1]-1
    p_over = 0.0
    p_under = 0.0
    for i in range(kmax_h + 1):
        for j in range(kmax_a + 1):
            total = i + j
            if total > linha_total:
                p_over += M[i, j]
            else:
                p_under += M[i, j]
    return {"linha": linha_total, "p_over": round(p_over * 100, 2), "p_under": round(p_under * 100, 2)}


def prob_home_mais_cantos(resultados_cantos: dict):
    M = resultados_cantos["matriz_cantos"]
    p_home = np.tril(M, -1).sum()
    p_away = np.triu(M, 1).sum()
    p_emp = np.trace(M)
    return {
        "home_mais": round(p_home * 100, 2),
        "empate": round(p_emp * 100, 2),
        "away_mais": round(p_away * 100, 2),
    }


def btts_status(media_home_marcados, media_away_sofridos, media_away_marcados, media_home_sofridos):
    # O cálculo da expectativa de gols por time está correto
    btts_home = (media_home_marcados + media_away_sofridos) / 2
    btts_away = (media_away_marcados + media_home_sofridos) / 2

    # Sugestão de limites mais realistas para a ocorrência de 1 gol de cada lado
    if btts_home > 1.4 and btts_away > 1.4:
        return "🟢 Alta chance"
    elif btts_home > 1.2 and btts_away > 1.2:
        return "🟡 Moderada"
    else:
        return "🔴 Baixa"


def over_status(media_home_marcados, media_away_sofridos, media_away_marcados, media_home_sofridos):
    # Calcula a expectativa de gols para cada time
    expected_home_goals = (media_home_marcados + media_away_sofridos) / 2
    expected_away_goals = (media_away_marcados + media_home_sofridos) / 2

    # SOMA as duas expectativas para ter o total de gols esperado na partida
    total_expected_goals = expected_home_goals + expected_away_goals

    # Compara o TOTAL esperado com os limites para Over 2.5
    if total_expected_goals > 2.7:
        return "🟢 Alta chance"
    elif total_expected_goals > 2.4:
        return "🟡 Moderada"
    else:
        return "🔴 Baixa"
