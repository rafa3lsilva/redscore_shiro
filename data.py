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
    """
    Calcula a probabilidade de gol no HT usando o método padronizado 
    (dados combinados + suavização).
    """
    # 1. Unifica os DataFrames, assim como no Painel de Mercados
    df_total = pd.concat([df_home, df_away], ignore_index=True)
    total_jogos = df_total.shape[0]

    # Função de suavização (a mesma do Painel de Mercados)
    def contar_prob(sucessos, total):
        if total == 0:
            return 0.0
        return (sucessos + 1) / (total + 2) if suavizar else sucessos / total

    # 2. Conta os eventos (sucessos) para cada mercado no DataFrame unificado
    over_05ht_sucessos = df_total[(
        df_total['H_Gols_HT'] + df_total['A_Gols_HT']) > 0].shape[0]
    over_15ft_sucessos = df_total[(
        df_total['H_Gols_FT'] + df_total['A_Gols_FT']) > 1].shape[0]
    over_25ft_sucessos = df_total[(
        df_total['H_Gols_FT'] + df_total['A_Gols_FT']) > 2].shape[0]

    # 3. Calcula a probabilidade de cada mercado usando o método de suavização
    prob_05ht = contar_prob(over_05ht_sucessos, total_jogos)
    prob_15ft = contar_prob(over_15ft_sucessos, total_jogos)
    prob_25ft = contar_prob(over_25ft_sucessos, total_jogos)

    # 4. Calcula a média final das probabilidades, como na regra original
    prob_final_estimada = (prob_05ht + prob_15ft + prob_25ft) / 3

    # 5. Define a conclusão e a odd justa
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

    # 6. Retorna o dicionário com todos os resultados
    return {
        "conclusao": conclusao,
        "probabilidade": prob_final_estimada * 100,
        "odd_justa": odd_justa,
        "media_05ht": prob_05ht * 100,
        "media_15ft": prob_15ft * 100,
        "media_25ft": prob_25ft * 100,
    }

# Função para calcular estatísticas dos times
def calc_stats_team(df, team_name):
    """Calcula as estatísticas para um time específico dentro de um DataFrame."""
    # Escanteios feitos pelo time
    esc_feitos = pd.concat([
        df.loc[df['Home'] == team_name, 'H_Escanteios'],
        df.loc[df['Away'] == team_name, 'A_Escanteios']
    ])
    # Escanteios sofridos pelo time
    esc_sofridos = pd.concat([
        df.loc[df['Home'] == team_name, 'A_Escanteios'],
        df.loc[df['Away'] == team_name, 'H_Escanteios']
    ])

    # Finalizações feitas pelo time
    finalizacoes = pd.concat([
        df.loc[df['Home'] == team_name, 'H_Chute'],
        df.loc[df['Away'] == team_name, 'A_Chute']
    ])

    # Ataques feitos pelo time
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

# Função para calcular probabilidade de bater o over usando Poisson
def probabilidade_poisson_over(media_esperada, linha_str):
    try:
        # 1. Separa o tipo (Over/Under) e o número da linha
        parts = linha_str.split()
        # Converte para minúsculas ('over' ou 'under')
        tipo_linha = parts[0].lower()
        linha_num = float(parts[1])   # Ex: 9.5

        # 2. Aplica a fórmula correta para cada cenário
        if tipo_linha == 'over':
            # Probabilidade de ser MAIOR OU IGUAL ao próximo inteiro
            # Ex: Over 9.5 significa P(X >= 10)
            linha_int = int(linha_num) + 1
            prob = 1 - poisson.cdf(linha_int - 1, media_esperada)

        elif tipo_linha == 'under':
            # Probabilidade de ser MENOR OU IGUAL ao inteiro anterior
            # Ex: Under 9.5 significa P(X <= 9)
            linha_int = int(linha_num)
            prob = poisson.cdf(linha_int, media_esperada)

        else:
            return 0.0  # Retorna 0 se a linha não for 'Over' ou 'Under'

        return round(prob, 4)
    except:
        return 0.0

# Função principal
def estimar_linha_escanteios(df_home, df_away, home_team_name, away_team_name):
    stats_home = calc_stats_team(df_home, home_team_name)
    stats_away = calc_stats_team(df_away, away_team_name)

    # 1. Calcula a média de escanteios esperada (lógica mantida)
    esc_home = (stats_home['esc_feitos_mean'] +
                stats_away['esc_sofridos_mean']) / 2
    esc_away = (stats_away['esc_feitos_mean'] +
                stats_home['esc_sofridos_mean']) / 2
    fator_ofensivo = (stats_home['finalizacoes_mean'] + stats_away['finalizacoes_mean'] +
                      stats_home['ataques_mean'] + stats_away['ataques_mean']) / 600
    esc_total_ajustado = (esc_home + esc_away) * (1 + fator_ofensivo)

    # 2. Define uma lista de linhas de mercado padrão para analisar
    linhas_de_mercado = ['Over 8.5', 'Over 9.5', 'Over 10.5', 'Over 11.5',
                         'Under 8.5',  'Under 9.5', 'Under 10.5', 'Under 11.5']

    # 3. Calcula a probabilidade para CADA linha de mercado
    resultados_mercado = []
    for linha in linhas_de_mercado:
        prob = probabilidade_poisson_over(esc_total_ajustado, linha)
        odd_justa = round(1 / prob, 2) if prob > 0 else None
        resultados_mercado.append({
            'Mercado': linha,
            'Probabilidade (%)': round(prob * 100, 2),
            'Odd Justa': odd_justa
        })

    # 4. Retorna um dicionário com a média e a lista de probabilidades
    return {
        'Escanteios Mandante': round(esc_home, 2),
        'Escanteios Visitante': round(esc_away, 2),
        'Escanteios Totais Ajustados': round(esc_total_ajustado, 2),
        'Probabilidades por Mercado': resultados_mercado
    }

# Melhoria nas métricas de gol no HT
def analisar_consistencia_gols_ht(df: pd.DataFrame) -> float:
    """
    Calcula o desvio padrão dos gols totais no primeiro tempo (HT)
    de uma amostra de jogos.
    """
    # Garante que temos pelo menos 2 jogos para calcular o desvio padrão
    if len(df) < 2:
        return 0.0

    # Cria uma nova coluna somando os gols de casa e fora no HT para cada jogo
    gols_ht_por_jogo = df['H_Gols_HT'] + df['A_Gols_HT']

    # Calcula e retorna o desvio padrão dessa série de gols
    desvio_padrao_ht = gols_ht_por_jogo.std()

    return desvio_padrao_ht


def calcular_forca_times(df: pd.DataFrame, min_jogos: int = 3):
    """
    Calcula força de ataque e defesa de cada time em relação à média da liga.
    Se o time tiver menos que 'min_jogos', suas estatísticas são puxadas para a média.
    """
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

        # Ajuste: se poucos jogos, puxa para a média
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
    scenario: "Geral" ou "Casa/Fora"
    """

    if scenario == "Casa/Fora":
        # Últimos N jogos em casa do mandante
        df_home = df[df["Home"] == home].tail(num_jogos)
        # Últimos N jogos fora do visitante
        df_away = df[df["Away"] == away].tail(num_jogos)
    else:  # Geral
        # Últimos N jogos do time, independentemente do mando
        df_home = df[(df["Home"] == home) | (
            df["Away"] == home)].tail(num_jogos)
        df_away = df[(df["Home"] == away) | (
            df["Away"] == away)].tail(num_jogos)

    # Junta os jogos filtrados
    df_filtrado = pd.concat([df_home, df_away])

    # Forças ajustadas
    ataque, defesa, media_gols_casa, media_gols_fora = calcular_forca_times(
        df_filtrado, min_jogos=min_jogos)

    lambda_home = ataque[home]["casa"] * defesa[away]["fora"] * media_gols_casa
    lambda_away = ataque[away]["fora"] * defesa[home]["casa"] * media_gols_fora
    
    # λ esperados
    lambda_home = ataque[home]["casa"] * defesa[away]["fora"] * media_gols_casa
    lambda_away = ataque[away]["fora"] * defesa[home]["casa"] * media_gols_fora

    # Distribuições
    probs_home = [poisson.pmf(i, lambda_home) for i in range(max_gols+1)]
    probs_away = [poisson.pmf(i, lambda_away) for i in range(max_gols+1)]

    matriz = np.outer(probs_home, probs_away)

    # Calcula a soma total da matriz
    soma_total_matriz = matriz.sum()

    # Divide cada célula pela soma total para que a nova soma seja 1.
    if soma_total_matriz > 0:
        matriz = matriz / soma_total_matriz

    # Probabilidades agregadas
    p_home = np.tril(matriz, -1).sum()
    p_away = np.triu(matriz, 1).sum()
    p_draw = np.trace(matriz)

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "matriz": matriz,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "jogos_home_considerados": len(df_home),
        "jogos_away_considerados": len(df_away),
    }


def calcular_over_under(resultados: dict, linha: float = 2.5):
    """
    Calcula probabilidades de Over/Under X gols
    com base na matriz de placares prevista pelo modelo Poisson.
    
    resultados: dict retornado por prever_gols
    linha: float, ex.: 2.5 ou 3.5
    """
    matriz = resultados["matriz"]
    max_gols = matriz.shape[0] - 1

    p_over = 0
    p_under = 0

    for i in range(max_gols+1):   # gols home
        for j in range(max_gols+1):  # gols away
            total_gols = i + j
            if total_gols > linha:
                p_over += matriz[i, j]
            else:
                p_under += matriz[i, j]

    return {
        "linha": linha,
        "p_over": round(p_over * 100, 2),
        "p_under": round(p_under * 100, 2),
    }


def calcular_btts(resultados: dict):
    """
    Calcula probabilidades de Both Teams to Score (BTTS) 
    com base na matriz de placares prevista pelo modelo Poisson.
    
    resultados: dict retornado por prever_gols
    """
    matriz = resultados["matriz"]
    max_gols = matriz.shape[0] - 1

    p_btts_sim = 0
    p_btts_nao = 0

    for i in range(max_gols+1):   # gols home
        for j in range(max_gols+1):  # gols away
            if i > 0 and j > 0:
                p_btts_sim += matriz[i, j]
            else:
                p_btts_nao += matriz[i, j]

    return {
        "p_btts_sim": round(p_btts_sim * 100, 2),
        "p_btts_nao": round(p_btts_nao * 100, 2),
    }


def analisar_cenario_partida(
    home: str,
    away: str,
    df: pd.DataFrame,
    num_jogos: int = 6,
    min_jogos: int = 3,
    max_gols: int = 5,
    scenario: str = "Casa/Fora",
    linha_gols: float = 2.5
):
    """
    Consolida a análise do cenário da partida:
    - Probabilidades 1X2
    - Over/Under
    - BTTS
    - Placar mais provável
    - Cenário usado
    """
    # verifica se há dados históricos
    times_historicos = pd.unique(df[['Home', 'Away']].values.ravel('K'))

    if home not in times_historicos:
        return {"erro": f"Não há dados históricos suficientes para a equipa: {home}"}

    if away not in times_historicos:
        return {"erro": f"Não há dados históricos suficientes para a equipa: {away}"}
   
    # --- Calcula a matriz de gols esperados
    resultados = prever_gols(
        home, away, df,
        num_jogos=num_jogos,
        min_jogos=min_jogos,
        max_gols=max_gols,
        scenario=scenario
    )

    matriz = resultados["matriz"]

    # --- 1. Probabilidades 1X2
    p_home = resultados["p_home"] * 100
    p_draw = resultados["p_draw"] * 100
    p_away = resultados["p_away"] * 100

    # --- 2. Over/Under
    over_under = calcular_over_under(resultados, linha=linha_gols)

    # --- 3. BTTS
    btts = calcular_btts(resultados)

    # --- 4. Top 5 Placar(es) mais prováveis
    flat_probs = matriz.flatten()
    top_indices = flat_probs.argsort()[-5:][::-1]  # 5 maiores probabilidades

    placares_provaveis = []
    for idx in top_indices:
        i, j = divmod(idx, matriz.shape[1])
        placares_provaveis.append({
            "placar": f"{i} x {j}",
            "prob": round(flat_probs[idx] * 100, 2)
        })

    # --- Retorno consolidado
    return {
        "cenario_usado": scenario,
        "lambda_home": resultados["lambda_home"],
        "lambda_away": resultados["lambda_away"],
        "prob_home": round(p_home, 2),
        "prob_draw": round(p_draw, 2),
        "prob_away": round(p_away, 2),
        "placares_top": placares_provaveis,
        "over_under": over_under,
        "btts": btts,
    }


def _stats_ht(df, time, min_jogos, liga_ht_home, liga_ht_away, scenario, num_jogos):
    if scenario == "Casa/Fora":
        # Apenas jogos em casa para o mandante
        jogos_casa = df[df["Home"] == time].tail(num_jogos)
        # Apenas jogos fora para o visitante
        jogos_fora = df[df["Away"] == time].tail(num_jogos)
    else:
        # Todos os jogos, independente de casa/fora
        jogos_casa = df[(df["Home"] == time) | (
            df["Away"] == time)].tail(num_jogos)
        jogos_fora = jogos_casa  # usa o mesmo conjunto

    n_c, n_f = len(jogos_casa), len(jogos_fora)

    # Ataque/defesa relativos no HT
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
    home: str,
    away: str,
    df: pd.DataFrame,
    num_jogos: int = 6,
    min_jogos: int = 3,
    scenario: str = "Casa/Fora",
    max_gols_ht: int = 3,
):
    """
    Probabilidade de gol no 1º tempo:
      - P(>=1 gol no HT)  (Over 0.5 HT)
      - P(exatamente 1 gol no HT)
    Também retorna λ_home_ht e λ_away_ht e a matriz de gols HT (0..max_gols_ht).
    """

    # Filtra conforme cenário e últimos N
    if scenario == "Casa/Fora":
        df_home = df[df["Home"] == home].tail(num_jogos)
        df_away = df[df["Away"] == away].tail(num_jogos)
    else:
        df_home = df[(df["Home"] == home) | (
            df["Away"] == home)].tail(num_jogos)
        df_away = df[(df["Home"] == away) | (
            df["Away"] == away)].tail(num_jogos)

    # Médias da liga no HT
    liga_ht_home = df["H_Gols_HT"].mean()
    liga_ht_away = df["A_Gols_HT"].mean()

    # Forças relativas no HT com shrink
    s_home = _stats_ht(df, home, min_jogos, liga_ht_home,
                       liga_ht_away, scenario, num_jogos)
    s_away = _stats_ht(df, away, min_jogos, liga_ht_home,
                    liga_ht_away, scenario, num_jogos)


    # λ esperados no HT (mesma lógica do FT, mas com colunas de HT)
    lam_home_ht = s_home["atk_c"] * s_away["def_f"] * liga_ht_home
    lam_away_ht = s_away["atk_f"] * s_home["def_c"] * liga_ht_away

    # Distribuição conjunta HT (assumindo independência dos processos de gol no HT)
    probs_h = [poisson.pmf(i, lam_home_ht) for i in range(max_gols_ht + 1)]
    probs_a = [poisson.pmf(i, lam_away_ht) for i in range(max_gols_ht + 1)]
    matriz_ht = np.outer(probs_h, probs_a)

    # Probabilidades agregadas do total no HT
    # Total ~ Poisson(lam_total) quando soma de Poissons independentes
    lam_total_ht = lam_home_ht + lam_away_ht
    p_gol_ht = 1 - np.exp(-lam_total_ht)               # P(>=1)
    p_exato1_ht = lam_total_ht * np.exp(-lam_total_ht)  # P(exatamente 1)

    return {
        "lambda_home_ht": lam_home_ht,
        "lambda_away_ht": lam_away_ht,
        "lambda_total_ht": lam_total_ht,
        "p_gol_ht": round(p_gol_ht * 100, 2),           # Over 0.5 HT
        # Exatamente 1 gol no HT
        "p_exato1_ht": round(p_exato1_ht * 100, 2),
        "matriz_ht": matriz_ht,
        "jogos_home_considerados": len(df_home),
        "jogos_away_considerados": len(df_away),
        "cenario_usado": scenario,
    }


def _fit_nb_params(mu, var, eps=1e-9):
    """
    Ajuste dos parâmetros da Negativo Binomial na parametrização (r, p).
    Se não houver overdispersão (var <= mu), retorna None (usar Poisson).
    """
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
    home: str,
    away: str,
    df: pd.DataFrame,
    num_jogos: int = 6,
    scenario: str = "Casa/Fora",
    max_cantos: int = 20,
):
    """
    Modela escanteios com Negativo Binomial por time (fallback Poisson),
    e retorna matriz conjunta assumindo independência.
    """

    if scenario == "Casa/Fora":
        h = df[df["Home"] == home].tail(num_jogos)
        a = df[df["Away"] == away].tail(num_jogos)
        mu_h = h["H_Escanteios"].mean()
        var_h = h["H_Escanteios"].var(ddof=1)
        mu_a = a["A_Escanteios"].mean()
        var_a = a["A_Escanteios"].var(ddof=1)
    else:
        h_all = df[(df["Home"] == home) | (df["Away"] == home)].tail(num_jogos)
        a_all = df[(df["Home"] == away) | (df["Away"] == away)].tail(num_jogos)
        # médias do time considerando o lado em cada jogo
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
    matriz = np.outer(probs_h, probs_a)  # P(H=k, A=j)

    return {
        "mu_home_cantos": mu_h,
        "mu_away_cantos": mu_a,
        "matriz_cantos": matriz,
        "cenario_usado": scenario,
        "jogos_home_considerados": len(h) if scenario == "Casa/Fora" else len(h_all),
        "jogos_away_considerados": len(a) if scenario == "Casa/Fora" else len(a_all),
    }


def calcular_over_under_cantos(resultados_cantos: dict, linha_total: float = 10.5):
    """
    Probabilidade de Over/Under X cantos (total do jogo), usando matriz conjunta.
    """
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
    """
    Probabilidades de quem tem mais escanteios e de empate em cantos.
    """
    M = resultados_cantos["matriz_cantos"]
    p_home = np.tril(M, -1).sum()
    p_away = np.triu(M, 1).sum()
    p_emp = np.trace(M)
    return {
        "home_mais": round(p_home * 100, 2),
        "empate": round(p_emp * 100, 2),
        "away_mais": round(p_away * 100, 2),
    }
