import pandas as pd
import streamlit as st
import re
import datetime

#--------------------------
# FUNÇÃO PARA EXTRAIR JOGOS 
#--------------------------


def extrair_jogos_corrigida(texto_completo: str) -> pd.DataFrame:
    """
    Versão final e robusta que usa a linha de estatísticas como âncora e
    torna a busca pelo cabeçalho opcional dentro de cada bloco de texto.
    """
    linhas = texto_completo.strip().splitlines()
              
    if not linhas:
        return pd.DataFrame()

    jogos = []


    i = 0
    ano_atual = datetime.date.today().year
    padrao_data_liga = r'^\d{1,2}\.\d{1,2}\s+'

    while i < len(linhas):
        linha_atual = linhas[i].strip()

        # Usa o padrão original para encontrar o início de um jogo (linha com data + liga)
        if re.match(padrao_data_liga, linha_atual):
            try:
                # Extrai a data e a liga desta linha
                data_str = re.match(r'^\d{1,2}\.\d{1,2}', linha_atual).group(
                    0).replace('.', '-')
                data_formatada = f"{data_str}-{ano_atual}"
                liga = re.sub(padrao_data_liga, '', linha_atual).strip()

                # O nome do time se repete, pegamos a segunda instância
                home = linhas[i+2].strip()
                placar_ft_raw = linhas[i+3].strip()
                away = linhas[i+4].strip()  # O nome do time se repete
                stats_line = linhas[i+6].strip()
                odds_line = linhas[i+7].strip()

                placar_ft = re.search(r"(\d+)\s*-\s*(\d+)", placar_ft_raw)
                estat = re.findall(r"\d+\s*-\s*\d+", stats_line)
                # odds_match = re.findall(r"\d+\.\d+", odds_line)

                # Verifica se todas as partes essenciais foram encontradas
                if placar_ft and len(estat) >= 5 and len(estat) >= 3:
                    placar_ht = [int(x) for x in estat[0].split("-")]
                    chutes = [int(x) for x in estat[1].split("-")]
                    chutes_gol = [int(x) for x in estat[2].split("-")]
                    ataques = [int(x) for x in estat[3].split("-")]
                    escanteios = [int(x) for x in estat[4].split("-")]
                    # odds = [float(x) for x in odds_match]

                jogos.append({
                        "Data": data_formatada, "Liga": liga, "Home": home, "Away": away,
                        "H_Gols_FT": int(placar_ft.group(1)), "A_Gols_FT": int(placar_ft.group(2)),
                        "H_Gols_HT": placar_ht[0], "A_Gols_HT": placar_ht[1],
                        "H_Chute": chutes[0], "A_Chute": chutes[1],
                        "H_Chute_Gol": chutes_gol[0], "A_Chute_Gol": chutes_gol[1],
                        "H_Ataques": ataques[0], "A_Ataques": ataques[1],
                        "H_Escanteios": escanteios[0], "A_Escanteios": escanteios[1],
                        #"Odd_H": odds[0], "Odd_D": odds[1], "Odd_A": odds[2]
                        })
                i += 8  # Pula para o próximo bloco de jogo
            except (ValueError, IndexError, TypeError, AttributeError):
                i += 1  # Se falhar, apenas avança uma linha    
                continue
        else:
            i += 1

    df = pd.DataFrame(jogos)
    if not df.empty:
        df = df.drop_duplicates().reset_index(drop=True)

    return df


def processar_dados_e_identificar_times(texto_completo: str):
    """
    Função principal que identifica os times, divide o texto em blocos
    e extrai os jogos de cada bloco separadamente, retornando 4 itens.
    """
    if not texto_completo or texto_completo.isspace():
        return None, None, pd.DataFrame(), pd.DataFrame()

    regex_pattern = r'(.+?)\s+LAST\s+\d+\s+MATCHES'
    matches = list(re.finditer(regex_pattern, texto_completo, re.IGNORECASE))

    if len(matches) < 2:
        st.error(
            "Erro: Não foram encontrados os dois blocos de times com o padrão 'NOME DO TIME LAST X MATCHES'.")
        return None, None, pd.DataFrame(), pd.DataFrame()

    home_team_official = matches[0].group(1).strip()
    away_team_official = matches[1].group(1).strip()

    start_block_home = matches[0].end()
    start_block_away = matches[1].end()

    texto_bloco_home = texto_completo[start_block_home:start_block_away]
    texto_bloco_away = texto_completo[start_block_away:]

    # A função corrigida é chamada aqui para cada bloco de texto
    df_home_jogos = extrair_jogos_corrigida(texto_bloco_home)
    df_away_jogos = extrair_jogos_corrigida(texto_bloco_away)

    if df_home_jogos.empty or df_away_jogos.empty:
        st.warning(
            "Um ou ambos os times foram identificados, mas os jogos não puderam ser extraídos. Verifique o formato dos dados.")

    return home_team_official, away_team_official, df_home_jogos, df_away_jogos
