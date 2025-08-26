# services.py (VERSÃO FINAL USANDO A LÓGICA DO SEU TXT)
import pandas as pd
import streamlit as st
import re
import datetime


def extrair_jogos_do_txt(texto_completo: str) -> pd.DataFrame:
    """
    Esta é a sua função original, adaptada para o nosso aplicativo.
    Ela é a correta para ler os dados do arquivo .txt.
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

                # A estrutura do .txt é fixa e previsível
                # O nome do time se repete, pegamos a segunda instância
                home = linhas[i+2].strip()
                placar_ft_raw = linhas[i+3].strip()
                away = linhas[i+5].strip()  # O nome do time se repete
                stats_line = linhas[i+6].strip()
                odds_line = linhas[i+7].strip()

                placar_ft = re.search(r"(\d+)\s*-\s*(\d+)", placar_ft_raw)
                estat = re.findall(r"\d+\s*-\s*\d+", stats_line)
                #odds_match = re.findall(r"\d+\.\d+", odds_line)

                # Verifica se todas as partes essenciais foram encontradas
                if placar_ft and len(estat) >= 5 and len(estat) >= 3:
                    placar_ht = [int(x) for x in estat[0].split("-")]
                    chutes = [int(x) for x in estat[1].split("-")]
                    chutes_gol = [int(x) for x in estat[2].split("-")]
                    ataques = [int(x) for x in estat[3].split("-")]
                    escanteios = [int(x) for x in estat[4].split("-")]
                    #odds = [float(x) for x in odds_match]

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
            except (ValueError, IndexError, TypeError):
                i += 1
                continue
        else:
            i += 1

    df = pd.DataFrame(jogos)
    if not df.empty:
        df = df.drop_duplicates().sort_values(
            by="Data", ascending=False).reset_index(drop=True)
        df['Data'] = pd.to_datetime(
            df['Data'], format="%d-%m-%Y", errors="coerce").dt.date
        df.dropna(subset=['Data'], inplace=True)

    return df


def processar_dados_e_identificar_times(texto_completo: str):
    """
    Função principal que identifica os times e depois chama a extração correta para o formato .txt.
    """
    if not texto_completo or texto_completo.isspace():
        return None, None, pd.DataFrame()

    regex_pattern = r'(.+?)\s+LAST\s+\d+\s+MATCHES'
    nomes_times = re.findall(regex_pattern, texto_completo, re.IGNORECASE)

    if len(nomes_times) < 2:
        st.error(
            "Erro: Não foram encontrados os dois times com o padrão 'NOME DO TIME LAST 30 MATCHES' no texto.")
        return None, None, pd.DataFrame()

    home_team = nomes_times[0].strip()
    away_team = nomes_times[1].strip()

    df_jogos = extrair_jogos_do_txt(texto_completo)

    if df_jogos.empty:
        st.warning("Os times foram identificados, mas nenhum jogo pôde ser extraído. Verifique se o texto colado tem a mesma estrutura do arquivo .txt.")

    return home_team, away_team, df_jogos
