import streamlit as st

def sidebar():
    st.sidebar.markdown(
        "<h3 style='text-align: center;'>⚽ Análise Futebol RedScore</h3>", unsafe_allow_html=True)
    tutorial_url = "https://www.notion.so/Tutorial-Redscore-2504bab1283b80f1af08fca804deb248"

    st.sidebar.markdown(f"""
        <div style="text-align: center; font-size: 16px;">
            <a href="{tutorial_url}" target="_blank" style="text-decoration: none;">
                <div style="margin-bottom: 10px; background-color:#1f77b4; padding:8px; border-radius:6px; color:white;">
                    📚 Tutorial
                </div>
            </a>
        </div>
        <br>
    """, unsafe_allow_html=True)

def entrada_de_dados_principal():
    """
    Cria uma única área de texto na barra lateral para colar todos os dados.
    """
    st.sidebar.markdown("### 📥 Cole os Dados da Análise")
    texto_completo = st.sidebar.text_area("", height=150)
    return texto_completo