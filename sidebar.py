import streamlit as st

def sidebar():
    st.sidebar.markdown(
        "<h3 style='text-align: center;'>⚽ Análise Futebol RedScore</h3>", unsafe_allow_html=True)
    
def entrada_de_dados_principal():
    """
    Cria uma única área de texto na barra lateral para colar todos os dados.
    """
    st.sidebar.markdown("### 📥 Cole os Dados da Análise")
    texto_completo = st.sidebar.text_area("", height=150)
    return texto_completo