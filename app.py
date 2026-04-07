import streamlit as st
import pandas as pd
import joblib
import os
import spacy
import spacy.cli
import numpy as np
from datetime import datetime

st.set_page_config(page_title="XPTO Triagem Inteligente", layout="wide")
st.markdown("""
<style>
textarea {
    font-size: 20px !important;
}           
.stTextArea label p {
    font-size: 20px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'simulador'

@st.cache_resource
def load_resources():
    try:
        nlp = spacy.load('pt_core_news_sm', disable=['ner', 'parser'])
    except OSError:
        spacy.cli.download('pt_core_news_sm')
        nlp = spacy.load('pt_core_news_sm', disable=['ner', 'parser'])
    
    # Carregando Modelos .joblib
    m_macro_of = joblib.load('models/macro_oficial.joblib')
    m_det_of = joblib.load('models/detalhada_oficial.joblib')
    m_macro_ov = joblib.load('models/macro_overfit.joblib')
    m_det_ov = joblib.load('models/detalhada_overfit.joblib')
    
    return nlp, m_macro_of, m_det_of, m_macro_ov, m_det_ov

nlp, m_macro_of, m_det_of, m_macro_ov, m_det_ov = load_resources()

# SIMULADOR


if st.session_state.page == 'simulador':
    col_t, col_b = st.columns([8, 2])
    with col_t:
        st.title("🚀 XPTO Data Solutions - Simulador")
        st.markdown("## Interface de validação para **Classificação Hierárquica**.")
    with col_b:
        st.write("")
        if st.button("INSIGHTS ➔"):
            st.session_state.page = 'dashboard'
            st.rerun()

    st.markdown("#### Digite um chamado para testar a resposta dos modelos.")
    texto_usuario = st.text_area("Chamado:", placeholder="Ex: Meu boleto veio com valor errado e preciso de estorno...")

    if st.button("PROCESSAR E CLASSIFICAR"):
        if texto_usuario.strip() == "":
            st.warning("Por favor, digite um texto para analisar.")
        else:
            # 1. PRÉ-PROCESSAMENTO 
            doc = nlp(texto_usuario.lower().strip())
            tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
            texto_processado = " ".join(tokens)
            
            # 2. MODO OFICIAL 
            # Macro
            prob_mac_of = m_macro_of.predict_proba([texto_processado])[0]
            cat_mac_of = m_macro_of.classes_[np.argmax(prob_mac_of)]
            conf_mac_of = max(prob_mac_of) * 100
            # Detalhado
            prob_det_of = m_det_of.predict_proba([texto_processado])[0]
            cat_det_of = m_det_of.classes_[np.argmax(prob_det_of)]
            conf_det_of = max(prob_det_of) * 100

            # 3. MODO OVERFIT 
            # Macro
            prob_mac_ov = m_macro_ov.predict_proba([texto_processado])[0]
            cat_mac_ov = m_macro_ov.classes_[np.argmax(prob_mac_ov)]
            conf_mac_ov = max(prob_mac_ov) * 100
            # Detalhado
            prob_det_ov = m_det_ov.predict_proba([texto_processado])[0]
            cat_det_ov = m_det_ov.classes_[np.argmax(prob_det_ov)]
            conf_det_ov = max(prob_det_ov) * 100

            st.divider()
            col_of, col_ov = st.columns(2)

            with col_of:
                st.subheader("✅ Modelo Oficial")
                st.info(f"**MACRO:** {cat_mac_of} ({conf_mac_of:.1f}%)")
                st.success(f"**DETALHADO:** {cat_det_of} ({conf_det_of:.1f}%)")
                
                if conf_mac_of < 75 or conf_det_of < 65:
                    st.warning("⚠️ Status: Baixa confiança detectada no roteamento.")
                else:
                    st.write("🟢 Status: Triagem Automatizada.")

            with col_ov:
                st.subheader("⚠️ Modelo Overfit")
                st.info(f"**MACRO:** {cat_mac_ov} ({conf_mac_ov:.1f}%)")
                st.error(f"**DETALHADO:** {cat_det_ov} ({conf_det_ov:.1f}%)")

                if conf_mac_ov < 75 or conf_det_ov < 65:
                    st.warning("⚠️ Status: Baixa confiança detectada no roteamento.")
                else:
                    st.write("🟢 Status: Triagem Automatizada.")

            # log --> Dados para o testing.csv
            novo_log = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'texto_original': texto_usuario,
                'texto_lematizado': texto_processado,
                'oficial_macro': cat_mac_of,
                'oficial_macro_conf': conf_mac_of,
                'oficial_det': cat_det_of,
                'oficial_det_conf': conf_det_of,
                'overfit_macro': cat_mac_ov,
                'overfit_macro_conf': conf_mac_ov,
                'overfit_det': cat_det_ov,
                'overfit_det_conf': conf_det_ov
            }
            
            log_df = pd.DataFrame([novo_log])
            if not os.path.exists('data'): os.makedirs('data')
            
            log_df.to_csv('data/testing.csv', mode='a', index=False, header=not os.path.exists('data/testing.csv'))
            st.toast("Chamado salvo com sucesso, cheque os insights!", icon="💾")




# DASHBOARD DE INSIGHTS


elif st.session_state.page == 'dashboard':
    if st.button("⬅ Voltar para o Simulador"):
        st.session_state.page = 'simulador'
        st.rerun()

    st.title("📈 Insights e Conversão")
    st.markdown(" #### Comparativo visual focado em Data Leakage e Adoção Autônoma do modelo.")

    try:
        df = pd.read_csv('data/testing.csv')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    if df.empty:
        st.warning("Ainda não há dados suficientes no arquivo de logs para gerar métricas.")
    else:
        import plotly.express as px
        import plotly.graph_objects as go
        
        #  MÉTRICAS
        total = len(df)
        df['automacao_ok'] = (df['oficial_macro_conf'] >= 75) & (df['oficial_det_conf'] >= 65)
        taxa_automacao = (df['automacao_ok'].sum() / total) * 100
        
        media_ov = df['overfit_macro_conf'].mean()
        media_of = df['oficial_macro_conf'].mean()
        gap_realismo = media_ov - media_of

        col1, col2, col3 = st.columns(3)
        col1.metric("Volume de Interações", f"{total}")

        with col2:
            st.metric(
                "Triagem 100% Autônoma (oficial)", 
                f"{taxa_automacao:.1f}%", 
                help="Chamados que poderiam ser encaminhados para o setor / agente."
            )

        with col3:
            st.metric(
                "Oficial X Overfit", 
                f"-{gap_realismo:.1f}%",  
                help="O quanto o modelo oficial é mais cauteloso que o overfit."
            )
        
        st.divider()



        #GRÁFICOS ANALÍTICOS


        
        r1_col1, r1_col2 = st.columns(2)
        r2_col1, r2_col2 = st.columns(2)
        r3_col1, r3_col2 = st.columns(2)
        
      
        with r1_col1:
            st.subheader("1. Volumetria por Fila Macro (Oficial)")
            vol_macro = df['oficial_macro'].value_counts().reset_index()
            fig1 = px.bar(vol_macro, x='oficial_macro', y='count', color='oficial_macro', labels={'count':'Chamados', 'oficial_macro': 'Rota'})
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with r1_col2:
            st.subheader("2. Volumetria do Especialista (Oficial)")
            vol_det = df['oficial_det'].value_counts().reset_index()
            vol_det_top = vol_det.head(5)
            fig2 = px.pie(vol_det_top, names='oficial_det', values='count', hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)

        with r2_col1:
            st.subheader("3. Dispersão de Autoconfiança (MACRO)")
            fig3 = go.Figure()
            fig3.add_trace(go.Box(y=df['oficial_macro_conf'], name='Real/Oficial', marker_color='blue'))
            fig3.add_trace(go.Box(y=df['overfit_macro_conf'], name='Overfit/Ilusório', marker_color='red'))
            st.plotly_chart(fig3, use_container_width=True)

        with r2_col2:
            st.subheader("4. Dispersão de Autoconfiança (DETALHADA)")
            fig4 = go.Figure()
            fig4.add_trace(go.Box(y=df['oficial_det_conf'], name='Real/Oficial', marker_color='dodgerblue'))
            fig4.add_trace(go.Box(y=df['overfit_det_conf'], name='Overfit/Ilusório', marker_color='crimson'))
            st.plotly_chart(fig4, use_container_width=True)

        with r3_col1:
            st.subheader("5. Conversão: Revisão Humana vs Autônomo")
            status = df['automacao_ok'].value_counts().reset_index()
            status['Status'] = status['automacao_ok'].map({True: 'Triagem Recomendada', False: 'Revisão Humana'})
            fig5 = px.pie(status, names='Status', values='count', color='Status', color_discrete_map={'Triagem Recomendada':'green', 'Revisão Humana':'orange'})
            st.plotly_chart(fig5, use_container_width=True)

        with r3_col2:
            st.subheader("6. Anomalia do Overfit Linear")
            fig6 = px.scatter(df, x='oficial_macro_conf', y='overfit_macro_conf', 
                              labels={'oficial_macro_conf': 'Confiança Oficial (%)', 'overfit_macro_conf': 'Confiança Overfit (%)'},
                              opacity=0.7)
            fig6.add_shape(type='line', x0=0, y0=0, x1=100, y1=100, line=dict(color='gray', dash='dash'))
            st.plotly_chart(fig6, use_container_width=True)

        st.divider()
        st.subheader("Tabela de Dados Extraídos (Log Oficial)")
        st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True)

# --- RODAPÉ ---
st.divider()
st.caption("Igor Amaral - Simulação para XPTO Data Solutions / Kurier.")