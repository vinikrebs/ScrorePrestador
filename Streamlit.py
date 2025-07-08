import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
from streamlit_option_menu import option_menu
import unicodedata
import re
import io

# --- Configurações Iniciais do Streamlit ---
st.set_page_config(
    page_title="Monitoramento Inteligente da Rede de Prestadores A24h",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="favicon.ico"
)

# --- Gerenciamento do Estado da Sessão ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# --- Constantes ---
MIN_ATTENDANCES_FOR_RANKING = 1 # Contagem mínima de atendimentos para rankings de prestador/segmento/seguradora
MIN_ATTENDANCES_FOR_CITY_ANALYSIS = 10 # Atendimentos mínimos padrão para análise de cidade em Capilaridade
ALL_OPTION = "TODOS" # Constante para a opção "TODOS" nos filtros
ATENDIMENTO_FILE_PATH = 'https://github.com/vinikrebs/ScrorePrestador/raw/main/processed_atendimentos.parquet'
PROCESSED_CAPILARIDADE_FILE_PATH = 'https://github.com/vinikrebs/ScrorePrestador/raw/main/processed_capilaridade_cidade.parquet'
FINANCIAL_KPI_FILE = 'https://github.com/vinikrebs/ScrorePrestador/raw/main/processed_financeiro.parquet'
NPS_FILE_PATH = 'https://github.com/vinikrebs/ScrorePrestador/raw/main/processed_capilaridade_cidade.parquet'

# --- Função da Página de Login ---
def login_page():
    """Renderiza a página de login."""
    col_logo_left, col_logo_center, col_logo_right = st.columns([1.5, 3, 1.5])
    with col_logo_center:
        # Caminho da imagem no servidor em produção pode ser diferente
        # Centralizando a imagem usando markdown e HTML
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image("C:\\Temp\\logo.png", use_container_width=False, width=1600)
        st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<h1 style='text-align: center; color: #333333;'>Acesso ao Dashboard Score do Prestador A24h</h1>", unsafe_allow_html=True)
    st.empty() # Adiciona espaço vertical
    st.empty() # Adiciona espaço vertical

    with st.form("login_form"):
        st.markdown("<p style='text-align: center;'>Por favor, insira suas credenciais.</p>", unsafe_allow_html=True)
        username = st.text_input("Usuário", key="username_input")
        password = st.text_input("Senha", type="password", key="password_input")

        col_login_btn, col_forgot_btn = st.columns(2)
        with col_login_btn:
            login_button = st.form_submit_button("Entrar")
        with col_forgot_btn:
            # Botão "Criar uma nova conta"
            create_account_button = st.form_submit_button("Criar uma nova conta")

        if login_button:
            if username == "maxpar" and password == "Max!Q@W":
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Login realizado com sucesso!")
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos. Tente novamente.")

        if create_account_button:
            # Mensagem para "Criar uma nova conta"
            st.info("Por favor, envie um e-mail para vinicius.krebs@autoglass.com.br para criar uma nova conta.")

# --- Função de Carregamento e Preparação de Dados (com cache para performance) ---
@st.cache_data
def load_and_prepare_data(atendimentos_file_path, nps_file_path):
    """
    Carrega e pré-processa dados, realizando limpeza e conversão de tipo.
    Integra dados de NPS.
    """
    pd.set_option('future.no_silent_downcasting', True)

    try:
        # Alterado para ler arquivo Parquet
        df = pd.read_parquet(atendimentos_file_path)

    except FileNotFoundError:
        st.error(f"Erro: Arquivo '{atendimentos_file_path}' não encontrado. Verifique o caminho.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Parquet de atendimentos: {e}. Verifique se o arquivo está no formato correto.")
        st.stop()

    def normalize_string(text):
        if pd.isna(text):
            return ''
        text = str(text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        return text.lower().replace(' ', '_').replace('.', '')

    # As colunas já devem estar normalizadas se vierem do data_processor
    # df.columns = [normalize_string(col) for col in df.columns] # Comentar ou remover se já normalizado

    # A lógica de data e preenchimento de NaNs deve ser feita no data_processor
    # Apenas garantir que as colunas importantes estão no tipo correto
    if 'data_abertura_atendimento' in df.columns:
        df['data_abertura_atendimento'] = pd.to_datetime(df['data_abertura_atendimento'], errors='coerce')
    else:
        st.error("Coluna 'data_abertura_atendimento' não encontrada no DataFrame processado.")
        st.stop()

    df_final = df.dropna(subset=['data_abertura_atendimento']).copy()

    for col in ['segmento', 'seguradora', 'uf', 'municipio', 'nome_do_prestador', 'protocolo_atendimento']:
        if col in df_final.columns:
            df_final[col] = df_final[col].astype(str).fillna('NAO INFORMADO').str.upper()
            if col not in ['protocolo_atendimento']:
                df_final[col] = df_final[col].astype('category')

    if 'gerou_reembolso' in df_final.columns:
        df_final['gerou_reembolso'] = df_final['gerou_reembolso'].astype(bool)
    if 'val_reembolso' in df_final.columns:
        df_final['val_reembolso'] = df_final['val_reembolso'].astype(float)
    if 'is_reembolso' in df_final.columns:
        df_final['is_reembolso'] = df_final['is_reembolso'].astype(bool)
    if 'is_intermediacao' in df_final.columns:
        df_final['is_intermediacao'] = df_final['is_intermediacao'].astype(bool)
    if 'tempo_chegada_min' in df_final.columns:
        df_final['tempo_chegada_min'] = df_final['tempo_chegada_min'].astype(float)
    if 'val_total_items' in df_final.columns:
        df_final['val_total_items'] = df_final['val_total_items'].astype(float)


    # --- CORREÇÃO: Integração de Dados de NPS (agora lendo parquet) ---
    try:
        # Alterado para ler arquivo Parquet
        df_nps = pd.read_parquet(nps_file_path)

        # As colunas já devem vir processadas do data_processor.py
        # Apenas garantir a existência das colunas para a fusão.
        if 'nps_score_calculado' not in df_nps.columns:
            st.warning(f"Aviso: Coluna 'nps_score_calculado' não encontrada em '{nps_file_path}'. A análise de qualidade será limitada.")
            df_final['nps_score_calculado'] = np.nan
            return df_final


    except FileNotFoundError:
    #    st.warning(f"Aviso: Arquivo NPS '{nps_file_path}' não encontrado. A análise de qualidade será limitada.")
        df_final['nps_score_calculado'] = np.nan
    #except Exception as e:
    #    st.warning(f"Aviso ao ler o arquivo NPS Parquet: {e}. A análise de qualidade pode estar incompleta.")
        df_final['nps_score_calculado'] = np.nan


    # Criação de novas colunas de tempo (se ainda não existirem do processamento)
    if 'mes_ano_abertura' not in df_final.columns:
        df_final['mes_ano_abertura'] = df_final['data_abertura_atendimento'].dt.to_period('M').astype(str)
    if 'dia_da_semana' not in df_final.columns:
        df_final['dia_da_semana'] = df_final['data_abertura_atendimento'].dt.day_name(locale='pt_BR')
    if 'hora_do_dia' not in df_final.columns:
        df_final['hora_do_dia'] = df_final['data_abertura_atendimento'].dt.hour
    if 'periodo_do_dia' not in df_final.columns:
        df_final['periodo_do_dia'] = df_final['hora_do_dia'].apply(
            lambda x: 'Manhã' if 6 <= x < 12 else ('Tarde' if 12 <= x < 18 else ('Noite' if 18 <= x < 24 else 'Madrugada'))
        )

    return df_final



df_atendimentos = load_and_prepare_data(ATENDIMENTO_FILE_PATH, NPS_FILE_PATH)

if df_atendimentos.empty:
    st.error("Nenhum dado válido disponível após o pré-processamento. Verifique seus arquivos Parquet e o formato das colunas.")
    st.stop()

# --- CSS para Filtros da Barra Lateral e Estilo do DataFrame ---
def inject_css():
    st.markdown("""
    <style>
    /* Estilo geral da barra lateral */
    div[data-testid="stSidebar"] {
        font-family: "Inter", sans-serif;
    }
    div[data-testid="stSidebar"] h2 {
        color: #2021D4;
        font-weight: bold;
        margin-top: 0.5rem !important; /* Reduce top margin */
        margin-bottom: 0.5rem !important; /* Reduce bottom margin */
    }
    div[data-testid="stSidebar"] label {
        color: #333333;
        font-weight: normal;
    }

    /* Ajustar o gap entre elementos dentro da barra lateral */
    div[data-testid="stSidebar"] .stVerticalBlock {
        gap: 0.5rem; /* Adjust this value to control spacing */
    }

    /* Bordas e cores do MultiSelect, DateInput, NumberInput */
    div[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"],
    div[data-testid="stSidebar"] .stDateInput input[type="text"],
    div[data-testid="stSidebar"] .stNumberInput input[type="number"] {
        border: 1px solid #2021D4 !important;
        border-radius: 5px;
        color: #333333;
    }

    /* Tags do MultiSelect */
    div[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"] {
        background-color: #2021D4 !important;
        color: white !important;
        border-radius: 5px;
    }

    /* Estado de foco para inputs */
    div[data-testid="stSidebar"] input:focus {
        border-color: #2021D1 !important;
        box-shadow: 0 0 0 0.2rem rgba(32, 33, 212, 0.25) !important;
    }

    /* Estilo do Popover (dropdowns) */
    div[data-baseweb="popover"] > div > div {
        background-color: #FFFFFF !important;
        border: 1px solid #2021D4 !important;
        border-radius: 5px;
    }
    div[data-baseweb="popover"] ul > li {
        color: #333333 !important;
    }
    div[data-baseweb="popover"] ul > li:hover {
        background-color: #E6F0F8 !important;
        color: #333333 !important;
    }

    /* Ajustar largura da barra lateral se necessário (exemplo) */
    .st-emotion-cache-1uj76pr {
        flex: 3;
    }

    /* Specific for the image in the sidebar */
    div[data-testid="stSidebar"] .stImage {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
    }

    /* Specific for markdown separators (---) to reduce their margins */
    div[data-testid="stSidebar"] .st-emotion-cache-1cypf8t { /* This targets the divider line */
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Estilo para o expander de "dica" */
    div[data-testid="stExpander"] > div[role="button"] {
        background-color: #E6F0F8; /* Light blue background */
        border-left: 5px solid #2021D4; /* Blue left border */
        color: #2021D4; /* Dark blue text */
        font-weight: bold;
        padding: 10px 15px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Soft shadow for a button-like feel */
    }
    div[data-testid="stExpander"] > div[role="button"]:hover {
        background-color: #DDEBF7; /* Slightly darker on hover */
    }
    div[data-testid="stExpander"] .st-emotion-cache-p5mxxl { /* This targets the expander caret icon */
        color: #2021D4 !important;
    }

    /* Estilo Geral do DataFrame para melhor legibilidade */
    .dataframe {
        font-family: "Inter", sans-serif;
        font-size: 14px;
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe th {
        background-color: #2021D4;
        color: white;
        padding: 8px;
        text-align: left;
        border-bottom: 2px solid #ddd;
    }
    .dataframe td {
        padding: 8px;
        border-bottom: 1px solid #eee;
    }
    .dataframe tr:hover {
        background-color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Função Geral de Aplicação de Filtros ---
def apply_filters(df, selected_segments, selected_insurers, selected_states, selected_municipios, start_date, end_date):
    """Aplica filtros comuns ao DataFrame."""
    df_filtered = df[
        (df['segmento'].isin(selected_segments)) &
        (df['seguradora'].isin(selected_insurers)) &
        (df['uf'].isin(selected_states)) &
        (df['municipio'].isin(selected_municipios)) &
        (df['data_abertura_atendimento'].dt.date >= start_date) &
        (df['data_abertura_atendimento'].dt.date <= end_date)
    ].copy()
    return df_filtered

# --- Funções para Páginas (Pilares) ---
def page_informacao():
    """Renderiza a página de informações gerais do dashboard."""
    st.markdown("""
    # Monitoramento Inteligente da Rede de Prestadores A24h

    **Bem-vindo ao Dashboard de Monitoramento Inteligente da Rede de Prestadores A24h!** Esta plataforma foi desenhada para transformar dados complexos em *insights acionáveis*, permitindo decisões ágeis e estratégicas para otimizar performance, reduzir custos e garantir a melhor experiência para nossos clientes.

    👉 **Explore os pilares abaixo** e descubra diferentes perspectivas da nossa rede de parceiros:

    ---
    """)

    st.markdown("## 📍 Pilar Capilaridade")
    st.markdown("""
    Monitora a **distribuição** e a **disponibilidade** dos prestadores em relação à demanda da região.

    ✅ **Por que é importante?** Ajuda a identificar áreas descobertas, equilibrar recursos e evitar sobrecarga em determinados prestadores.
    """)

    st.subheader("Indicadores de Capilaridade")
    df_capilaridade_kpis = pd.DataFrame({
        "KPI": ["Índice de Capilaridade", "Vazio Assistencial", "Intermediações por Atendimento"],
        "O que mede": [
            "Equilíbrio entre rede e demanda",
            "Municípios atendidos sem prestadores disponíveis",
            "Complexidade operacional"
        ],
        "Fórmula": [
            "Nº de Prestadores / Nº de Atendimentos",
            "Nº de Atendimentos / Nº de Prestadores",
            "Nº de Intermediações / Nº de Atendimentos"
        ]
    })
    st.dataframe(df_capilaridade_kpis, hide_index=True)

    st.markdown("---")

    st.markdown("## 💰 Pilar Financeiro")
    st.markdown("""
    Analisa a **eficiência econômica** da rede e o impacto no custo médio por serviço (CMS).

    ✅ **Por que é importante?** Permite controlar custos, identificar oportunidades de economia e evitar distorções regionais.
    """)

    st.subheader("Indicadores Financeiros")
    df_financeiro_kpis = pd.DataFrame({
        "KPI": ["Diferença CMS", "Proporção de Reembolso"],
        "O que mede": [
            "Custo médio por serviço em relação à média estadual e segmentada",
            "Dependência de reembolsos na operação"
        ],
        "Fórmula": [
            "(CMS Cidade Segmento - CMS Estadual Segmento) / CMS Estadual Segmento",
            "Nº de Reembolsos / Total de Atendimentos"
        ]
    })
    st.dataframe(df_financeiro_kpis, hide_index=True)

    st.markdown("---")

    st.markdown("## ⟳ Pilar Frequência de Uso")
    st.markdown("""
    Acompanha o quanto os serviços estão sendo utilizados por apólice em cada região.

    ✅ **Por que é importante?** Ajuda a identificar padrões de uso, prever demandas futuras e gerenciar melhor a capacidade da rede.
    """)

    st.subheader("Indicadores de Frequência de Uso")
    df_frequencia_kpis = pd.DataFrame({
        "KPI": ["Frequência por Apólice"],
        "O que mede": ["Média de atendimentos por cliente/apólice"],
        "Fórmula": ["Nº de Atendimentos / Nº de Apólices"]
    })
    st.dataframe(df_frequencia_kpis, hide_index=True)

    st.markdown("---")

    st.markdown("## ✨ Pilar Qualidade")
    st.markdown("""
    Avalia a **satisfação do cliente** e a **qualidade do serviço entregue**.

    ✅ **Por que é importante?** Monitora o nível de confiança do cliente e identifica oportunidades de melhoria.
    """)

    st.subheader("Indicadores de Qualidade")
    df_qualidade_kpis = pd.DataFrame({
        "KPI": ["NPS", "TMC"],
        "O que mede": [
            "Índice de satisfação do cliente",
            "Tempo médio de chegada ao local do sinistro"
        ],
        "Fórmula": [
            "% Promotores - % Detratores",
            "Média de tempo de chegada ao segurado"
        ]
    })
    st.dataframe(df_qualidade_kpis, hide_index=True)

    st.markdown("""
    ### 🛠️ Tecnologias Utilizadas
    * **Python**: Linguagem de programação principal.
    * **Streamlit**: Framework para construção da interface web interativa.
    * **Pandas**: Biblioteca para manipulação e análise de dados.
    * **NumPy**: Biblioteca para operações numéricas.
    * **Plotly Express**: Biblioteca para criação de gráficos interativos.
    * **Streamlit-Option-Menu**: Componente para a barra lateral de navegação.
    * **Unicodedata, re**: Módulos para normalização e limpeza de strings.
    * **Xlsxwriter**: Engine para exportação de dados para arquivos Excel.

    ### 💡 Contribuição
    Este dashboard foi desenvolvido por **Vinicius Krebs** como parte do projeto de **Redução de Custos A24H e Aumento de Capilaridade da Rede**.
    """)
    st.markdown("---")
    st.info("Utilize o menu na barra lateral para navegar por cada pilar. Cada seção oferece filtros detalhados para uma análise personalizada.")


def display_capilaridade_kpis(df_filtered, df_agregado_cidade_com_indice=None):
    """Exibe os principais indicadores de desempenho para Capilaridade."""
    st.markdown("---")
    st.header("KPIs Gerais de Capilaridade")
    total_servicos = len(df_filtered)
    total_prestadores_unicos = df_filtered['nome_do_prestador'].nunique()
    total_cidades_atendidas = df_filtered['municipio'].nunique()
    media_tempo_chegada = df_filtered['tempo_chegada_min'].mean()
    pct_reembolso = (df_filtered['is_reembolso'].sum() / total_servicos) * 100 if total_servicos > 0 else 0
    pct_intermediacao = (df_filtered['is_intermediacao'].sum() / total_servicos) * 100 if total_servicos > 0 else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total de Serviços", f"{total_servicos:,}".replace(',', '.'), help="Número total de atendimentos registrados com os filtros aplicados.")
    col2.metric("Prestadores Únicos", f"{total_prestadores_unicos:,}".replace(',', '.'), help="Número de prestadores distintos que realizaram atendimentos com os filtros aplicados.")
    col3.metric("Cidades Atendidas", f"{total_cidades_atendidas:,}".replace(',', '.'), help="Número de municípios onde houve atendimentos com os filtros aplicados.")
    col4.metric("TMC (min)", f"{media_tempo_chegada:.0f} min" if not pd.isna(media_tempo_chegada) else "N/A", help="Tempo Médio de Chegada do prestador ao local do serviço, em minutos.")
    col5.metric("Perc. Reembolso", f"{pct_reembolso:.2f}%", help="Percentual de serviços que geraram algum tipo de reembolso, indicando falha na cobertura direta ou preferência do cliente.")
    col6.metric("Perc. Intermediação", f"{pct_intermediacao:.2f}%", help="Percentual de serviços que foram realizados por meio de intermediação, e não por prestadores diretos da rede.")

    # --- GRÁFICO DE CAPILARIDADE ---
    if df_agregado_cidade_com_indice is not None and not df_agregado_cidade_com_indice.empty:
        st.markdown("---")
        st.header("Dispersão de Capilaridade por Cidade")
        st.info("Visualize a relação entre o número de serviços e a quantidade de prestadores. Círculos maiores indicam maior volume de atendimentos. As cores representam o status de capilaridade da cidade.")

        category_order = ['Carência Assistencial', 'Capilaridade Regular', 'Boa Capilaridade']
        status_colors = {
            'Carência Assistencial': '#EF5350',  # Red
            'Capilaridade Regular': '#FFCA28',    # Amber
            'Boa Capilaridade': '#66BB6A'        # Green
        }

        # Create the scatter plot
        fig_capilaridade = px.scatter(
            df_agregado_cidade_com_indice,
            x='num_servicos',
            y='num_prestadores',
            color='status_capilaridade',
            size='num_servicos', # Size of markers based on number of services
            hover_name='municipio',
            hover_data={
                'uf': True,
                'num_servicos': ':.0f',
                'num_prestadores': ':.0f',
                'indice_capilaridade': ':.2f',
                'status_capilaridade': True,
                'pct_reembolso': ':.2f',
                'pct_intermediacao': ':.2f',
                'media_tempo_chegada': ':.0f'
            },
            title='Capilaridade: Serviços vs. Prestadores por Cidade e Status',
            labels={
                'num_servicos': 'Número de Serviços (Atendimentos)',
                'num_prestadores': 'Número de Prestadores',
                'status_capilaridade': 'Status de Capilaridade'
            },
            color_discrete_map=status_colors,
            category_orders={'status_capilaridade': category_order},
            height=600,
            log_x=True, # Apply log scale to x-axis
            )

        fig_capilaridade.update_layout(
            xaxis_title="Número de Serviços (Atendimentos)",
            yaxis_title="Número de Prestadores",
            legend_title="Status de Capilaridade",
            hovermode="closest",
            yaxis=dict(showgrid=False)  # Remove y-axis grid lines
        )
        
        st.plotly_chart(fig_capilaridade, use_container_width=True)


def calculate_capilaridade_index(df_agregado_cidade):
    """Calcula o Índice de Capilaridade e seu status."""
    if df_agregado_cidade.empty:
        return pd.DataFrame()

    # Handle potential division by zero for max()
    max_servicos = df_agregado_cidade['num_servicos'].max()
    df_agregado_cidade['norm_atendimentos'] = df_agregado_cidade['num_servicos'] / max_servicos if max_servicos > 0 else 0
    
    max_prestadores = df_agregado_cidade['num_prestadores'].max()
    df_agregado_cidade['norm_prestadores'] = df_agregado_cidade['num_prestadores'] / max_prestadores if max_prestadores > 0 else 0

    max_pct_reembolso = df_agregado_cidade['pct_reembolso'].max()
    df_agregado_cidade['norm_pct_reembolso'] = df_agregado_cidade['pct_reembolso'] / max_pct_reembolso if max_pct_reembolso > 0 else 0
    df_agregado_cidade['contrib_reembolso'] = (1 - df_agregado_cidade['norm_pct_reembolso']) if max_pct_reembolso > 0 else 1

    max_pct_intermediacao = df_agregado_cidade['pct_intermediacao'].max()
    df_agregado_cidade['norm_pct_intermediacao'] = df_agregado_cidade['pct_intermediacao'] / max_pct_intermediacao if max_pct_intermediacao > 0 else 0
    df_agregado_cidade['contrib_intermediacao'] = (1 - df_agregado_cidade['norm_pct_intermediacao']) if max_pct_intermediacao > 0 else 1

    max_tempo_chegada = df_agregado_cidade['media_tempo_chegada'].max()
    df_agregado_cidade['norm_tempo_chegada'] = df_agregado_cidade['media_tempo_chegada'] / max_tempo_chegada if max_tempo_chegada > 0 else 0
    df_agregado_cidade['contrib_tempo_chegada'] = (1 - df_agregado_cidade['norm_tempo_chegada']) if max_tempo_chegada > 0 else 1

    df_agregado_cidade['indice_capilaridade'] = (
        df_agregado_cidade['norm_atendimentos'] * 0.3 +
        df_agregado_cidade['norm_prestadores'] * 0.3 +
        df_agregado_cidade['contrib_reembolso'] * 0.2 +
        df_agregado_cidade['contrib_intermediacao'] * 0.1 +
        df_agregado_cidade['contrib_tempo_chegada'] * 0.1
    )

    if not df_agregado_cidade['indice_capilaridade'].empty and df_agregado_cidade['indice_capilaridade'].nunique() > 0:
        unique_indices = df_agregado_cidade['indice_capilaridade'].nunique()
        if unique_indices <= 1:
            df_agregado_cidade['status_capilaridade'] = 'Capilaridade Regular'
        else:
            q1 = df_agregado_cidade['indice_capilaridade'].quantile(0.25)
            q3 = df_agregado_cidade['indice_capilaridade'].quantile(0.75)
            
            # Ensure bins are unique and correctly ordered
            # Add small epsilon to max to ensure all values fall into a bin
            bins = sorted(list(set([df_agregado_cidade['indice_capilaridade'].min() - 0.001, q1, q3, df_agregado_cidade['indice_capilaridade'].max() + 0.001])))
            
            labels_map = {
                2: ['Carência Assistencial', 'Boa Capilaridade'],
                3: ['Carência Assistencial', 'Capilaridade Regular', 'Boa Capilaridade']
            }
            # Use the most appropriate labels based on the number of unique bins created
            labels = labels_map.get(len(bins) - 1, ['Capilaridade Regular']) 

            df_agregado_cidade['status_capilaridade'] = pd.cut(
                df_agregado_cidade['indice_capilaridade'],
                bins=bins,
                labels=labels,
                right=True,
                duplicates='drop'
            )
            # Fill any NaN created by pd.cut (e.g., if a value falls exactly on a boundary not covered by 'right=True')
            df_agregado_cidade['status_capilaridade'] = df_agregado_cidade['status_capilaridade'].fillna('Capilaridade Regular')
            
            # Ensure the categories are set correctly for plotting order
            df_agregado_cidade['status_capilaridade'] = pd.Categorical(
                df_agregado_cidade['status_capilaridade'],
                categories=['Carência Assistencial', 'Capilaridade Regular', 'Boa Capilaridade'],
                ordered=True
            )
    else:
        df_agregado_cidade['status_capilaridade'] = 'N/A'
    return df_agregado_cidade


def get_sugestao_acao(row, df_agregado_cidade, min_atendimentos_cidade):
    """Gera sugestões de ação com base no desempenho da cidade."""
    sugestoes = set()
    if row['status_capilaridade'] == 'Carência Assistencial':
        sugestoes.add("Recrutamento urgente de prestadores. Analisar concorrência local.")
    
    # Check for empty prestadores only if services > 0
    if row['num_servicos'] > 0 and row['num_prestadores'] == 0:
        sugestoes.add("Ausência de prestadores. Foco total em parceria local.")
    
    # Only provide suggestions for cities above min_atendimentos_cidade
    if row['num_servicos'] >= min_atendimentos_cidade:
        # Calculate percentiles for each metric excluding current row to avoid self-influence,
        # or use overall percentiles if data is large enough. For simplicity here,
        # we'll use overall for now, but in a real scenario, robust percentiles are better.
        
        # Ensure there are enough unique values to calculate quantiles
        if df_agregado_cidade['pct_reembolso'].nunique() > 1:
            pct_reembolso_80 = df_agregado_cidade['pct_reembolso'].quantile(0.80)
            if row['pct_reembolso'] > pct_reembolso_80:
                sugestoes.add("Alto % de reembolso. Investigar causas de insatisfação ou deficiência de prestadores.")
        
        if df_agregado_cidade['pct_intermediacao'].nunique() > 1:
            pct_intermediacao_80 = df_agregado_cidade['pct_intermediacao'].quantile(0.80)
            if row['pct_intermediacao'] > pct_intermediacao_80:
                sugestoes.add("Alto % de intermediação. Otimizar processos de acionamento ou recrutar prestadores diretos.")
        
        if df_agregado_cidade['media_tempo_chegada'].nunique() > 1:
            tempo_chegada_80 = df_agregado_cidade['media_tempo_chegada'].quantile(0.80)
            if row['media_tempo_chegada'] > tempo_chegada_80:
                sugestoes.add("Alto tempo de chegada. Otimizar rotas ou aumentar a densidade de prestadores próximos.")

    return " | ".join(sorted(list(sugestoes))) if sugestoes else "Nenhuma sugestão específica."

def display_specific_problem_rankings(df_agregado_cidade):
    """Exibe os rankings para cidades com problemas específicos, como alto reembolso ou intermediação."""
    st.markdown("---")
    st.header("Classificações de Problemas Específicos")
    st.markdown("Cidades com os maiores desafios em reembolso e intermediação.")
    
    col_reembolso_rank, col_intermediacao_rank = st.columns(2)
    
    with col_reembolso_rank:
        st.subheader("Cidades por % de Reembolso")
        df_reembolso_rank = df_agregado_cidade.sort_values('pct_reembolso', ascending=False).head(10).copy()
        
        df_reembolso_rank_display = df_reembolso_rank.rename(columns={
            'municipio': 'Cidade',
            'uf': 'UF',
            'num_servicos': 'Qtd. Serviços',
            'pct_reembolso': '% Reembolso',
            'total_valor_servicos': 'Valor Total'
        })
        st.dataframe(
            df_reembolso_rank_display[['Cidade', 'UF', 'Qtd. Serviços', '% Reembolso', 'Valor Total']].style.format({
                'Qtd. Serviços': '{:,.0f}',
                '% Reembolso': '{:.2f}%',
                'Valor Total': 'R$ {:,.2f}'
            }),
            use_container_width=True
        )
        st.markdown("**Sugestão:** Busca de novos prestadores para reduzir a taxa de reembolso e diminuir o CMS.")
    
    with col_intermediacao_rank:
        st.subheader("Cidades por % de Intermediação")
        df_intermediacao_rank = df_agregado_cidade.sort_values('pct_intermediacao', ascending=False).head(10).copy()
        
        df_intermediacao_rank_display = df_intermediacao_rank.rename(columns={
            'municipio': 'Cidade',
            'uf': 'UF',
            'num_servicos': 'Qtd. Serviços',
            'pct_intermediacao': '% Intermediação', # This is where it's correctly defined with "ção"
            'total_valor_servicos': 'Valor Total'
        })
        st.dataframe(
            df_intermediacao_rank_display[['Cidade', 'UF', 'Qtd. Serviços', '% Intermediação', 'Valor Total']].style.format({ # CORRECTED HERE: '% Intermediação'
                'Qtd. Serviços': '{:,.0f}',
                '% Intermediação': '{:.2f}%', # And here for consistency in formatting
                'Valor Total': 'R$ {:,.2f}'
            }),
            use_container_width=True
        )
        st.markdown("**Sugestão:** Otimizar processos de acionamento ou recrutar prestadores diretos para reduzir intermediações.")

def page_capilaridade(df, min_atendimentos_cidade):
    """Renderiza a página de Capilaridade."""
    st.title("📍 Capilaridade da Rede")
    st.markdown("Esta seção oferece uma visão detalhada da distribuição e cobertura dos nossos prestadores, identificando áreas de alta demanda e oportunidades de expansão.")

    df_agregado_cidade = df.groupby(['uf', 'municipio'], observed=True).agg(
        num_servicos=('protocolo_atendimento', 'nunique'),
        num_prestadores=('nome_do_prestador', 'nunique'),
        num_reembolsos=('is_reembolso', lambda x: x.sum()),
        num_intermediacoes=('is_intermediacao', lambda x: x.sum()),
        media_tempo_chegada=('tempo_chegada_min', 'mean'),
        total_valor_servicos=('val_total_items', 'sum')
    ).reset_index()

    df_agregado_cidade['pct_reembolso'] = np.where(
        df_agregado_cidade['num_servicos'] > 0,
        (df_agregado_cidade['num_reembolsos'] / df_agregado_cidade['num_servicos']) * 100,
        0
    )
    df_agregado_cidade['pct_intermediacao'] = np.where(
        df_agregado_cidade['num_servicos'] > 0,
        (df_agregado_cidade['num_intermediacoes'] / df_agregado_cidade['num_servicos']) * 100,
        0
    )
    df_agregado_cidade['media_tempo_chegada'] = df_agregado_cidade['media_tempo_chegada'].fillna(0)

    # NOVO CÁLCULO: Qtd. Serviços Não Atendidos
    df_agregado_cidade['num_servicos_nao_atendidos'] = df_agregado_cidade['num_servicos'] - \
                                                        df_agregado_cidade['num_reembolsos'] - \
                                                        df_agregado_cidade['num_intermediacoes']
    df_agregado_cidade['num_servicos_nao_atendidos'] = df_agregado_cidade['num_servicos_nao_atendidos'].clip(lower=0)

    df_agregado_cidade_filtrado = df_agregado_cidade[df_agregado_cidade['num_servicos'] >= min_atendimentos_cidade].copy()
    df_agregado_cidade_com_indice = calculate_capilaridade_index(df_agregado_cidade_filtrado)

    display_capilaridade_kpis(df, df_agregado_cidade_com_indice)

    if not df_agregado_cidade_com_indice.empty:
        df_agregado_cidade_com_indice['sugestao_acao'] = df_agregado_cidade_com_indice.apply(
            lambda row: get_sugestao_acao(row, df_agregado_cidade_com_indice, min_atendimentos_cidade), axis=1
        )

        st.markdown("---")
        st.subheader("Cidades com Necessidade de Atenção na Capilaridade")
        st.info("Foque nestas cidades para otimizar a cobertura da sua rede.")
        
        df_offenders = df_agregado_cidade_com_indice[
            (df_agregado_cidade_com_indice['status_capilaridade'] == 'Carência Assistencial') |
            (df_agregado_cidade_com_indice['pct_reembolso'] > df_agregado_cidade_com_indice['pct_reembolso'].quantile(0.75)) |
            (df_agregado_cidade_com_indice['pct_intermediacao'] > df_agregado_cidade_com_indice['pct_intermediacao'].quantile(0.75)) |
            (df_agregado_cidade_com_indice['media_tempo_chegada'] > df_agregado_cidade_com_indice['media_tempo_chegada'].quantile(0.75))
        ].sort_values('indice_capilaridade', ascending=True)

        if not df_offenders.empty:
            st.dataframe(
                df_offenders.rename(columns={
                    'municipio': 'Cidade',
                    'uf': 'UF',
                    'num_servicos': 'Qtd. Serviços',
                    'num_prestadores': 'Qtd. Prestadores',
                    'num_servicos_nao_atendidos': 'Qtd. Não Atendidos',
                    'pct_reembolso': '% Reembolso',
                    'pct_intermediacao': '% Intermediação',
                    'media_tempo_chegada': 'TMC Médio (min)',
                    'indice_capilaridade': 'Índice Capilaridade',
                    'status_capilaridade': 'Status Capilaridade',
                    'sugestao_acao': 'Sugestão de Ação'
                })[[
                    'Cidade', 
                    'UF', 
                    'Qtd. Serviços', 
                    'Qtd. Não Atendidos', # Ordem da coluna na exibição
                    'Qtd. Prestadores', 
                    '% Reembolso', 
                    '% Intermediação', 
                    'TMC Médio (min)', 
                    'Índice Capilaridade', 
                    'Status Capilaridade', 
                    'Sugestão de Ação'
                ]].style.format({
                        'Qtd. Serviços': '{:,.0f}',
                        'Qtd. Serviços Não Atendidos': '{:,.0f}', # Formatação
                        'Qtd. Prestadores': '{:,.0f}',
                        '% Reembolso': '{:.2f}%',
                        '% Intermediação': '{:.2f}%',
                        'TMC Médio (min)': '{:.0f}',
                        'Índice Capilaridade': '{:.2f}'
                    }),
                use_container_width=True
            )
          

            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl2:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_offenders.to_excel(writer, index=False, sheet_name='Cidades Ofensoras')
                output.seek(0)
                st.download_button(
                    label="Baixar Cidades Ofensoras (XLSX)",
                    data=output,
                    file_name='cidades_ofensoras_capilaridade.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    help="Baixa os dados das cidades identificadas como principais ofensoras na capilaridade em formato XLSX."
                )
            with col_dl1:
                output_full = io.BytesIO()
                with pd.ExcelWriter(output_full, engine='xlsxwriter') as writer:
                    df_agregado_cidade_com_indice.to_excel(writer, index=False, sheet_name='Capilaridade Completa')
                output_full.seek(0)
                st.download_button(
                    label="Baixar Todos os Dados de Capilaridade (XLSX)",
                    data=output_full,
                    file_name='capilaridade_por_cidade_completo.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    help="Baixa todos os dados agregados de capilaridade por cidade com os filtros aplicados e sugestões de ação em formato XLSX."
                )
        else:
            st.info("Nenhuma cidade identificada como 'ofensora' com base nos critérios atuais. Excelente!")

        display_specific_problem_rankings(df_agregado_cidade_com_indice)

    else:
        st.info("Nenhum dado de capilaridade disponível com os filtros e limites selecionados.")

    # Expander de cálculo do Índice de Capilaridade (mantido para contexto)
    with st.expander("💡 Como é calculado o Índice de Capilaridade?"):
        st.markdown(r"""
        O **Índice de Capilaridade** é um score composto que avalia a eficiência e a cobertura da rede em cada cidade, combinando múltiplos fatores:

        * **Normalização dos Dados:** Todos os componentes são normalizados entre 0 e 1 (ou 0 a 100) para garantir que tenham o mesmo peso e não sejam dominados por valores absolutos.
        * **Componentes e Pesos:**
            * **Volume de Serviços (30%):** Cidades com maior volume de serviços contribuem positivamente, pois representam demanda onde a capilaridade é crítica.
            * **Número de Prestadores (30%):** Uma maior quantidade de prestadores únicos em uma cidade indica melhor oferta de serviços.
            * **Reembolso (20%):** Menor percentual de serviços que resultam em reembolso indica que a rede está mais eficaz em resolver o problema diretamente. (Peso inverso: quanto menor o reembolso, maior a contribuição positiva).
            * **Intermediação (10%):** Menor percentual de serviços que necessitam de intermediação (ou seja, resolvidos diretamente pela rede própria) indica maior eficiência. (Peso inverso).
            * **Tempo Médio de Chegada (10%):** Tempos de chegada menores indicam agilidade e proximidade dos prestadores. (Peso inverso).

        **Fórmula Simplificada:**
        $$ \text{Índice Capilaridade} = \left( \text{Serviços Normalizados} \times 0.3 \right) + \left( \text{Prestadores Normalizados} \times 0.3 \right) + \left( (1 - \text{Reembolso Normalizado}) \times 0.2 \right) + \left( (1 - \text{Intermediação Normalizada}) \times 0.1 \right) + \left( (1 - \text{TMC Normalizado}) \times 0.1 \right) $$

        **Classificação do Status de Capilaridade:**
        O status é determinado pelos quartis do Índice de Capilaridade:
        * **Carência Assistencial:** Cidades no quartil inferior (piores 25%).
        * **Capilaridade Regular:** Cidades entre o primeiro e o terceiro quartil (25% a 75%).
        * **Boa Capilaridade:** Cidades no quartil superior (melhores 25%).
        """)

    # Expander de sugestões de ação (mantido para contexto)
    with st.expander("💡 Como são geradas as Sugestões de Ação?"):
        st.markdown("""
        As sugestões de ação são geradas dinamicamente para cada município com base em suas características e desvios em relação à média:
        * **Carência Assistencial:** Se o status de capilaridade for 'Carência Assistencial', a sugestão é 'Recrutamento urgente de prestadores. Analisar concorrência local.'
        * **Ausência de Prestadores:** Se a cidade tem atendimentos, mas nenhum prestador registrado ('Qtd. Prestadores' é zero), a sugestão é 'Ausência de prestadores. Foco total em parceria local.'
        * **Alto % de Reembolso:** Se o percentual de reembolso da cidade está acima do percentil 80 das cidades analisadas, a sugestão é 'Alto % de reembolso. Investigar causas de insatisfação ou deficiência de prestadores.'
        * **Alto % de Intermediação:** Se o percentual de intermediação da cidade está acima do percentil 80, a sugestão é 'Alto % de intermediação. Otimizar processos de acionamento ou recrutar prestadores diretos.'
        * **Alto Tempo Médio de Chegada (TMC):** Se o TMC da cidade está acima do percentil 80, a sugestão é 'Alto tempo de chegada. Otimizar rotas ou aumentar a densidade de prestadores próximos.'
        As sugestões são combinadas e ordenadas para fornecer um plano de ação abrangente para cada município.
        """)

def page_financeiro(df):
    """Renderiza a página Financeira, incluindo filtros, KPIs financeiros e rankings."""
    st.title("Análise Financeira da Rede de Prestadores")
    st.markdown("Monitore os custos, otimize as despesas e melhore a rentabilidade da sua rede.")

    # --- Barra Lateral para Filtros ---
    st.sidebar.markdown("## ⚙️ Filtros de Análise Financeira") # Changed to markdown with gear icon

    segmento_selecionado = st.sidebar.multiselect(
        "Filtrar por Segmento (Financeiro)",
        options=[ALL_OPTION] + sorted(df['segmento'].dropna().unique().tolist()),
        default=[ALL_OPTION]
    )
    if ALL_OPTION in segmento_selecionado:
        segmento_selecionado = df['segmento'].dropna().unique().tolist()

    seguradora_selecionada = st.sidebar.multiselect(
        "Filtrar por Seguradora (Financeiro)",
        options=[ALL_OPTION] + sorted(df['seguradora'].dropna().unique().tolist()),
        default=[ALL_OPTION]
    )
    if ALL_OPTION in seguradora_selecionada:
        seguradora_selecionada = df['seguradora'].dropna().unique().tolist()

    estado_selecionado = st.sidebar.multiselect(
        "Filtrar por Estado (Financeiro)",
        options=[ALL_OPTION] + sorted(df['uf'].dropna().unique().tolist()),
        default=[ALL_OPTION]
    )
    if ALL_OPTION in estado_selecionado:
        estado_selecionado = df['uf'].dropna().unique().tolist()

    municipio_selecionado = st.sidebar.multiselect(
        "Filtrar por Cidade (Financeiro)",
        options=[ALL_OPTION] + sorted(df['municipio'].dropna().unique().tolist()),
        default=[ALL_OPTION]
    )
    if ALL_OPTION in municipio_selecionado:
        municipio_selecionado = df['municipio'].dropna().unique().tolist()

    min_date_data = df['data_abertura_atendimento'].min().date()
    max_date_data = df['data_abertura_atendimento'].max().date()
    data_inicio, data_fim = st.sidebar.date_input(
        "Período de Análise (Financeiro)",
        value=(min_date_data, max_date_data),
        min_value=min_date_data,
        max_value=max_date_data,
        format="DD/MM/YYYY",
        help="O período máximo de análise permitido é de 6 meses."
    )

    df_filtrado = apply_filters(df, segmento_selecionado, seguradora_selecionada, estado_selecionado, municipio_selecionado, data_inicio, data_fim)

    if df_filtrado.empty:
        st.info("Nenhum dado financeiro disponível com os filtros selecionados. Ajuste os filtros.")
        return

    # --- Seções da Página Financeira ---

    # 1. KPIs Financeiros Gerais
    st.markdown("---")
    st.header("KPIs Financeiros Gerais")
    st.markdown("Visualize os indicadores financeiros chave da sua rede.")

    total_gasto = df_filtrado['val_total_items'].sum()
    total_servicos = len(df_filtrado)
    cms_medio = total_gasto / total_servicos if total_servicos > 0 else 0
    total_reembolso = df_filtrado['val_reembolso'].sum()
    pct_gasto_reembolso = (total_reembolso / total_gasto) * 100 if total_gasto > 0 else 0
    total_intermediacao_servicos = df_filtrado['is_intermediacao'].sum()
    pct_intermediacao_servicos = (total_intermediacao_servicos / total_servicos) * 100 if total_servicos > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Gasto Total", f"R$ {total_gasto:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), help="Soma total dos valores dos itens em todos os serviços.")
    col2.metric("CMS Médio", f"R$ {cms_medio:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), help="Custo Médio por Serviço (CMS) por serviço.")
    col3.metric("Total de Reembolso", f"R$ {total_reembolso:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), help="Valor total de todos os reembolsos.")
    col4.metric("% Gasto c/ Reembolso", f"{pct_gasto_reembolso:.2f}%", help="Percentual do gasto total que foi via reembolso.")
    col5.metric("P/ Serv. Intermediação", f"{pct_intermediacao_servicos:.2f}%", help="Percentual de serviços que foram de intermediação.")

    # 2. Ranking de CMS por Prestador
    st.markdown("---")
    st.header("Ranking de Custo Médio por Serviço (CMS) por Prestador")
    st.markdown("Identifique os **prestadores com maior e menor CMS**. Uma alta variância pode indicar oportunidades de negociação ou revisão de processos.")

    cms_por_prestador = df_filtrado.groupby('nome_do_prestador', observed=True).agg(
        qtd_servicos=('protocolo_atendimento', 'count'),
        cms=('val_total_items', 'mean')
    ).reset_index()

    cms_por_prestador = cms_por_prestador[cms_por_prestador['qtd_servicos'] >= MIN_ATTENDANCES_FOR_RANKING]

    # RE-INSERÇÃO DA CORREÇÃO: Garantir que 'cms' não tenha NaNs antes de formatar
    cms_por_prestador['cms'] = cms_por_prestador['cms'].fillna(0)

    if not cms_por_prestador.empty:
        cms_por_prestador_display = cms_por_prestador.rename(columns={
            'nome_do_prestador': 'Prestador',
            'qtd_servicos': 'Qtd. Serviços',
            'cms': 'CMS'
        })

        st.subheader(f"Top {min(10, len(cms_por_prestador_display))} Prestadores por CMS")
        top_10_cms = cms_por_prestador_display.sort_values('CMS', ascending=False).head(10)
        fig_top_cms = px.bar(
            top_10_cms,
            x='Prestador',
            y='CMS',
            title='Prestadores com Maior Custo Médio por Serviço',
            labels={'CMS': 'CMS (R$)'},
            color_discrete_sequence=['#2021D4']
        )
        fig_top_cms.update_layout(xaxis_title="", yaxis_title="CMS (R$)", hovermode="x unified")
        st.plotly_chart(fig_top_cms, use_container_width=True)

        st.subheader("Tabela Completa de CMS por Prestador")
        st.dataframe(
            cms_por_prestador_display.sort_values('CMS', ascending=False).style.format({
                'Qtd. Serviços': '{:,.0f}',
                'CMS': 'R$ {:,.2f}'
            }),
            use_container_width=True
        )
    else:
        st.info(f"Nenhum dado de CMS por prestador (com mais de {MIN_ATTENDANCES_FOR_RANKING} serviços) disponível com os filtros selecionados.")


    # 4. Custo Médio por Faixa de Tempo de Chegada
    st.markdown("---")
    st.header("Custo Médio por Faixa de Tempo de Chegada")
    st.markdown("Avalie o **impacto do tempo de chegada no custo do serviço**. Tempos de chegada muito curtos (urgência) ou muito longos (ineficiência) podem influenciar o custo final.")

    if 'tempo_chegada_min' in df_filtrado.columns and pd.api.types.is_numeric_dtype(df_filtrado['tempo_chegada_min']):
        # Define os bins e rótulos apenas se 'tempo_chegada_min' tiver dados válidos após os filtros
        df_valid_tempo_chegada = df_filtrado.dropna(subset=['tempo_chegada_min'])

        if not df_valid_tempo_chegada.empty:
            bins = [0, 30, 60, 120, np.inf]
            labels = ['0-30 min', '31-60 min', '61-120 min', '>120 min']
            df_valid_tempo_chegada['faixa_tempo_chegada'] = pd.cut(df_valid_tempo_chegada['tempo_chegada_min'], bins=bins, labels=labels, right=True, include_lowest=True, ordered=True)

            cms_por_tempo = df_valid_tempo_chegada.groupby('faixa_tempo_chegada', observed=True).agg(
                qtd_servicos=('protocolo_atendimento', 'count'),
                cms=('val_total_items', 'mean')
            ).reset_index()

            # Filtra quaisquer categorias NaN se surgirem de `pd.cut` (por exemplo, se tempo_chegada_min era NaN)
            cms_por_tempo = cms_por_tempo.dropna(subset=['faixa_tempo_chegada'])

            # RE-INSERÇÃO DA CORREÇÃO: Garantir que 'cms' não tenha NaNs
            cms_por_tempo['cms'] = cms_por_tempo['cms'].fillna(0)


            if not cms_por_tempo.empty:
                # Garante a ordem das categorias para plotagem
                cms_por_tempo['faixa_tempo_chegada'] = pd.Categorical(cms_por_tempo['faixa_tempo_chegada'], categories=labels, ordered=True)
                cms_por_tempo = cms_por_tempo.sort_values('faixa_tempo_chegada')

                cms_por_tempo_display = cms_por_tempo.rename(columns={
                    'faixa_tempo_chegada': 'Faixa de Tempo de Chegada',
                    'qtd_servicos': 'Qtd. Serviços',
                    'cms': 'CMS'
                })

                st.subheader("CMS por Faixa de Tempo de Chegada")
                fig_cms_tempo = px.bar(
                    cms_por_tempo_display,
                    x='Faixa de Tempo de Chegada',
                    y='CMS',
                    title='Custo Médio por Tempo de Chegada',
                    labels={'CMS': 'CMS (R$)'},
                    color_discrete_sequence=['#2021D4']
                )
                fig_cms_tempo.update_layout(xaxis_title="", yaxis_title="CMS (R$)", hovermode="x unified")
                st.plotly_chart(fig_cms_tempo, use_container_width=True)

                st.subheader("Tabela de CMS por Faixa de Tempo de Chegada")
                st.dataframe(
                    cms_por_tempo_display.style.format({
                        'Qtd. Serviços': '{:,.0f}',
                        'CMS': 'R$ {:,.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("Nenhum dado de CMS por faixa de tempo de chegada disponível com os filtros selecionados e dados válidos.")
        else:
            st.info("Não há dados de 'Tempo de Chegada' válidos para as faixas de tempo após os filtros selecionados.")
    else:
        st.warning("Coluna 'tempo_chegada_min' não encontrada ou não é numérica no DataFrame. Não foi possível gerar a análise por tempo de chegada.")




    # 5. Análise de Ofensores CMS por Prestador, UF e Segmento
    st.markdown("---")
    st.header("Análise de Ofensores de CMS por Prestador")
    st.markdown("Identifique prestadores com CMS acima da **média de sua UF e segmento**, e calcule o potencial de economia.")

    # Filtrar apenas os segmentos de interesse
    segments_for_analysis = ['AUTO', 'RESID', 'VIDA']
    df_financeiro_analise = df_filtrado[df_filtrado['segmento'].isin(segments_for_analysis)].copy()

    if df_financeiro_analise.empty:
        st.info("Nenhum dado disponível para os segmentos AUTO, RESID ou VIDA com os filtros selecionados.")
        # Early exit if no data for analysis segments
    else:
        # --- NOVO CÁLCULO: CMS médio por UF e Segmento ---
        # Calcular CMS médio por UF E Segmento (globalmente nos dados filtrados)
        cms_medio_uf_segmento_df = df_financeiro_analise.groupby(['uf', 'segmento'], observed=True).agg(
            cms_medio_uf_segmento=('val_total_items', 'mean')
        ).reset_index()

        # Calcular CMS por prestador, UF e segmento
        cms_ofensores = df_financeiro_analise.groupby(['nome_do_prestador', 'uf', 'segmento'], observed=True).agg(
            qtd_servicos=('protocolo_atendimento', 'count'),
            cms_prestador=('val_total_items', 'mean')
        ).reset_index()

        # --- ATUALIZAÇÃO: Merge com o NOVO CMS médio por UF e Segmento ---
        cms_ofensores = pd.merge(cms_ofensores, cms_medio_uf_segmento_df, on=['uf', 'segmento'], how='left')

        # Filtrar prestadores com CMS significativo para análise (e.g., > MIN_ATTENDANCES_FOR_RANKING)
        cms_ofensores = cms_ofensores[cms_ofensores['qtd_servicos'] >= MIN_ATTENDANCES_FOR_RANKING]

        # Identificar ofensores: CMS do prestador é 10% maior que o CMS médio DA UF E SEGMENTO
        cms_ofensores['is_ofensor'] = (cms_ofensores['cms_prestador'] > cms_ofensores['cms_medio_uf_segmento'] * 1.10)
        
        # Calculate Potencial de Economia (R$)
        # Only calculate for offenders
        cms_ofensores['potencial_economia_rs'] = np.where(
            cms_ofensores['is_ofensor'],
            (cms_ofensores['cms_prestador'] - cms_ofensores['cms_medio_uf_segmento']) * cms_ofensores['qtd_servicos'],
            0
        )

        # RE-INSERÇÃO DA CORREÇÃO: Garantir que as colunas numéricas de ofensores não tenham NaNs
        cms_ofensores['cms_prestador'] = cms_ofensores['cms_prestador'].fillna(0)
        # --- ATUALIZAÇÃO: Usar o novo nome da coluna de CMS médio ---
        cms_ofensores['cms_medio_uf_segmento'] = cms_ofensores['cms_medio_uf_segmento'].fillna(0)
        cms_ofensores['potencial_economia_rs'] = cms_ofensores['potencial_economia_rs'].fillna(0)


        # Reordenar colunas e renomear para exibição
        cms_ofensores_display = cms_ofensores.rename(columns={
            'nome_do_prestador': 'Prestador',
            'uf': 'UF',
            'segmento': 'Segmento',
            'qtd_servicos': 'Qtd. Serviços',
            'cms_prestador': 'CMS do Prestador',
            # --- ATUALIZAÇÃO: Novo nome da coluna para exibição ---
            'cms_medio_uf_segmento': 'CMS Médio UF/Segmento',
            'is_ofensor': 'Ofensor?',
            'potencial_economia_rs': 'Potencial de Economia (R$)'
        })

        # --- FILTRAGEM ADICIONAL AQUI ---
        # Prestadores a serem excluídos (em letras maiúsculas, conforme o processamento de dados)
        prestadores_para_excluir = ['VAZIO', 'MOVIDA', 'LOCALIZA RENT A CAR']
        
        # Aplicar o filtro
        cms_ofensores_display = cms_ofensores_display[
            ~cms_ofensores_display['Prestador'].isin(prestadores_para_excluir)
        ]
        # --- FIM DA FILTRAGEM ADICIONAL ---

        if not cms_ofensores_display.empty:
            st.subheader("Tabela de Ofensores de CMS")
            st.dataframe(
                cms_ofensores_display.sort_values('Potencial de Economia (R$)', ascending=False).style.format({
                    'Qtd. Serviços': '{:,.0f}',
                    'CMS do Prestador': 'R$ {:,.2f}',
                    'CMS Médio UF/Segmento': 'R$ {:,.2f}',
                    'Potencial de Economia (R$)': 'R$ {:,.2f}'
                }),
                use_container_width=True
            )
            total_economia = cms_ofensores_display['Potencial de Economia (R$)'].sum()
            st.metric("Total Potencial de Economia (Prestadores Ofensores)", f"R$ {total_economia:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        else:
            st.info("Nenhum prestador identificado como 'ofensor' de CMS com base nos critérios e filtros selecionados.")



def page_qualidade(df_quality_kpis):
    st.title("✨ Qualidade e NPS")
    st.markdown("Esta seção exibe a evolução do Net Promoter Score (NPS) e os rankings de qualidade por cidade e prestador.")

    if df_quality_kpis.empty:
        st.info("Nenhum dado de qualidade (NPS) disponível. Por favor, verifique se o arquivo 'qualidade_kpis.parquet' existe e está preenchido.")
        return
    
    # Evolução do NPS
    st.markdown("---")
    st.subheader("Evolução do Net Promoter Score (NPS)")
    fig_nps = px.line(
        df_quality_kpis.sort_values('mes_ano'),
        x='mes_ano',
        y='nps',
        title='Evolução Mensal do NPS',
        labels={'mes_ano': 'Mês/Ano', 'nps': 'NPS'},
        markers=True
    )
    fig_nps.update_xaxes(dtick="M1", tickformat="%b\n%Y")
    fig_nps.update_yaxes(range=[-100, 100]) # NPS ranges from -100 to 100
    st.plotly_chart(fig_nps, use_container_width=True)

    # Ranking NPS por Cidade (Top 10 e Piores 10)
    st.markdown("---")
    st.subheader("NPS por Cidade")
    st.info("Visualize as cidades com melhor e pior desempenho no NPS. Isso pode indicar onde focar esforços de melhoria de serviço.")

    df_nps_cidade = df_quality_kpis.groupby('municipio')['nps'].mean().reset_index().sort_values('nps', ascending=False)

    col_nps_best_city, col_nps_worst_city = st.columns(2)

    with col_nps_best_city:
        st.markdown("#### Top 10 Cidades (Melhor NPS)")
        st.dataframe(
            df_nps_cidade.head(10).rename(columns={'municipio': 'Cidade', 'nps': 'NPS Médio'}).style.format({'NPS Médio': '{:.1f}'}),
            use_container_width=True
        )
    with col_nps_worst_city:
        st.markdown("#### Top 10 Cidades (Pior NPS)")
        st.dataframe(
            df_nps_cidade.tail(10).sort_values('nps', ascending=True).rename(columns={'municipio': 'Cidade', 'nps': 'NPS Médio'}).style.format({'NPS Médio': '{:.1f}'}),
            use_container_width=True
        )
    st.markdown("**Sugestão:** Implementar programas de incentivo ou treinamento nas cidades com baixo NPS, e replicar as melhores práticas das cidades com alto NPS.")

    # Ranking NPS por Prestador (Top 10 e Piores 10)
    st.markdown("---")
    st.subheader("NPS por Prestador")
    st.info("Identifique os prestadores com melhor e pior performance no NPS. Use esta informação para reconhecimento ou para planos de desenvolvimento.")

    df_nps_prestador = df_quality_kpis.groupby('nome_do_prestador')['nps'].mean().reset_index().sort_values('nps', ascending=False)

    col_nps_best_prestador, col_nps_worst_prestador = st.columns(2)

    with col_nps_best_prestador:
        st.markdown("#### Top 10 Prestadores (Melhor NPS)")
        st.dataframe(
            df_nps_prestador.head(10).rename(columns={'nome_do_prestador': 'Prestador', 'nps': 'NPS Médio'}).style.format({'NPS Médio': '{:.1f}'}),
            use_container_width=True
        )
    with col_nps_worst_prestador:
        st.markdown("#### Top 10 Prestadores (Pior NPS)")
        st.dataframe(
            df_nps_prestador.tail(10).sort_values('nps', ascending=True).rename(columns={'nome_do_prestador': 'Prestador', 'nps': 'NPS Médio'}).style.format({'NPS Médio': '{:.1f}'}),
            use_container_width=True
        )
    st.markdown("**Sugestão:** Avaliar treinamentos específicos ou programas de mentoria para prestadores com baixo NPS. Reconhecer e aprender com os de alto desempenho.")

# Assume these are defined elsewhere in your code, or define them here for this example
MIN_ATTENDANCES_FOR_RANKING = 10  # Example value
def calculate_prestador_score(df):
    if df.empty:
        df['score_prestador'] = []
        df['status_score'] = []
        return df

    # Ensure these columns exist before calculating.
    # If 'nps_score_calculado' doesn't exist, use 'media_nps' from aggregation directly.
    # The error message implies 'nps_score_calculado' is expected earlier.
    # Let's use the aggregated 'media_nps' and others.

    # Normalize metrics. Assuming higher is better for NPS and lower is better for others.
    # You'll need actual normalization logic if you want a precise score.
    # This is a dummy for demonstration.
    df['score_prestador'] = (
        (df['media_nps'] / 100 * 0.30) + # Normalize NPS (e.g., if 0-100)
        (df['total_atendimentos'] / df['total_atendimentos'].max() * 0.25) + # Normalize total atendimentos
        (1 - (df['pct_reembolso'] / 100)) * 0.15 + # Inverse for reimbursement
        (1 - (df['pct_intermediacao'] / 100)) * 0.10 + # Inverse for intermediation
        (1 - (df['media_tempo_chegada'] / df['media_tempo_chegada'].max())) * 0.20 # Inverse for TMC
    )
    # Re-normalize score to a more readable scale, e.g., 0-100
    df['score_prestador'] = (df['score_prestador'] - df['score_prestador'].min()) / (df['score_prestador'].max() - df['score_prestador'].min()) * 100
    
    # Classify based on quartiles for status, even if we remove the chart, it might be used by suggestions
    df['status_score'] = pd.qcut(df['score_prestador'], 4, labels=['Precisa de Atenção', 'Regular', 'Bom', 'Excelente'], duplicates='drop')
    
    return df

def get_prestador_sugestao_acao(row):
    suggestions = []
    if row['status_score'] == 'Precisa de Atenção':
        suggestions.append('Avaliar performance geral. Considerar treinamento ou revisão de contrato.')
    
    if row['media_nps'] < 50: # Example threshold for low NPS
        suggestions.append('Baixo NPS. Investigar causas de insatisfação do cliente.')
    
    if row['pct_reembolso'] > 10: # Example threshold for high reimbursement
        suggestions.append('Alto % de reembolso. Rever processos ou precificação.')
    
    if row['pct_intermediacao'] > 5: # Example threshold for high intermediation
        suggestions.append('Alto % de intermediação. Aumentar capacidade ou eficiência.')

    if row['media_tempo_chegada'] > 90: # Example threshold for high TMC
        suggestions.append('Alto TMC. Otimizar logística ou realocar prestador.')
        
    return "; ".join(suggestions) if suggestions else "N/A"

def page_score_prestador(df):
    st.title("Score do Prestador")
    st.markdown("Esta seção apresenta um score consolidado para cada prestador, com base em diversos indicadores de performance. Identifique os principais ofensores e as oportunidades de melhoria na rede.")

    if df.empty:
        st.info("Nenhum dado disponível para calcular o score dos prestadores com os filtros selecionados.")
        return

    # Aggregate data by provider
    # Ensure all columns used here exist in your actual df. If not, check your load_and_prepare_data function.
    df_prestadores_agg = df.groupby('nome_do_prestador', observed=True).agg(
        total_atendimentos=('protocolo_atendimento', 'nunique'),
        media_nps=('nps_score_calculado', 'mean'), # Make sure 'nps_score_calculado' exists and is numeric
        num_reembolsos=('is_reembolso', 'sum'), # Make sure 'is_reembolso' exists and is numeric (0/1)
        num_intermediacoes=('is_intermediacao', 'sum'), # Make sure 'is_intermediacao' exists and is numeric (0/1)
        media_tempo_chegada=('tempo_chegada_min', 'mean') # Make sure 'tempo_chegada_min' exists and is numeric
    ).reset_index()

    # Filter out providers with insufficient attendances for a meaningful score
    # Use .copy() to avoid SettingWithCopyWarning
    df_prestadores_agg = df_prestadores_agg[df_prestadores_agg['total_atendimentos'] >= MIN_ATTENDANCES_FOR_RANKING].copy()

    if df_prestadores_agg.empty:
        st.info(f"Nenhum prestador com mais de {MIN_ATTENDANCES_FOR_RANKING} atendimento(s) encontrado com os filtros aplicados.")
        return

    # Calculate percentages for reimbursement and intermediation
    df_prestadores_agg['pct_reembolso'] = np.where(
        df_prestadores_agg['total_atendimentos'] > 0,
        (df_prestadores_agg['num_reembolsos'] / df_prestadores_agg['total_atendimentos']) * 100,
        0
    )
    df_prestadores_agg['pct_intermediacao'] = np.where(
        df_prestadores_agg['total_atendimentos'] > 0,
        (df_prestadores_agg['num_intermediacoes'] / df_prestadores_agg['total_atendimentos']) * 100,
        0
    )

    # Fill NaNs for mean values if no data exists for a provider for that metric
    # Crucial for calculations and display
    df_prestadores_agg['media_nps'] = df_prestadores_agg['media_nps'].fillna(0)
    df_prestadores_agg['media_tempo_chegada'] = df_prestadores_agg['media_tempo_chegada'].fillna(df_prestadores_agg['media_tempo_chegada'].mean()) # Fill with overall mean or 0 if mean is not desired for NaNs

    # Calculate the composite score for each provider
    df_prestadores_scored = calculate_prestador_score(df_prestadores_agg)

    st.markdown("---")
    st.subheader("Desempenho Geral dos Prestadores")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5, col_kpi6 = st.columns(6) # Increased columns for more KPIs
    col_kpi1.metric("Média de Atendimentos por Prestador", f"{df_prestadores_scored['total_atendimentos'].mean():,.0f}")
    col_kpi2.metric("Média de NPS", f"{df_prestadores_scored['media_nps'].mean():.1f}")
    col_kpi3.metric("Média de TMC (min)", f"{df_prestadores_scored['media_tempo_chegada'].mean():.0f}")
    col_kpi4.metric("Média % Reembolso", f"{df_prestadores_scored['pct_reembolso'].mean():.2f}%") # New KPI
    col_kpi5.metric("Média % Intermediação", f"{df_prestadores_scored['pct_intermediacao'].mean():.2f}%") # New KPI
    col_kpi6.metric("Média Score Prestador", f"{df_prestadores_scored['score_prestador'].mean():.2f}")

   

    st.markdown("---")
    st.subheader("Prestadores com Necessidade de Atenção")
    st.info("Filtre e analise os prestadores que precisam de maior foco para melhorar a qualidade ou eficiência.")

    df_prestadores_scored['sugestao_acao'] = df_prestadores_scored.apply(get_prestador_sugestao_acao, axis=1)

    # Filter and display 'offenders' based on score and individual metrics
    df_offenders_prestador = df_prestadores_scored[
        (df_prestadores_scored['status_score'] == 'Precisa de Atenção') | # Keep score-based filtering for main offenders table
        (df_prestadores_scored['pct_reembolso'] > df_prestadores_scored['pct_reembolso'].quantile(0.75)) |
        (df_prestadores_scored['pct_intermediacao'] > df_prestadores_scored['pct_intermediacao'].quantile(0.75)) |
        (df_prestadores_scored['media_tempo_chegada'] > df_prestadores_scored['media_tempo_chegada'].quantile(0.75)) |
        (df_prestadores_scored['media_nps'] < df_prestadores_scored['media_nps'].quantile(0.25))
    ].sort_values('score_prestador', ascending=True).reset_index(drop=True)

    if not df_offenders_prestador.empty:
        st.dataframe(
            df_offenders_prestador.rename(columns={
                'nome_do_prestador': 'Prestador',
                'total_atendimentos': 'Qtd. Atendimentos',
                'media_nps': 'NPS Médio',
                'media_tempo_chegada': 'TMC Médio (min)',
                'pct_reembolso': '% Reembolso',
                'pct_intermediacao': '% Intermediação',
                'score_prestador': 'Score do Prestador',
                'status_score': 'Status',
                'sugestao_acao': 'Sugestão de Ação'
            })[[
                'Prestador', 'Qtd. Atendimentos', 'NPS Médio', 'TMC Médio (min)',
                '% Reembolso', '% Intermediação', 'Score do Prestador', 'Status', 'Sugestão de Ação'
            ]].style.format({
                'Qtd. Atendimentos': '{:,.0f}',
                'NPS Médio': '{:.2f}',
                'TMC Médio (min)': '{:.0f}',
                '% Reembolso': '{:.2f}%',
                '% Intermediacao': '{:.2f}%',
                'Score do Prestador': '{:.2f}'
            }),
            use_container_width=True
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_offenders_prestador.to_excel(writer, index=False, sheet_name='Prestadores Ofensores')
        output.seek(0)
        st.download_button(
            label="Baixar Prestadores com Necessidade de Atenção (XLSX)",
            data=output,
            file_name='prestadores_ofensores_score.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            help="Baixa os dados dos prestadores identificados com necessidade de atenção em formato XLSX."
        )
    else:
        st.info("Nenhum prestador identificado com necessidade de atenção com base nos critérios atuais.")

   

    st.markdown("---")
    st.subheader("Todos os Prestadores por Score")
    st.info("Lista completa de todos os prestadores e seus respectivos scores.")
    st.dataframe(
        df_prestadores_scored.rename(columns={
            'nome_do_prestador': 'Prestador',
            'total_atendimentos': 'Qtd. Atendimentos',
            'media_nps': 'NPS Médio',
            'media_tempo_chegada': 'TMC Médio (min)',
            'pct_reembolso': '% Reembolso',
            'pct_intermediacao': '% Intermediação',
            'score_prestador': 'Score do Prestador',
            'status_score': 'Status'
        })[[
            'Prestador', 'Qtd. Atendimentos', 'NPS Médio', 'TMC Médio (min)',
            '% Reembolso', '% Intermediação', 'Score do Prestador', 'Status'
        ]].sort_values('Score do Prestador', ascending=False).style.format({
            'Qtd. Atendimentos': '{:,.0f}',
            'NPS Médio': '{:.2f}',
            'TMC Médio (min)': '{:.0f}',
            '% Reembolso': '{:.2f}%',
            '% Intermediacao': '{:.2f}%',
            'Score do Prestador': '{:.2f}'
        }),
        use_container_width=True
    )
    output_all = io.BytesIO()
    with pd.ExcelWriter(output_all, engine='xlsxwriter') as writer:
        df_prestadores_scored.to_excel(writer, index=False, sheet_name='Score Prestadores Completo')
    output_all.seek(0)
    st.download_button(
        label="Baixar Todos os Scores de Prestadores (XLSX)",
        data=output_all,
        file_name='score_prestadores_completo.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        help="Baixa todos os dados de score dos prestadores em formato XLSX."
    )

    with st.expander("💡 Como é calculado o Score do Prestador?"):
        st.markdown(r"""
        O **Score do Prestador** é um índice composto que avalia a performance individual de cada prestador na rede, combinando diversos KPIs importantes:
        * **Normalização dos Dados:** Todos os componentes são normalizados para uma escala comum (0 a 1) para garantir que cada um contribua proporcionalmente ao score, independentemente de suas unidades originais.
        * **Componentes e Pesos:**
            * **Total de Atendimentos (25%):** Um maior volume de atendimentos indica maior contribuição para a rede. (Peso positivo)
            * **NPS Médio (30%):** Maior NPS (satisfação do cliente) é um forte indicador de qualidade. (Peso positivo)
            * **Tempo Médio de Chegada (TMC) (20%):** Menor tempo de chegada reflete agilidade e eficiência. (Peso inverso: quanto menor o TMC, maior a contribuição positiva)
            * **Percentual de Reembolso (15%):** Um menor percentual de serviços que resultam em reembolso indica maior capacidade de resolução direta e satisfação. (Peso inverso)
            * **Percentual de Intermediação (10%):** Um menor percentual de serviços intermediados significa que o prestador está atendendo diretamente as demandas. (Peso inverso)

        **Fórmula Simplificada:**
        $$ \text{Score Prestador} = \left( \text{Atendimentos Normalizados} \times 0.25 \right) + \left( \text{NPS Normalizado} \times 0.30 \right) + \left( (1 - \text{TMC Normalizado}) \times 0.20 \right) + \left( (1 - \text{Reembolso Normalizado}) \times 0.15 \right) + \left( (1 - \text{Intermediação Normalizada}) \times 0.10 \right) $$

        **Classificação do Status do Score:** O status é determinado pelos quartis do Score do Prestador:
        * **Precisa de Atenção:** Prestadores no quartil inferior (piores 25%).
        * **Regular:** Prestadores entre o primeiro e o segundo quartil (25% a 50%).
        * **Bom:** Prestadores entre o segundo e o terceiro quartil (50% a 75%).
        * **Excelente:** Prestadores no quartil superior (melhores 25%).
        """)
    with st.expander("💡 Como são geradas as Sugestões de Ação?"):
        st.markdown("""
        As sugestões de ação são geradas dinamicamente para cada prestador com base no seu score e nos seus indicadores específicos:
        * **Precisa de Atenção:** Se o status do score for 'Precisa de Atenção', a sugestão é 'Avaliar performance geral. Treinamento ou revisão de contrato.'
        * **Regular:** Se o status do score for 'Regular', a sugestão é 'Otimizar processos para melhoria contínua.'
        * **Baixo NPS:** Se o NPS Médio for inferior a um limiar (ex: 0), a sugestão é 'Baixo NPS. Investigar causas de insatisfação do cliente.'
        * **Alto % de Reembolso:** Se o percentual de reembolso for significativamente maior que a média, sugere-se 'Alto % de reembolso. Rever processos ou precificação.'
        * **Alto % de Intermediação:** Se o percentual de intermediação for significativamente maior que a média, sugere-se 'Alto % de intermediação. Aumentar capacidade ou eficiência.'
        * **Alto Tempo de Chegada (TMC):** Se o tempo médio de chegada for superior a um limiar (ex: 60 minutos), sugere-se 'Alto TMC. Otimizar logística ou realocar prestador.'
        """)

def page_qualidade_nps(df: pd.DataFrame) -> None:
    """Renderiza a página de Qualidade (NPS e TMC)."""
    st.title("✨ Qualidade (NPS & TMC)")
    st.markdown("Avalie a satisfação do cliente (NPS) e a eficiência no tempo de chegada (TMC) para identificar oportunidades de melhoria.")

    # Injeta CSS personalizado, se houver
    # inject_css() # Descomente se tiver essa função

    # --- Sidebar de Filtros ---
    st.sidebar.title("Opções de Filtro")
    # st.sidebar.image("C:\\Temp\\logo.png", use_container_width=False, width=1000) # Verifique o caminho da imagem
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros de Dados")

    # Garante que 'data_abertura_atendimento' é datetime para min/max
    df['data_abertura_atendimento'] = pd.to_datetime(df['data_abertura_atendimento'])
    
    # Filtros
    segmento_selecionado = st.sidebar.multiselect(
        "Filtrar por Segmento",
        options=[ALL_OPTION] + sorted(df['segmento'].dropna().unique().tolist()),
        default=[ALL_OPTION],
        key="qualidade_segmento"
    )
    if ALL_OPTION in segmento_selecionado:
        segmento_selecionado = df['segmento'].dropna().unique().tolist()

    seguradora_selecionada = st.sidebar.multiselect(
        "Filtrar por Seguradora",
        options=[ALL_OPTION] + sorted(df['seguradora'].dropna().unique().tolist()),
        default=[ALL_OPTION],
        key="qualidade_seguradora"
    )
    if ALL_OPTION in seguradora_selecionada:
        seguradora_selecionada = df['seguradora'].dropna().unique().tolist()

    estado_selecionado = st.sidebar.multiselect(
        "Filtrar por Estado",
        options=[ALL_OPTION] + sorted(df['uf'].dropna().unique().tolist()),
        default=[ALL_OPTION],
        key="qualidade_estado"
    )
    if ALL_OPTION in estado_selecionado:
        estado_selecionado = df['uf'].dropna().unique().tolist()

    municipio_selecionado = st.sidebar.multiselect(
        "Filtrar por Cidade",
        options=[ALL_OPTION] + sorted(df['municipio'].dropna().unique().tolist()),
        default=[ALL_OPTION],
        key="qualidade_municipio"
    )
    if ALL_OPTION in municipio_selecionado:
        municipio_selecionado = df['municipio'].dropna().unique().tolist()

    min_date_data = df['data_abertura_atendimento'].min().date()
    max_date_data = df['data_abertura_atendimento'].max().date()
    data_inicio, data_fim = st.sidebar.date_input(
        "Período de Análise",
        value=(min_date_data, max_date_data),
        min_value=min_date_data,
        max_value=max_date_data,
        key="qualidade_data_range"
    )

    df_filtered = apply_filters(df, segmento_selecionado, seguradora_selecionada, estado_selecionado, municipio_selecionado, data_inicio, data_fim)

    if df_filtered.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados. Ajuste os filtros para ver os resultados.")
        return

    # --- Carregar Dados Pré-processados para Rankings (melhora performance) ---
    # Certifique-se de que esses arquivos existam e estejam no caminho correto.
    try:
        df_nps_cidade_pre = pd.read_parquet("https://github.com/vinikrebs/ScrorePrestador/raw/main/processed_nps_by_city.parquet")
        df_nps_prestador_pre = pd.read_parquet("https://github.com/vinikrebs/ScrorePrestador/raw/main/processed_nps_by_provider.parquet")

        # Garante que 'mes_ano' é datetime para filtragem
        df_nps_cidade_pre['mes_ano'] = pd.to_datetime(df_nps_cidade_pre['mes_ano'])
        df_nps_prestador_pre['mes_ano'] = pd.to_datetime(df_nps_prestador_pre['mes_ano'])

        # Filtrar os dados pré-processados pelo período selecionado
        df_nps_cidade_pre_filtered = df_nps_cidade_pre[
            (df_nps_cidade_pre['mes_ano'].dt.date >= data_inicio) &
            (df_nps_cidade_pre['mes_ano'].dt.date <= data_fim)
        ].copy()

        df_nps_prestador_pre_filtered = df_nps_prestador_pre[
            (df_nps_prestador_pre['mes_ano'].dt.date >= data_inicio) &
            (df_nps_prestador_pre['mes_ano'].dt.date <= data_fim)
        ].copy()

    except FileNotFoundError:
        st.error("Arquivos de dados pré-processados de NPS não encontrados. Verifique os caminhos: 'https://github.com/vinikrebs/ScrorePrestador/raw/main/processed_nps_by_city.parquet' e 'https://github.com/vinikrebs/ScrorePrestador/raw/main/processed_nps_by_provider.parquet'.")
        return


    # --- TOP CIDADES OFENSORAS E MELHORES NPS ---
    st.markdown("---")
    st.header("Cidades: Piores e Melhores NPS")
    st.info("Identifique cidades com baixo NPS para ações focadas e celebre as de alto desempenho.")
    # Filtrar por um número mínimo de avaliações para relevância
    min_nps_evaluations_city = st.number_input("Mín. Avaliações (Cidades)", min_value=1, value=5, key="min_eval_city_nps")
  
    col_nps_cidade_low, col_nps_cidade_high = st.columns(2)

    with col_nps_cidade_low:
        st.subheader("Cidades com Pior NPS")
      
        # Filtra os dados pré-processados e agrega por município para o NPS final (se não já agregado)
        df_nps_cidade_agg = df_nps_cidade_pre_filtered.groupby(['municipio', 'uf']).agg(
            total_avaliacoes=('nps_detratores', 'count'), # Usa qualquer coluna para contar o total de respostas
            nps_score=('nps_score_calculado', 'mean')
        ).reset_index()

        df_nps_cidade_low = df_nps_cidade_agg[df_nps_cidade_agg['total_avaliacoes'] >= min_nps_evaluations_city] \
                                .sort_values('nps_score', ascending=True).head(10).copy()

        if not df_nps_cidade_low.empty:
            st.dataframe(
                df_nps_cidade_low.rename(columns={'municipio': 'Cidade', 'uf': 'UF', 'total_avaliacoes': 'Qtd. Avaliações', 'nps_score': 'NPS'}).style.format({
                    'Qtd. Avaliações': '{:,.0f}',
                    'NPS': '{:,.1f}'
                }),
                use_container_width=True
            )
        else:
            st.info("Nenhuma cidade encontrada com baixo NPS e número mínimo de avaliações.")

    with col_nps_cidade_high:
        st.subheader("Cidades com Melhor NPS")
        df_nps_cidade_high = df_nps_cidade_agg[df_nps_cidade_agg['total_avaliacoes'] >= min_nps_evaluations_city] \
                                .sort_values('nps_score', ascending=False).head(10).copy()

        if not df_nps_cidade_high.empty:
            st.dataframe(
                df_nps_cidade_high.rename(columns={'municipio': 'Cidade', 'uf': 'UF', 'total_avaliacoes': 'Qtd. Avaliações', 'nps_score': 'NPS'}).style.format({
                    'Qtd. Avaliações': '{:,.0f}',
                    'NPS': '{:,.1f}'
                }),
                use_container_width=True
            )
        else:
            st.info("Nenhuma cidade encontrada com alto NPS e número mínimo de avaliações.")

    # --- TOP PRESTADORES OFENSORES E MELHORES NPS ---
    st.markdown("---")
    st.header("Prestadores: Piores e Melhores NPS")
    st.info("Identifique prestadores que impactam negativamente o NPS e reconheça os de melhor desempenho.")
    min_nps_evaluations_provider = st.number_input("Mín. Avaliações (Prestadores)", min_value=1, value=5, key="min_eval_provider_nps")

    col_nps_prestador_low, col_nps_prestador_high = st.columns(2)

    with col_nps_prestador_low:
        st.subheader("Prestadores com Pior NPS")
        
        # Filtra os dados pré-processados e agrega por prestador para o NPS final
        df_nps_prestador_agg = df_nps_prestador_pre_filtered.groupby('nome_do_prestador').agg(
            total_avaliacoes=('nps_detratores', 'count'), # Usa qualquer coluna para contar o total de respostas
            nps_score=('nps_score_calculado', 'mean')
        ).reset_index()

        df_nps_prestador_low = df_nps_prestador_agg[df_nps_prestador_agg['total_avaliacoes'] >= min_nps_evaluations_provider] \
                                .sort_values('nps_score', ascending=True).head(10).copy()

        if not df_nps_prestador_low.empty:
            st.dataframe(
                df_nps_prestador_low.rename(columns={'nome_do_prestador': 'Prestador', 'total_avaliacoes': 'Qtd. Avaliações', 'nps_score': 'NPS'}).style.format({
                    'Qtd. Avaliações': '{:,.0f}',
                    'NPS': '{:,.1f}'
                }),
                use_container_width=True
            )
        else:
            st.info("Nenhum prestador encontrado com baixo NPS e número mínimo de avaliações.")

    with col_nps_prestador_high:
        st.subheader("Prestadores com Melhor NPS")
        df_nps_prestador_high = df_nps_prestador_agg[df_nps_prestador_agg['total_avaliacoes'] >= min_nps_evaluations_provider] \
                                .sort_values('nps_score', ascending=False).head(10).copy()

        if not df_nps_prestador_high.empty:
            st.dataframe(
                df_nps_prestador_high.rename(columns={'nome_do_prestador': 'Prestador', 'total_avaliacoes': 'Qtd. Avaliações', 'nps_score': 'NPS'}).style.format({
                    'Qtd. Avaliações': '{:,.0f}',
                    'NPS': '{:,.1f}'
                }),
                use_container_width=True
            )
        else:
            st.info("Nenhum prestador encontrado com alto NPS e número mínimo de avaliações.")

    # --- Distribuição do TMC 
    st.markdown("---")
    st.header("Distribuição do TMC por Segmento e Seguradora")

    col_tmc_segmento, col_tmc_seguradora = st.columns(2)

    with col_tmc_segmento:
        st.subheader("TMC por Segmento")
        tmc_por_segmento = df_filtered.groupby('segmento')['tempo_chegada_min'].mean().reset_index()
        tmc_por_segmento['tempo_chegada_min'] = tmc_por_segmento['tempo_chegada_min'].fillna(0)
        st.dataframe(
            tmc_por_segmento.rename(columns={'tempo_chegada_min': 'TMC Médio (min)'}).style.format(
                {'TMC Médio (min)': '{:.0f}'}
            ),
            use_container_width=True
        )

    with col_tmc_seguradora:
        st.subheader("TMC por Seguradora")
        tmc_por_seguradora = df_filtered.groupby('seguradora')['tempo_chegada_min'].mean().reset_index()
        tmc_por_seguradora['tempo_chegada_min'] = tmc_por_seguradora['tempo_chegada_min'].fillna(0)
        st.dataframe(
            tmc_por_seguradora.rename(columns={'tempo_chegada_min': 'TMC Médio (min)'}).style.format(
                {'TMC Médio (min)': '{:.0f}'}
            ),
            use_container_width=True
        )
     
# --- Main ---
def main():
    inject_css()

    if not st.session_state.get('logged_in'):
        login_page()
    else:
        # --- PARTE 1: Score e Logo ---
        st.sidebar.markdown(
            "<h2 style='color: #000000; text-align: center; font-size: 1.5rem;'>Score do Prestador</h2>",
            unsafe_allow_html=True
        )
        st.sidebar.image("https://github.com/vinikrebs/ScrorePrestador/raw/main/logo.png", use_container_width=False, width=1000)

        with st.sidebar:
            selected_page = option_menu(
                menu_title=None,
                options=["Informações", "Capilaridade", "Financeiro", "Score Prestador", "Qualidade"],
                icons=["info-circle-fill", "globe", "currency-dollar", "graph-up", "award"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#FFFFFF", "box-shadow": "0px 0px 10px rgba(0, 0, 0, 0.1)"},
                    "icon": {"color": "#2021D4", "font-size": "20px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#E6F0F8", "color": "#333333"},
                    "nav-link-selected": {"background-color": "white", "color": "#2021D4", "font-weight": "bold"},
                }
            )

        # --- PARTE 2: Filtros centrais (Capilaridade e Financeiro) ---
        # Filtros compartilhados
        with st.sidebar:
            st.markdown("### ⚙️ Filtros de Dados")
            # Filtros Capilaridade e Financeiro (compartilhados)
            # You need df_atendimentos to be available here. It's usually loaded outside main
            # or passed as an argument. Assuming it's a global variable here due to previous context.
            global df_atendimentos # Ensure df_atendimentos is accessible

            segmento_options = [ALL_OPTION] + sorted(df_atendimentos['segmento'].dropna().unique().tolist())
            seguradora_options = [ALL_OPTION] + sorted(df_atendimentos['seguradora'].dropna().unique().tolist())
            estado_options = [ALL_OPTION] + sorted(df_atendimentos['uf'].dropna().unique().tolist())
            municipio_options = [ALL_OPTION] + sorted(df_atendimentos['municipio'].dropna().unique().tolist())
            min_date_data = df_atendimentos['data_abertura_atendimento'].min().date()
            max_date_data = df_atendimentos['data_abertura_atendimento'].max().date()

            segmento_selecionado = st.multiselect(
                "Filtrar por Segmento",
                options=segmento_options,
                default=[ALL_OPTION],
                key="sidebar_segmento"
            )
            if ALL_OPTION in segmento_selecionado:
                segmento_selecionado = df_atendimentos['segmento'].dropna().unique().tolist()

            seguradora_selecionada = st.multiselect(
                "Filtrar por Seguradora",
                options=seguradora_options,
                default=[ALL_OPTION],
                key="sidebar_seguradora"
            )
            if ALL_OPTION in seguradora_selecionada:
                seguradora_selecionada = df_atendimentos['seguradora'].dropna().unique().tolist()

            estado_selecionado = st.multiselect(
                "Filtrar por Estado",
                options=estado_options,
                default=[ALL_OPTION],
                key="sidebar_estado"
            )
            if ALL_OPTION in estado_selecionado:
                estado_selecionado = df_atendimentos['uf'].dropna().unique().tolist()

            municipio_selecionado = st.multiselect(
                "Filtrar por Cidade",
                options=municipio_options,
                default=[ALL_OPTION],
                key="sidebar_municipio"
            )
            if ALL_OPTION in municipio_selecionado:
                municipio_selecionado = df_atendimentos['municipio'].dropna().unique().tolist()

            data_inicio, data_fim = st.date_input(
                "Período de Análise",
                value=(min_date_data, max_date_data),
                min_value=min_date_data,
                max_value=max_date_data,
                key="sidebar_data_range"
            )

            # Slider moved to sidebar
            min_atendimentos_cidade_sidebar = st.slider(
                "Mínimo de Atendimentos por Cidade para Análise",
                min_value=1,
                max_value=200,
                value=MIN_ATTENDANCES_FOR_CITY_ANALYSIS,
                help="Cidades com número de atendimentos abaixo deste valor não serão incluídas na análise de capilaridade detalhada."
            )

            st.markdown("---")

        # --- PARTE 3: Botões de navegação e logoff ---

            st.write(f"Última Atualização: {datetime.date.today().strftime('%d/%m/%Y')}")
            if st.button("Sair", key="logoff_button_sidebar_global"):
                st.session_state['logged_in'] = False
                st.session_state.pop('username', None)
                st.rerun()

        # --- Aplicação dos filtros globais nas páginas relevantes ---
        df_filtered = apply_filters(
            df_atendimentos,
            segmento_selecionado,
            seguradora_selecionada,
            estado_selecionado,
            municipio_selecionado,
            data_inicio,
            data_fim
        )

        if selected_page == "Informações":
            page_informacao()
        elif selected_page == "Capilaridade":
            page_capilaridade(df_filtered, min_atendimentos_cidade_sidebar)
        elif selected_page == "Financeiro":
            page_financeiro(df_filtered)
        elif selected_page == "Score Prestador":
            page_score_prestador(df_filtered) # <-- Corrected line!
        elif selected_page == "Qualidade":
            page_qualidade_nps(df_filtered)

if __name__ == "__main__":
    main()