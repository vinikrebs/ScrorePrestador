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
FINANCIAL_KPI_FILE = "C:\\Temp\\processed_financeiro.parquet"
ATENDIMENTO_FILE_PATH = 'C:\\Temp\\processed_atendimentos.parquet'
NPS_CIDADE_PATH = "C:\Temp\processed_nps_by_city.parquet"
NPS_PRESTADOR_PATH = "C:\Temp\processed_nps_by_provider.parquet"
LOGO_PATH = "C:\\Temp\\logo.png"

# --- Função da Página de Login ---
def login_page():
    """Renderiza a página de login."""
    col_logo_left, col_logo_center, col_logo_right = st.columns([1.5, 3, 1.5])
    with col_logo_center:
        # Caminho da imagem no servidor em produção pode ser diferente
        # Centralizando a imagem usando markdown e HTML
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(LOGO_PATH, use_container_width=False, width=1600)
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
def load_and_prepare_data(atendimentos_file_path, nps_cidade_path, nps_prestador_path):
    pd.set_option('future.no_silent_downcasting', True)

    # Carrega e processa df_final
    try:
        df_final = pd.read_parquet(atendimentos_file_path)
    except FileNotFoundError:
        st.error(f"Erro: Arquivo '{atendimentos_file_path}' não encontrado. Verifique o caminho.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Parquet de atendimentos: {e}. Verifique se o arquivo está no formato correto.")
        st.stop()

    if 'data_abertura_atendimento' in df_final.columns:
        df_final['data_abertura_atendimento'] = pd.to_datetime(df_final['data_abertura_atendimento'], errors='coerce')
    else:
        st.error("Coluna 'data_abertura_atendimento' não encontrada no DataFrame de atendimentos.")
        st.stop()

    df_final = df_final.dropna(subset=['data_abertura_atendimento']).copy()

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
    
    # --- Carrega e processa df_nps_cidade ---
    try:
        df_nps_cidade = pd.read_parquet(nps_cidade_path)
        # Ajustar tipos e lidar com NaNs nas colunas de NPS
        for col in ['nps_score_calculado', 'nps_promotores', 'nps_neutros', 'nps_detratores']:
            if col in df_nps_cidade.columns:
                df_nps_cidade[col] = pd.to_numeric(df_nps_cidade[col], errors='coerce').fillna(0) # Trata ' ' como NaN e preenche com 0
        if 'mes_ano' in df_nps_cidade.columns:
             df_nps_cidade['mes_ano'] = pd.to_datetime(df_nps_cidade['mes_ano'], errors='coerce').dt.to_period('M')
        else:
             st.warning("Coluna 'mes_ano' não encontrada no arquivo NPS por cidade.")
    except FileNotFoundError:
        st.warning(f"Aviso: Arquivo '{nps_cidade_path}' não encontrado. A análise de NPS por cidade pode estar incompleta.")
        df_nps_cidade = pd.DataFrame() # Retorna DataFrame vazio para evitar erros
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Parquet de NPS por cidade: {e}.")
        df_nps_cidade = pd.DataFrame()

    # --- Carrega e processa df_nps_prestador ---
    try:
        df_nps_prestador = pd.read_parquet(nps_prestador_path)
        # Ajustar tipos e lidar com NaNs nas colunas de NPS
        for col in ['nps_score_calculado', 'nps_promotores', 'nps_neutros', 'nps_detratores']:
            if col in df_nps_prestador.columns:
                df_nps_prestador[col] = pd.to_numeric(df_nps_prestador[col], errors='coerce').fillna(0) # Trata ' ' como NaN e preenche com 0
        if 'mes_ano' in df_nps_prestador.columns:
            df_nps_prestador['mes_ano'] = pd.to_datetime(df_nps_prestador['mes_ano'], errors='coerce').dt.to_period('M')
        else:
            st.warning("Coluna 'mes_ano' não encontrada no arquivo NPS por prestador.")
    except FileNotFoundError:
        st.warning(f"Aviso: Arquivo '{nps_prestador_path}' não encontrado. A análise de NPS por prestador pode estar incompleta.")
        df_nps_prestador = pd.DataFrame() # Retorna DataFrame vazio para evitar erros
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Parquet de NPS por prestador: {e}.")
        df_nps_prestador = pd.DataFrame()

    return df_final, df_nps_cidade, df_nps_prestador,

# --- Função Geral de Aplicação de Filtros ---
def apply_filters(df, selected_segments, selected_insurers, selected_states, selected_municipios, start_date, end_date):
    """Aplica filtros comuns ao DataFrame."""
    df_filtrado = df[
        (df['segmento'].isin(selected_segments)) &
        (df['seguradora'].isin(selected_insurers)) &
        (df['uf'].isin(selected_states)) &
        (df['municipio'].isin(selected_municipios)) &
        (df['data_abertura_atendimento'].dt.date >= start_date) &
        (df['data_abertura_atendimento'].dt.date <= end_date)
    ].copy()
    return df_filtrado

# --- Funções para Páginas (Pilares) ---
def page_informacao():
    """Renderiza a página de informações gerais do dashboard."""
    st.markdown("""
    # Monitoramento Inteligente da Rede de Prestadores A24h

    **Bem-vindo ao Dashboard de Monitoramento Inteligente da Rede de Prestadores A24h!** Esta plataforma foi desenhada para transformar dados complexos em *insights acionáveis*, permitindo decisões ágeis e estratégicas para otimizar performance, reduzir custos e garantir a melhor experiência para nossos clientes.

    **Explore os pilares abaixo** e descubra diferentes perspectivas da nossa rede de parceiros:

    ---
    """)

    st.markdown("## 📍 Pilar Capilaridade")
    st.markdown("""
    Monitora a **distribuição** e a **disponibilidade** dos prestadores em relação à demanda da região.

    **Objetivo:** Ajuda a identificar áreas descobertas, equilibrar recursos e evitar sobrecarga em determinados prestadores.
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

    **Objetivo:** Permite controlar custos, identificar oportunidades de economia e evitar distorções regionais.
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

    **Objetivo:** Ajuda a identificar padrões de uso, prever demandas futuras e gerenciar melhor a capacidade da rede.
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

    **Objetivo:** Monitora o nível de confiança do cliente e identifica oportunidades de melhoria.
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


def calculate_capilaridade_index(df_agregado_cidade):
    if df_agregado_cidade.empty:
        return pd.DataFrame()

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
            
            bins = sorted(list(set([
                df_agregado_cidade['indice_capilaridade'].min() - 0.001, 
                q1, 
                q3, 
                df_agregado_cidade['indice_capilaridade'].max() + 0.001
            ])))
            
            labels_map = {
                2: ['Carência Assistencial', 'Boa Capilaridade'],
                3: ['Carência Assistencial', 'Capilaridade Regular', 'Boa Capilaridade']
            }
            labels = labels_map.get(len(bins) - 1, ['Capilaridade Regular']) 

            df_agregado_cidade['status_capilaridade'] = pd.cut(
                df_agregado_cidade['indice_capilaridade'],
                bins=bins,
                labels=labels,
                right=True,
                duplicates='drop'
            )
            df_agregado_cidade['status_capilaridade'] = df_agregado_cidade['status_capilaridade'].fillna('Capilaridade Regular')
            
            df_agregado_cidade['status_capilaridade'] = pd.Categorical(
                df_agregado_cidade['status_capilaridade'],
                categories=['Carência Assistencial', 'Capilaridade Regular', 'Boa Capilaridade'],
                ordered=True
            )
    else:
        df_agregado_cidade['status_capilaridade'] = 'N/A'
    return df_agregado_cidade

def get_sugestao_acao(row, df_agregado_cidade, min_atendimentos_cidade):
    sugestoes = set()
    if row['status_capilaridade'] == 'Carência Assistencial':
        sugestoes.add("Recrutamento urgente de prestadores. Analisar concorrência local.")
    
    if row['num_servicos'] > 0 and row['num_prestadores'] == 0:
        sugestoes.add("Ausência de prestadores. Foco total em parceria local.")
    
    if row['num_servicos'] >= min_atendimentos_cidade: 
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
    st.markdown("---")
    st.header("Classificações de Problemas Específicos")
    st.markdown("Cidades com os maiores desafios em reembolso e intermediação.")
    
    col_reembolso_rank, col_intermediacao_rank = st.columns(2)
    
    with col_reembolso_rank:
        st.subheader("Cidades por % de Reembolso")
        if 'total_valor_servicos' not in df_agregado_cidade.columns:
            df_agregado_cidade['total_valor_servicos'] = 0 
        
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
        if 'total_valor_servicos' not in df_agregado_cidade.columns:
            df_agregado_cidade['total_valor_servicos'] = 0 

        df_intermediacao_rank = df_agregado_cidade.sort_values('pct_intermediacao', ascending=False).head(10).copy()
        
        df_intermediacao_rank_display = df_intermediacao_rank.rename(columns={
            'municipio': 'Cidade',
            'uf': 'UF',
            'num_servicos': 'Qtd. Serviços',
            'pct_intermediacao': '% Intermediação',
            'total_valor_servicos': 'Valor Total'
        })
        st.dataframe(
            df_intermediacao_rank_display[['Cidade', 'UF', 'Qtd. Serviços', '% Intermediação', 'Valor Total']].style.format({
                'Qtd. Serviços': '{:,.0f}',
                '% Intermediacao': '{:.2f}%',
                'Valor Total': 'R$ {:,.2f}'
            }),
            use_container_width=True
        )
        st.markdown("**Sugestão:** Otimizar processos de acionamento ou recrutar prestadores diretos para reduzir intermediações.")


def display_capilaridade_kpis(df_filtrado_original, df_agregado_cidade_com_indice=None):
    st.markdown("---")
    st.header("KPIs Gerais de Capilaridade")
    
    total_servicos = len(df_filtrado_original)
    total_prestadores_unicos = df_filtrado_original['nome_do_prestador'].nunique()
    total_cidades_atendidas = df_filtrado_original['municipio'].nunique()
    media_tempo_chegada = df_filtrado_original['tempo_chegada_min'].mean()
    pct_reembolso = (df_filtrado_original['is_reembolso'].sum() / total_servicos) * 100 if total_servicos > 0 else 0
    pct_intermediacao = (df_filtrado_original['is_intermediacao'].sum() / total_servicos) * 100 if total_servicos > 0 else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total de Serviços", f"{total_servicos:,}".replace(',', '.'), help="Número total de atendimentos registrados com os filtros aplicados.")
    col2.metric("Prestadores Únicos", f"{total_prestadores_unicos:,}".replace(',', '.'), help="Número de prestadores distintos que realizaram atendimentos com os filtros aplicados.")
    col3.metric("Cidades Atendidas", f"{total_cidades_atendidas:,}".replace(',', '.'), help="Número de municípios onde houve atendimentos com os filtros aplicados.")
    col4.metric("TMC (min)", f"{media_tempo_chegada:.0f} min" if not pd.isna(media_tempo_chegada) else "N/A", help="Tempo Médio de Chegada do prestador ao local do serviço, em minutos.")
    col5.metric("Perc. Reembolso", f"{pct_reembolso:.2f}%", help="Percentual de serviços que geraram algum tipo de reembolso, indicando falha na cobertura direta ou preferência do cliente.")
    col6.metric("Perc. Intermediação", f"{pct_intermediacao:.2f}%", help="Percentual de serviços que foram realizados por meio de intermediação, e não por prestadores diretos da rede.")

    if df_agregado_cidade_com_indice is not None and not df_agregado_cidade_com_indice.empty:
        st.markdown("---")
        st.header("Dispersão de Capilaridade por Cidade")
        st.info("Visualize a relação entre o número de serviços e a quantidade de prestadores. Círculos maiores indicam maior volume de atendimentos. As cores representam o status de capilaridade da cidade.")

        category_order = ['Carência Assistencial', 'Capilaridade Regular', 'Boa Capilaridade']
        status_colors = {
            'Carência Assistencial': '#EF5350',
            'Capilaridade Regular': '#FFCA28',
            'Boa Capilaridade': '#66BB6A'
        }

        fig_capilaridade = px.scatter(
            df_agregado_cidade_com_indice, 
            x='num_servicos',
            y='num_prestadores',
            color='status_capilaridade',
            size='num_servicos',
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
            log_x=True,
            )

        fig_capilaridade.update_layout(
            xaxis_title="Número de Serviços (Atendimentos)",
            yaxis_title="Número de Prestadores",
            legend_title="Status de Capilaridade",
            hovermode="closest",
            yaxis=dict(showgrid=False) 
        )
        
        st.plotly_chart(fig_capilaridade, use_container_width=True)

def page_capilaridade(df):
    st.title("Capilaridade da Rede")
    st.markdown("Esta seção oferece uma visão detalhada da distribuição e cobertura dos nossos prestadores, identificando áreas de alta demanda e oportunidades de expansão.")


    st.markdown("---")

    df_agregado_cidade = df.groupby(['uf', 'municipio'], observed=True).agg(
        num_servicos=('protocolo_atendimento', 'nunique'),
        num_prestadores=('nome_do_prestador', 'nunique'),
        num_reembolsos=('is_reembolso', lambda x: x.sum()),
        num_intermediacoes=('is_intermediacao', lambda x: x.sum()),
        media_tempo_chegada=('tempo_chegada_min', 'mean'),
        total_valor_servicos=('val_total_items', 'sum')
    ).reset_index()

    min_atendimentos_cidade = st.number_input(
        label="Defina o Mínimo de Atendimentos por Prestador", 
        min_value=1,
        max_value=200, 
        value=MIN_ATTENDANCES_FOR_CITY_ANALYSIS, 
        step=1,
        help="Cidades com número de atendimentos abaixo deste valor não serão incluídas na análise de capilaridade detalhada."
    )
 


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
                    'Qtd. Não Atendidos', 
                    'Qtd. Prestadores', 
                    '% Reembolso', 
                    '% Intermediação', 
                    'TMC Médio (min)', 
                    'Índice Capilaridade', 
                    'Status Capilaridade', 
                    'Sugestão de Ação'
                ]].style.format({
                        'Qtd. Serviços': '{:,.0f}',
                        'Qtd. Não Atendidos': '{:,.0f}', 
                        'Qtd. Prestadores': '{:,.0f}',
                        '% Reembolso': '{:.2f}%',
                        '% Intermediacao': '{:.2f}%',
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
    st.title("Análise Financeira da Rede de Prestadores")
    st.markdown("Monitore os custos, otimize as despesas e melhore a rentabilidade da sua rede.")

    st.markdown("---")
    st.header("KPIs Financeiros Gerais")
    st.markdown("Visualize os indicadores financeiros chave da sua rede.")

    total_gasto = df['val_total_items'].sum()
    total_servicos = len(df)
    cms_medio = total_gasto / total_servicos if total_servicos > 0 else 0
    total_reembolso = df['val_reembolso'].sum()
    pct_gasto_reembolso = (total_reembolso / total_gasto) * 100 if total_gasto > 0 else 0
    total_intermediacao_servicos = df['is_intermediacao'].sum()
    pct_intermediacao_servicos = (total_intermediacao_servicos / total_servicos) * 100 if total_servicos > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Gasto Total", f"R$ {total_gasto:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), help="Soma total dos valores dos itens em todos os serviços.")
    col2.metric("CMS Médio", f"R$ {cms_medio:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), help="Custo Médio por Serviço (CMS) por serviço.")
    col3.metric("Total de Reembolso", f"R$ {total_reembolso:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), help="Valor total de todos os reembolsos.")
    col4.metric("% Gasto c/ Reembolso", f"{pct_gasto_reembolso:.2f}%", help="Percentual do gasto total que foi via reembolso.")
    col5.metric("P/ Serv. Intermediação", f"{pct_intermediacao_servicos:.2f}%", help="Percentual de serviços que foram de intermediação.")

    st.markdown("---")
    min_servicos_prestador = st.number_input(
        label="Defina o Mínimo de Atendimentos por Prestador", 
        min_value=1,
        max_value=200,
        value=MIN_ATTENDANCES_FOR_RANKING,
        step=1,
        help="Prestadores com um número de serviços abaixo deste valor não serão incluídos nas análises de ranking e ofensores."
    )

    st.markdown("---")
    st.header("Ranking de Custo Médio por Serviço (CMS) por Prestador")
    st.markdown("Identifique os **prestadores com maior e menor CMS**. Uma alta variância pode indicar oportunidades de negociação ou revisão de processos.")

    cms_por_prestador = df.groupby('nome_do_prestador', observed=True).agg(
        qtd_servicos=('protocolo_atendimento', 'count'),
        cms=('val_total_items', 'mean')
    ).reset_index()

    cms_por_prestador = cms_por_prestador[cms_por_prestador['qtd_servicos'] >= min_servicos_prestador]

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
        st.info(f"Nenhum dado de CMS por prestador (com mais de {min_servicos_prestador} serviços) disponível com os filtros selecionados.")


    st.markdown("---")
    st.header("Custo Médio por Faixa de Tempo de Chegada")
    st.markdown("Avalie o **impacto do tempo de chegada no custo do serviço**. Tempos de chegada muito curtos (urgência) ou muito longos (ineficiência) podem influenciar o custo final.")

    if 'tempo_chegada_min' in df.columns and pd.api.types.is_numeric_dtype(df['tempo_chegada_min']):
        df_valid_tempo_chegada = df.dropna(subset=['tempo_chegada_min'])

        if not df_valid_tempo_chegada.empty:
            bins = [0, 30, 60, 120, np.inf]
            labels = ['0-30 min', '31-60 min', '61-120 min', '>120 min']
            df_valid_tempo_chegada['faixa_tempo_chegada'] = pd.cut(df_valid_tempo_chegada['tempo_chegada_min'], bins=bins, labels=labels, right=True, include_lowest=True, ordered=True)

            cms_por_tempo = df_valid_tempo_chegada.groupby('faixa_tempo_chegada', observed=True).agg(
                qtd_servicos=('protocolo_atendimento', 'count'),
                cms=('val_total_items', 'mean')
            ).reset_index()

            cms_por_tempo = cms_por_tempo.dropna(subset=['faixa_tempo_chegada'])

            cms_por_tempo['cms'] = cms_por_tempo['cms'].fillna(0)

            if not cms_por_tempo.empty:
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

    st.markdown("---")
    st.header("Análise de Ofensores de CMS por Prestador")
    st.markdown("Identifique prestadores com CMS acima da **média de sua UF e segmento**, e calcule o potencial de economia.")

    segments_for_analysis = ['AUTO', 'RESID', 'VIDA']
    df_financeiro_analise = df[df['segmento'].isin(segments_for_analysis)].copy()

    if df_financeiro_analise.empty:
        st.info("Nenhum dado disponível para os segmentos AUTO, RESID ou VIDA com os filtros selecionados.")
    else:
        cms_medio_uf_segmento_df = df_financeiro_analise.groupby(['uf', 'segmento'], observed=True).agg(
            cms_medio_uf_segmento=('val_total_items', 'mean')
        ).reset_index()

        cms_ofensores = df_financeiro_analise.groupby(['nome_do_prestador', 'uf', 'segmento'], observed=True).agg(
            qtd_servicos=('protocolo_atendimento', 'count'),
            cms_prestador=('val_total_items', 'mean')
        ).reset_index()

        cms_ofensores = cms_ofensores[cms_ofensores['qtd_servicos'] >= min_servicos_prestador]

        cms_ofensores = pd.merge(cms_ofensores, cms_medio_uf_segmento_df, on=['uf', 'segmento'], how='left')

        cms_ofensores['is_ofensor'] = (cms_ofensores['cms_prestador'] > cms_ofensores['cms_medio_uf_segmento'] * 1.10)
        
        cms_ofensores['potencial_economia_rs'] = np.where(
            cms_ofensores['is_ofensor'],
            (cms_ofensores['cms_prestador'] - cms_ofensores['cms_medio_uf_segmento']) * cms_ofensores['qtd_servicos'],
            0
        )

        cms_ofensores['cms_prestador'] = cms_ofensores['cms_prestador'].fillna(0)
        cms_ofensores['cms_medio_uf_segmento'] = cms_ofensores['cms_medio_uf_segmento'].fillna(0)
        cms_ofensores['potencial_economia_rs'] = cms_ofensores['potencial_economia_rs'].fillna(0)

        cms_ofensores_display = cms_ofensores.rename(columns={
            'nome_do_prestador': 'Prestador',
            'uf': 'UF',
            'segmento': 'Segmento',
            'qtd_servicos': 'Qtd. Serviços',
            'cms_prestador': 'CMS do Prestador',
            'cms_medio_uf_segmento': 'CMS Médio UF/Segmento',
            'is_ofensor': 'Ofensor?',
            'potencial_economia_rs': 'Potencial de Economia (R$)'
        })

        prestadores_para_excluir = ['VAZIO', 'MOVIDA', 'LOCALIZA RENT A CAR']
        
        cms_ofensores_display = cms_ofensores_display[
            ~cms_ofensores_display['Prestador'].isin(prestadores_para_excluir)
        ]

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

def page_qualidade(df_final_filtrado, df_nps_cidade_full, df_nps_prestador):
    st.title("✨ Qualidade e NPS")
    st.markdown("Esta seção exibe a evolução do Net Promoter Score (NPS) e os rankings de qualidade por cidade e prestador.")

    # Função para formatar números no padrão PT-BR
    def format_pt_br(value, precision=0):
        if pd.isna(value):
            return "N/A"
        if precision == 0:
            formatted = f"{value:,.0f}"
        else:
            formatted = f"{value:,.{precision}f}"
        # Inverte vírgula e ponto para padrão PT-BR
        return formatted.replace(",", "X").replace(".", ",").replace("X", ".")

    # Exemplo usando df_nps_cidade_full para evolução do NPS
    if not df_nps_cidade_full.empty:
        # Garante que 'mes_ano' seja datetime para o gráfico
        if pd.api.types.is_period_dtype(df_nps_cidade_full['mes_ano']):
            df_nps_cidade_full['mes_ano_dt'] = df_nps_cidade_full['mes_ano'].dt.to_timestamp()
        else:
            df_nps_cidade_full['mes_ano_dt'] = pd.to_datetime(df_nps_cidade_full['mes_ano'], errors='coerce')


        df_nps_evolucao = df_nps_cidade_full.groupby('mes_ano_dt').agg(
            promotores=('nps_promotores', 'sum'),
            detratores=('nps_detratores', 'sum'),
            neutros=('nps_neutros', 'sum')
        ).reset_index()

        df_nps_evolucao['total_responses'] = df_nps_evolucao['promotores'] + df_nps_evolucao['detratores'] + df_nps_evolucao['neutros']
        df_nps_evolucao['nps_score'] = np.where(
            df_nps_evolucao['total_responses'] > 0,
            ((df_nps_evolucao['promotores'] - df_nps_evolucao['detratores']) / df_nps_evolucao['total_responses']) * 100,
            np.nan
        )
        df_nps_evolucao = df_nps_evolucao.dropna(subset=['nps_score']).sort_values('mes_ano_dt')

        st.markdown("---")
        st.subheader("Evolução do Net Promoter Score (NPS) Geral")
        if not df_nps_evolucao.empty:
            fig_nps = px.line(
                df_nps_evolucao,
                x='mes_ano_dt',
                y='nps_score',
                title='Evolução Mensal do NPS Geral',
                labels={'mes_ano_dt': 'Mês/Ano', 'nps_score': 'NPS'},
                markers=True
            )
            fig_nps.update_xaxes(dtick="M1", tickformat="%b\n%Y")
            fig_nps.update_yaxes(range=[-100, 100])
            st.plotly_chart(fig_nps, use_container_width=True)
        else:
            st.info("Nenhum dado disponível para a evolução mensal do NPS.")
    else:
        st.info("Nenhum dado de NPS por cidade disponível para calcular a evolução do NPS geral.")
    
    # --- Ranking NPS por Cidade (Top 10 e Piores 10) ---
    st.markdown("---")
    st.subheader("NPS por Cidade")
    st.info("Visualize as cidades com melhor e pior desempenho no NPS. Isso pode indicar onde focar esforços de melhoria de serviço.")

    if df_nps_cidade_full.empty:
        st.warning("Nenhum dado de NPS por cidade disponível para esta análise. Verifique o arquivo 'processed_nps_by_city.parquet'.")
    else:
        # Agregamos o NPS por cidade, considerando as avaliações totais para relevância
        df_nps_cidade_agg = df_nps_cidade_full.groupby('municipio', observed=True).agg(
            promotores=('nps_promotores', 'sum'),
            detratores=('nps_detratores', 'sum'),
            neutros=('nps_neutros', 'sum')
        ).reset_index()

        df_nps_cidade_agg['total_avaliacoes'] = df_nps_cidade_agg['promotores'] + df_nps_cidade_agg['detratores'] + df_nps_cidade_agg['neutros']
        df_nps_cidade_agg['nps_score'] = np.where(
            df_nps_cidade_agg['total_avaliacoes'] > 0,
            ((df_nps_cidade_agg['promotores'] - df_nps_cidade_agg['detratores']) / df_nps_cidade_agg['total_avaliacoes']) * 100,
            np.nan
        )
        
        df_nps_cidade_agg = df_nps_cidade_agg.dropna(subset=['nps_score'])

        min_avaliacoes_cidade = st.slider("Mínimo de Avaliações para Cidades", min_value=1, max_value=50, value=10, key="min_eval_city_nps_table")
        df_nps_cidade_filtered = df_nps_cidade_agg[df_nps_cidade_agg['total_avaliacoes'] >= min_avaliacoes_cidade]

        if not df_nps_cidade_filtered.empty:
            col_nps_best_city, col_nps_worst_city = st.columns(2)

            with col_nps_best_city:
                st.markdown("#### Top 10 Cidades (Melhor NPS)")
                st.dataframe(
                    df_nps_cidade_filtered.sort_values('nps_score', ascending=False).head(10)
                    .rename(columns={'municipio': 'Cidade', 'nps_score': 'NPS Médio', 'total_avaliacoes': 'Qtd. Avaliações'}).style.format({
                        'NPS Médio': lambda x: format_pt_br(x, 1),
                        'Qtd. Avaliações': lambda x: format_pt_br(x, 0)
                    }),
                    use_container_width=True
                )
            with col_nps_worst_city:
                st.markdown("#### Top 10 Cidades (Pior NPS)")
                st.dataframe(
                    df_nps_cidade_filtered.sort_values('nps_score', ascending=True).head(10) # Piores são os primeiros na ordem crescente
                    .rename(columns={'municipio': 'Cidade', 'nps_score': 'NPS Médio', 'total_avaliacoes': 'Qtd. Avaliações'}).style.format({
                        'NPS Médio': lambda x: format_pt_br(x, 1),
                        'Qtd. Avaliações': lambda x: format_pt_br(x, 0)
                    }),
                    use_container_width=True
                )
        else:
            st.info(f"Nenhuma cidade encontrada com NPS calculado e pelo menos {min_avaliacoes_cidade} avaliações.")
        st.markdown("**Sugestão:** Implementar programas de incentivo ou treinamento nas cidades com baixo NPS, e replicar as melhores práticas das cidades com alto NPS.")


    # --- Ranking NPS por Prestador (Top 10 e Piores 10) ---
    st.markdown("---")
    st.subheader("NPS por Prestador")
    st.info("Identifique os prestadores com melhor e pior performance no NPS. Use esta informação para reconhecimento ou para planos de desenvolvimento.")

    if df_nps_prestador.empty:
        st.warning("Nenhum dado de NPS por prestador disponível para esta análise. Verifique o arquivo 'processed_nps_by_provider.parquet'.")
    else:
        # Agregamos o NPS por prestador, considerando as avaliações totais para relevância
        df_nps_prestador_agg = df_nps_prestador.groupby('nome_do_prestador', observed=True).agg(
            promotores=('nps_promotores', 'sum'),
            detratores=('nps_detratores', 'sum'),
            neutros=('nps_neutros', 'sum')
        ).reset_index()

        df_nps_prestador_agg['total_avaliacoes'] = df_nps_prestador_agg['promotores'] + df_nps_prestador_agg['detratores'] + df_nps_prestador_agg['neutros']
        df_nps_prestador_agg['nps_score'] = np.where(
            df_nps_prestador_agg['total_avaliacoes'] > 0,
            ((df_nps_prestador_agg['promotores'] - df_nps_prestador_agg['detratores']) / df_nps_prestador_agg['total_avaliacoes']) * 100,
            np.nan
        )

        df_nps_prestador_agg = df_nps_prestador_agg.dropna(subset=['nps_score'])

        min_avaliacoes_prestador = st.slider("Mínimo de Avaliações para Prestadores", min_value=1, max_value=50, value=10, key="min_eval_provider_nps_table")
        df_nps_prestador_filtered = df_nps_prestador_agg[df_nps_prestador_agg['total_avaliacoes'] >= min_avaliacoes_prestador]

        if not df_nps_prestador_filtered.empty:
            col_nps_best_prestador, col_nps_worst_prestador = st.columns(2)

            with col_nps_best_prestador:
                st.markdown("#### Top 10 Prestadores (Melhor NPS)")
                st.dataframe(
                    df_nps_prestador_filtered.sort_values('nps_score', ascending=False).head(10)
                    .rename(columns={'nome_do_prestador': 'Prestador', 'nps_score': 'NPS Médio', 'total_avaliacoes': 'Qtd. Avaliações'}).style.format({
                        'NPS Médio': lambda x: format_pt_br(x, 1),
                        'Qtd. Avaliações': lambda x: format_pt_br(x, 0)
                    }),
                    use_container_width=True
                )
            with col_nps_worst_prestador:
                st.markdown("#### Top 10 Prestadores (Pior NPS)")
                st.dataframe(
                    df_nps_prestador_filtered.sort_values('nps_score', ascending=True).head(10) # Piores são os primeiros na ordem crescente
                    .rename(columns={'nome_do_prestador': 'Prestador', 'nps_score': 'NPS Médio', 'total_avaliacoes': 'Qtd. Avaliações'}).style.format({
                        'NPS Médio': lambda x: format_pt_br(x, 1),
                        'Qtd. Avaliações': lambda x: format_pt_br(x, 0)
                    }),
                    use_container_width=True
                )
        else:
            st.info(f"Nenhum prestador encontrado com NPS calculado e pelo menos {min_avaliacoes_prestador} avaliações.")
        st.markdown("**Sugestão:** Avaliar treinamentos específicos ou programas de mentoria para prestadores com baixo NPS. Reconhecer e aprender com os de alto desempenho.")

    # --- Distribuição do TMC (utiliza o df_final_filtrado) ---
    st.markdown("---")
    st.header("Distribuição do TMC por Segmento e Seguradora")

    if df_final_filtrado.empty or 'tempo_chegada_min' not in df_final_filtrado.columns:
        st.warning("Dados de atendimentos ou coluna 'tempo_chegada_min' não disponíveis para a análise de TMC.")
    else:
        col_tmc_segmento, col_tmc_seguradora = st.columns(2)

        with col_tmc_segmento:
            st.subheader("TMC por Segmento")
            tmc_por_segmento = df_final_filtrado.groupby('segmento', observed=True)['tempo_chegada_min'].mean().reset_index()
            tmc_por_segmento['tempo_chegada_min'] = tmc_por_segmento['tempo_chegada_min'].fillna(0)
            st.dataframe(
                tmc_por_segmento.rename(columns={'tempo_chegada_min': 'TMC Médio (min)'}).style.format(
                    {'TMC Médio (min)': lambda x: format_pt_br(x, 0)}
                ),
                use_container_width=True
            )

        with col_tmc_seguradora:
            st.subheader("TMC por Seguradora")
            tmc_por_seguradora = df_final_filtrado.groupby('seguradora', observed=True)['tempo_chegada_min'].mean().reset_index()
            tmc_por_seguradora['tempo_chegada_min'] = tmc_por_seguradora['tempo_chegada_min'].fillna(0)
            st.dataframe(
                tmc_por_seguradora.rename(columns={'tempo_chegada_min': 'TMC Médio (min)'}).style.format(
                    {'TMC Médio (min)': lambda x: format_pt_br(x, 0)}
                ),
                use_container_width=True
            )

def calculate_prestador_score(df):
    if df.empty:
        df['score_prestador'] = []
        df['status_score'] = []
        return df

    # Preenchimento de NaNs para evitar erros, usando a mediana
    for col in ['media_nps', 'media_tempo_chegada', 'total_atendimentos', 'pct_reembolso', 'pct_intermediacao']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Normalização das métricas
    max_atendimentos = df['total_atendimentos'].max() if df['total_atendimentos'].max() > 0 else 1
    max_tempo_chegada = df['media_tempo_chegada'].max() if df['media_tempo_chegada'].max() > 0 else 1
    max_pct_reembolso = df['pct_reembolso'].max() if df['pct_reembolso'].max() > 0 else 1
    max_pct_intermediacao = df['pct_intermediacao'].max() if df['pct_intermediacao'].max() > 0 else 1
    
    pesos = {'atendimentos': 0.25, 'nps': 0.30, 'tempo_chegada': 0.20, 'reembolso': 0.15, 'intermediacao': 0.10}

    df['score_prestador'] = (
        (df['total_atendimentos'] / max_atendimentos * pesos['atendimentos']) +
        (df['media_nps'] / 100 * pesos['nps']) +
        (1 - (df['media_tempo_chegada'] / max_tempo_chegada)) * pesos['tempo_chegada'] +
        (1 - (df['pct_reembolso'] / max_pct_reembolso)) * pesos['reembolso'] +
        (1 - (df['pct_intermediacao'] / max_pct_intermediacao)) * pesos['intermediacao']
    )
    
    min_score, max_score = df['score_prestador'].min(), df['score_prestador'].max()
    if max_score > min_score:
        df['score_prestador'] = (df['score_prestador'] - min_score) / (max_score - min_score) * 100
    else:
        df['score_prestador'] = 50

    if len(df['score_prestador'].unique()) > 1:
        df['status_score'] = pd.qcut(df['score_prestador'], 4, labels=['Precisa de Atenção', 'Regular', 'Bom', 'Excelente'], duplicates='drop')
    else:
        df['status_score'] = 'Regular'
        
    return df

# --- FUNÇÃO DE SUGESTÕES (Texto puro) ---
def get_prestador_sugestao_acao(row, df_completo):
    """
    Gera sugestões de ação simples, em texto puro.
    """
    suggestions = []
    # Usar quartis para limiares dinâmicos
    q75_reembolso = df_completo['pct_reembolso'].quantile(0.75)
    q75_intermediacao = df_completo['pct_intermediacao'].quantile(0.75)
    q75_tmc = df_completo['media_tempo_chegada'].quantile(0.75)
    q25_nps = df_completo['media_nps'].quantile(0.25)

    if row['status_score'] == 'Precisa de Atenção':
        suggestions.append('Performance geral crítica. Avaliar treinamento ou revisão de contrato.')
    
    if row['media_nps'] <= q25_nps and row['media_nps'] < 75:
        suggestions.append('NPS baixo. Investigar causas de insatisfação do cliente.')
    
    if row['pct_reembolso'] > q75_reembolso and row['pct_reembolso'] > 0:
        suggestions.append('Alto percentual de reembolso. Rever processos ou precificação.')
    
    if row['pct_intermediacao'] > q75_intermediacao and row['pct_intermediacao'] > 0:
        suggestions.append('Alto percentual de intermediação. Aumentar capacidade ou eficiência.')

    if row['media_tempo_chegada'] > q75_tmc:
        suggestions.append('Tempo médio de chegada elevado. Otimizar logística ou realocação.')
        
    return "; ".join(suggestions) if suggestions else "Bom desempenho. Nenhuma ação crítica necessária."

# --- PÁGINA PRINCIPAL DO STREAMLIT (VERSÃO FINAL AJUSTADA) ---
def page_score_prestador(df_atendimentos_filtrado, df_nps_prestador):
    st.title("Score de Performance do Prestador")
    st.markdown("Análise de performance da rede de prestadores com foco em KPIs e ações corretivas.")

    if df_atendimentos_filtrado.empty:
        st.info("Nenhum dado de atendimento disponível para os filtros selecionados.")
        return
    
    # --- FILTRO DE ANÁLISE NA PÁGINA PRINCIPAL ---
    st.markdown("---")
    min_atendimentos = st.number_input(
        label="Defina o Mínimo de Atendimentos por Prestador", 
        min_value=1, 
        value=5, 
        step=1,
        help="Apenas prestadores com um número de atendimentos igual ou superior a este valor serão exibidos na análise."
    )
    
    # --- PROCESSAMENTO DE DADOS ---

    df_atendimentos_filtrado['nome_do_prestador'] = df_atendimentos_filtrado['nome_do_prestador'].astype(str)
    if not df_nps_prestador.empty and 'nome_do_prestador' in df_nps_prestador.columns:
        df_nps_prestador['nome_do_prestador'] = df_nps_prestador['nome_do_prestador'].astype(str)
        df_merged = pd.merge(df_atendimentos_filtrado, df_nps_prestador[['nome_do_prestador', 'nps_score_calculado']], on='nome_do_prestador', how='left')
        df_merged['nps_score_calculado'] = df_merged['nps_score_calculado'].fillna(0)
    else:
        df_merged = df_atendimentos_filtrado.copy()
        df_merged['nps_score_calculado'] = 0

    df_prestadores_agg = df_merged.groupby('nome_do_prestador', observed=True).agg(
        total_atendimentos=('protocolo_atendimento', 'nunique'),
        media_nps=('nps_score_calculado', 'mean'),
        num_reembolsos=('is_reembolso', 'sum'),
        num_intermediacoes=('is_intermediacao', 'sum'),
        media_tempo_chegada=('tempo_chegada_min', 'mean')
    ).reset_index()


    # APLICAÇÃO DO FILTRO DE MÍNIMO DE ATENDIMENTOS
    df_prestadores_filtrado = df_prestadores_agg[df_prestadores_agg['total_atendimentos'] >= min_atendimentos].copy()

    if df_prestadores_filtrado.empty:
        st.warning(f"Nenhum prestador encontrado com {min_atendimentos} ou mais atendimentos para os filtros aplicados.")
        return

    df_prestadores_filtrado['pct_reembolso'] = (df_prestadores_filtrado['num_reembolsos'] / df_prestadores_filtrado['total_atendimentos'] * 100).fillna(0)
    df_prestadores_filtrado['pct_intermediacao'] = (df_prestadores_filtrado['num_intermediacoes'] / df_prestadores_filtrado['total_atendimentos'] * 100).fillna(0)
    
    df_prestadores_scored = calculate_prestador_score(df_prestadores_filtrado)
    df_prestadores_scored['sugestao_acao'] = df_prestadores_scored.apply(lambda row: get_prestador_sugestao_acao(row, df_prestadores_scored), axis=1)

    # --- KPIs GERAIS DA REDE ---
    st.markdown("---")
    st.subheader("Desempenho Geral da Rede")
    
    avg_score = df_prestadores_scored['score_prestador'].mean()
    avg_atendimentos = df_prestadores_scored['total_atendimentos'].mean()
    avg_nps = df_prestadores_scored['media_nps'].mean()
    avg_tmc = df_prestadores_scored['media_tempo_chegada'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Score Médio da Rede", f"{avg_score:.2f}")
    col2.metric("Média de Atendimentos", f"{avg_atendimentos:.1f}")
    col3.metric("NPS Médio", f"{avg_nps:.1f}")
    col4.metric("TMC Médio (min)", f"{avg_tmc:.0f}")

    # --- RANKING COMPLETO DE PRESTADORES ---
    st.markdown("---")
    st.subheader("Ranking Completo de Prestadores")
    st.markdown("Análise detalhada de todos os prestadores que atendem aos critérios de filtro. Use as setas para ordenar.")

    st.dataframe(
        df_prestadores_scored.rename(columns={
            'nome_do_prestador': 'Prestador',
            'total_atendimentos': 'Atendimentos',
            'media_nps': 'NPS Médio',
            'media_tempo_chegada': 'TMC Médio (min)',
            'pct_reembolso': '% Reembolso',
            'pct_intermediacao': '% Intermediação',
            'score_prestador': 'Score',
            'status_score': 'Status'
        })[[
            'Prestador', 'Score', 'Status', 'Atendimentos', 'NPS Médio', 'TMC Médio (min)',
            '% Reembolso', '% Intermediação'
        ]]
        .sort_values('Score', ascending=True)
        .style
        .background_gradient(cmap='RdYlGn', subset=['Score'])
        .bar(subset=['Atendimentos'], color='#1f77b4')
        .format({
            'Score': '{:.2f}',
            'NPS Médio': '{:.2f}',
            'TMC Médio (min)': '{:.0f}',
            '% Reembolso': '{:.2f}%',
            '% Intermediação': '{:.2f}%',
        }),
        use_container_width=True,
        height=600
    )
    
    # DOWNLOAD DOS DADOS
    output_all = io.BytesIO()
    with pd.ExcelWriter(output_all, engine='xlsxwriter') as writer:
        df_prestadores_scored.to_excel(writer, index=False, sheet_name='Score_Prestadores_Completo')
    output_all.seek(0)
    st.download_button(
        label="Baixar Ranking Completo (XLSX)",
        data=output_all,
        file_name='score_prestadores_completo.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    # --- PLANO DE AÇÃO EM CARDS (APÓS O RANKING) ---
    st.markdown("---")
    st.subheader("Plano de Ação: Foco nos Principais Pontos de Melhoria")
    st.info("Recomendações para os 20 prestadores com os menores scores para direcionamento de ações.")
    
    df_offenders = df_prestadores_scored[
        df_prestadores_scored['status_score'].isin(['Precisa de Atenção', 'Regular'])
    ].sort_values('score_prestador', ascending=True).head(20)

    if not df_offenders.empty:
        for index, row in df_offenders.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Prestador:** {row['nome_do_prestador']}")
                    st.markdown(f"**Sugestões:** {row['sugestao_acao']}")
                with col2:
                    st.metric(label="Score", value=f"{row['score_prestador']:.2f}")
                    # Usando st.markdown para o status para evitar o delta/seta
                    st.markdown(f"**Status:** <span style='color: #d62728;'>{row['status_score']}</span>", unsafe_allow_html=True)
                
    else:
        st.success("🎉 Nenhum prestador com status 'Regular' ou 'Precisa de Atenção' encontrado. Ótimo resultado!")

    # --- EXPANDER COM A METODOLOGIA ---
    with st.expander("💡 Entenda a Metodologia do Score"):
        st.markdown(r"""
        O **Score do Prestador** é um índice de 0 a 100 que consolida múltiplos KPIs para avaliar a performance.

        - **Componentes e Pesos:**
          - **Total de Atendimentos (25%):** Maior volume é positivo.
          - **NPS Médio (30%):** Satisfação do cliente é crucial.
          - **Tempo Médio de Chegada (TMC) (20%):** Menor tempo é melhor.
          - **Percentual de Reembolso (15%):** Menor percentual é melhor.
          - **Percentual de Intermediação (10%):** Menor percentual é melhor.

        **Fórmula Simplificada:**
        $$ \text{Score} = f(\text{Atendimentos}, \text{NPS}, \text{TMC}, \text{Reembolso}, \text{Intermediação}) $$
        """)

def page_qualidade_nps(df_atendimentos_filtrado, df_nps_cidade_full, df_nps_prestador):
    st.title("Qualidade")
    st.markdown("Esta seção exibe a evolução do Net Promoter Score (NPS), o Tempo Médio de Chegada do Prestador e os rankings de qualidade por cidade e prestador.")

    def format_pt_br(value, precision=0):
        if pd.isna(value):
            return "N/A"
        if precision == 0:
            formatted = f"{value:,.0f}"
        else:
            formatted = f"{value:,.{precision}f}"
        return formatted.replace(",", "X").replace(".", ",").replace("X", ".")

    if not df_nps_cidade_full.empty:
        if pd.api.types.is_period_dtype(df_nps_cidade_full['mes_ano']):
            df_nps_cidade_full['mes_ano_dt'] = df_nps_cidade_full['mes_ano'].dt.to_timestamp()
        else:
            df_nps_cidade_full['mes_ano_dt'] = pd.to_datetime(df_nps_cidade_full['mes_ano'], errors='coerce')


        df_nps_evolucao = df_nps_cidade_full.groupby('mes_ano_dt').agg(
            promotores=('nps_promotores', 'sum'),
            detratores=('nps_detratores', 'sum'),
            neutros=('nps_neutros', 'sum')
        ).reset_index()

        df_nps_evolucao['total_responses'] = df_nps_evolucao['promotores'] + df_nps_evolucao['detratores'] + df_nps_evolucao['neutros']
        df_nps_evolucao['nps_score'] = np.where(
            df_nps_evolucao['total_responses'] > 0,
            ((df_nps_evolucao['promotores'] - df_nps_evolucao['detratores']) / df_nps_evolucao['total_responses']) * 100,
            np.nan
        )
        df_nps_evolucao = df_nps_evolucao.dropna(subset=['nps_score']).sort_values('mes_ano_dt')

        st.markdown("---")
        st.subheader("Evolução do Net Promoter Score (NPS) Geral")
        if not df_nps_evolucao.empty:
            fig_nps = px.line(
                df_nps_evolucao,
                x='mes_ano_dt',
                y='nps_score',
                title='Evolução Mensal do NPS Geral',
                labels={'mes_ano_dt': 'Mês/Ano', 'nps_score': 'NPS'},
                markers=True
            )
            fig_nps.update_xaxes(dtick="M1", tickformat="%b\n%Y")
            fig_nps.update_yaxes(range=[-100, 100])
            st.plotly_chart(fig_nps, use_container_width=True)
        else:
            st.info("Nenhum dado disponível para a evolução mensal do NPS.")
    else:
        st.info("Nenhum dado de NPS por cidade disponível para calcular a evolução do NPS geral.")
    
    st.markdown("---")
    st.subheader("NPS por Cidade")
    st.info("Visualize as cidades com melhor e pior desempenho no NPS. Isso pode indicar onde focar esforços de melhoria de serviço.")

    if df_nps_cidade_full.empty:
        st.warning("Nenhum dado de NPS por cidade disponível para esta análise. Verifique o arquivo 'processed_nps_by_city.parquet'.")
    else:
        df_nps_cidade_agg = df_nps_cidade_full.groupby('municipio', observed=True).agg(
            promotores=('nps_promotores', 'sum'),
            detratores=('nps_detratores', 'sum'),
            neutros=('nps_neutros', 'sum')
        ).reset_index()

        df_nps_cidade_agg['total_avaliacoes'] = df_nps_cidade_agg['promotores'] + df_nps_cidade_agg['detratores'] + df_nps_cidade_agg['neutros']
        df_nps_cidade_agg['nps_score'] = np.where(
            df_nps_cidade_agg['total_avaliacoes'] > 0,
            ((df_nps_cidade_agg['promotores'] - df_nps_cidade_agg['detratores']) / df_nps_cidade_agg['total_avaliacoes']) * 100,
            np.nan
        )
        
        df_nps_cidade_agg = df_nps_cidade_agg.dropna(subset=['nps_score'])

        min_avaliacoes_cidade = st.slider("Mínimo de Avaliações para Cidades", min_value=1, max_value=50, value=10, key="min_eval_city_nps_table")
        df_nps_cidade_filtered = df_nps_cidade_agg[df_nps_cidade_agg['total_avaliacoes'] >= min_avaliacoes_cidade]

        if not df_nps_cidade_filtered.empty:
            col_nps_best_city, col_nps_worst_city = st.columns(2)

            with col_nps_best_city:
                st.markdown("#### Top 10 Cidades (Melhor NPS)")
                st.dataframe(
                    df_nps_cidade_filtered.sort_values('nps_score', ascending=False).head(10)
                    .rename(columns={'municipio': 'Cidade', 'nps_score': 'NPS Médio', 'total_avaliacoes': 'Qtd. Avaliações'}).style.format({
                        'NPS Médio': lambda x: format_pt_br(x, 1),
                        'Qtd. Avaliações': lambda x: format_pt_br(x, 0)
                    }),
                    use_container_width=True
                )
            with col_nps_worst_city:
                st.markdown("#### Top 10 Cidades (Pior NPS)")
                st.dataframe(
                    df_nps_cidade_filtered.sort_values('nps_score', ascending=True).head(10)
                    .rename(columns={'municipio': 'Cidade', 'nps_score': 'NPS Médio', 'total_avaliacoes': 'Qtd. Avaliações'}).style.format({
                        'NPS Médio': lambda x: format_pt_br(x, 1),
                        'Qtd. Avaliações': lambda x: format_pt_br(x, 0)
                    }),
                    use_container_width=True
                )
        else:
            st.info(f"Nenhuma cidade encontrada com NPS calculado e pelo menos {min_avaliacoes_cidade} avaliações.")
        st.markdown("**Sugestão:** Implementar programas de incentivo ou treinamento nas cidades com baixo NPS, e replicar as melhores práticas das cidades com alto NPS.")


    st.markdown("---")
    st.subheader("NPS por Prestador")
    st.info("Identifique os prestadores com melhor e pior performance no NPS. Use esta informação para reconhecimento ou para planos de desenvolvimento.")

    if df_nps_prestador.empty:
        st.warning("Nenhum dado de NPS por prestador disponível para esta análise. Verifique o arquivo 'processed_nps_by_provider.parquet'.")
    else:
        df_nps_prestador_agg = df_nps_prestador.groupby('nome_do_prestador', observed=True).agg(
            promotores=('nps_promotores', 'sum'),
            detratores=('nps_detratores', 'sum'),
            neutros=('nps_neutros', 'sum')
        ).reset_index()

        df_nps_prestador_agg['total_avaliacoes'] = df_nps_prestador_agg['promotores'] + df_nps_prestador_agg['detratores'] + df_nps_prestador_agg['neutros']
        df_nps_prestador_agg['nps_score'] = np.where(
            df_nps_prestador_agg['total_avaliacoes'] > 0,
            ((df_nps_prestador_agg['promotores'] - df_nps_prestador_agg['detratores']) / df_nps_prestador_agg['total_avaliacoes']) * 100,
            np.nan
        )

        df_nps_prestador_agg = df_nps_prestador_agg.dropna(subset=['nps_score'])

        min_avaliacoes_prestador = st.slider("Mínimo de Avaliações para Prestadores", min_value=1, max_value=50, value=10, key="min_eval_provider_nps_table")
        df_nps_prestador_filtered = df_nps_prestador_agg[df_nps_prestador_agg['total_avaliacoes'] >= min_avaliacoes_prestador]

        if not df_nps_prestador_filtered.empty:
            col_nps_best_prestador, col_nps_worst_prestador = st.columns(2)

            with col_nps_best_prestador:
                st.markdown("#### Top 10 Prestadores (Melhor NPS)")
                st.dataframe(
                    df_nps_prestador_filtered.sort_values('nps_score', ascending=False).head(10)
                    .rename(columns={'nome_do_prestador': 'Prestador', 'nps_score': 'NPS Médio', 'total_avaliacoes': 'Qtd. Avaliações'}).style.format({
                        'NPS Médio': lambda x: format_pt_br(x, 1),
                        'Qtd. Avaliações': lambda x: format_pt_br(x, 0)
                    }),
                    use_container_width=True
                )
            with col_nps_worst_prestador:
                st.markdown("#### Top 10 Prestadores (Pior NPS)")
                st.dataframe(
                    df_nps_prestador_filtered.sort_values('nps_score', ascending=True).head(10)
                    .rename(columns={'nome_do_prestador': 'Prestador', 'nps_score': 'NPS Médio', 'total_avaliacoes': 'Qtd. Avaliações'}).style.format({
                        'NPS Médio': lambda x: format_pt_br(x, 1),
                        'Qtd. Avaliações': lambda x: format_pt_br(x, 0)
                    }),
                    use_container_width=True
                )
        else:
            st.info(f"Nenhum prestador encontrado com NPS calculado e pelo menos {min_avaliacoes_prestador} avaliações.")
        st.markdown("**Sugestão:** Avaliar treinamentos específicos ou programas de mentoria para prestadores com baixo NPS. Reconhecer e aprender com os de alto desempenho.")

    st.markdown("---")
    st.header("Distribuição do TMC por Segmento e Seguradora")

    if df_atendimentos_filtrado.empty or 'tempo_chegada_min' not in df_atendimentos_filtrado.columns:
        st.warning("Dados de atendimentos ou coluna 'tempo_chegada_min' não disponíveis para a análise de TMC.")
    else:
        col_tmc_segmento, col_tmc_seguradora = st.columns(2)

        with col_tmc_segmento:
            st.subheader("TMC por Segmento")
            tmc_por_segmento = df_atendimentos_filtrado.groupby('segmento', observed=True)['tempo_chegada_min'].mean().reset_index()
            tmc_por_segmento['tempo_chegada_min'] = tmc_por_segmento['tempo_chegada_min'].fillna(0)
            st.dataframe(
                tmc_por_segmento.rename(columns={'tempo_chegada_min': 'TMC Médio (min)'}).style.format(
                    {'TMC Médio (min)': lambda x: format_pt_br(x, 0)}
                ),
                use_container_width=True
            )

        with col_tmc_seguradora:
            st.subheader("TMC por Seguradora")
            tmc_por_seguradora = df_atendimentos_filtrado.groupby('seguradora', observed=True)['tempo_chegada_min'].mean().reset_index()
            tmc_por_seguradora['tempo_chegada_min'] = tmc_por_seguradora['tempo_chegada_min'].fillna(0)
            st.dataframe(
                tmc_por_seguradora.rename(columns={'tempo_chegada_min': 'TMC Médio (min)'}).style.format(
                    {'TMC Médio (min)': lambda x: format_pt_br(x, 0)}
                ),
                use_container_width=True
            )

def main():
    # Inicializa o estado de login
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login_page()
    else:
        # Carrega os dados em cache
        with st.spinner("Carregando e processando dados..."):
            df_atendimentos_full, df_nps_cidade_full, df_nps_prestador = load_and_prepare_data(
                ATENDIMENTO_FILE_PATH, NPS_CIDADE_PATH, NPS_PRESTADOR_PATH
            )
        
        if df_atendimentos_full.empty:
            st.error("Nenhum dado de atendimentos válido disponível. Verifique o arquivo de origem.")
            st.stop()

        # --- BARRA LATERAL ESTRUTURADA ---
        with st.sidebar:
            st.markdown("<h1 style='text-align: center;'>Score do Prestador</h1>", unsafe_allow_html=True)
            # 1. CABEÇALHO COM LOGO E TÍTULO
            try:
                st.image(LOGO_PATH)
                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            except FileNotFoundError:
                st.warning("Arquivo 'logo.png' não encontrado. Coloque-o na mesma pasta do script.")
            

            # 2. MENU DE NAVEGAÇÃO PRINCIPAL
            selected_page = option_menu(
                menu_title=None,
                # Ordem ajustada para manter todas as páginas
                options=["Informações", "Score Prestador", "Capilaridade", "Financeiro", "Qualidade"], 
                # Ícones correspondentes à nova ordem
                icons=["info-circle-fill", "graph-up-arrow", "globe2", "currency-dollar", "award"], 
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#FFFFFF"},
                    "icon": {"color": "#2021D4", "font-size": "20px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#E6F0F8"},
                    "nav-link-selected": {"background-color": "#2021D4", "color": "white", "font-weight": "bold"},
                    "icon-selected": {"color": "white"},
                }
            )


            # 3. FILTROS DE DADOS
            st.markdown("### ⚙️ Filtros de Dados")
            
            segmento_options = [ALL_OPTION] + sorted(df_atendimentos_full['segmento'].dropna().unique().tolist())
            seguradora_options = [ALL_OPTION] + sorted(df_atendimentos_full['seguradora'].dropna().unique().tolist())
            estado_options = [ALL_OPTION] + sorted(df_atendimentos_full['uf'].dropna().unique().tolist())
            
            min_date_data = df_atendimentos_full['data_abertura_atendimento'].min().date()
            max_date_data = df_atendimentos_full['data_abertura_atendimento'].max().date()

            segmento_selecionado = st.multiselect("Segmento", segmento_options, default=[ALL_OPTION])
            if ALL_OPTION in segmento_selecionado:
                segmento_selecionado = df_atendimentos_full['segmento'].dropna().unique().tolist()

            seguradora_selecionada = st.multiselect("Seguradora", seguradora_options, default=[ALL_OPTION])
            if ALL_OPTION in seguradora_selecionada:
                seguradora_selecionada = df_atendimentos_full['seguradora'].dropna().unique().tolist()

            estado_selecionado = st.multiselect("Estado", estado_options, default=[ALL_OPTION])
            if ALL_OPTION in estado_selecionado:
                estado_selecionado = df_atendimentos_full['uf'].dropna().unique().tolist()

            municipio_options = [ALL_OPTION]
            if estado_selecionado and ALL_OPTION not in estado_selecionado:
                municipio_options += sorted(df_atendimentos_full[df_atendimentos_full['uf'].isin(estado_selecionado)]['municipio'].dropna().unique().tolist())
            else:
                municipio_options += sorted(df_atendimentos_full['municipio'].dropna().unique().tolist())
            
            municipio_selecionado = st.multiselect("Cidade", municipio_options, default=[ALL_OPTION])
            if ALL_OPTION in municipio_selecionado:
                municipio_selecionado = df_atendimentos_full['municipio'].dropna().unique().tolist()

            data_inicio, data_fim = st.date_input(
                "Período de Análise",
                value=(min_date_data, max_date_data),
                min_value=min_date_data,
                max_value=max_date_data,
                format="DD/MM/YYYY"
            )

            # 4. RODAPÉ COM DATA DE ATUALIZAÇÃO E BOTÃO SAIR
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            st.caption("Última Atualização: 09/07/2025") 

            if st.button("Sair", use_container_width=True):
                st.session_state['logged_in'] = False
                st.session_state.pop('username', None)
                st.rerun()
            
            

        # --- APLICAÇÃO DOS FILTROS ---
        df_filtrado = apply_filters(
            df_atendimentos_full,
            segmento_selecionado,
            seguradora_selecionada,
            estado_selecionado,
            municipio_selecionado,
            data_inicio,
            data_fim
        )
        
        if df_filtrado.empty:
            st.info("Nenhum dado corresponde aos filtros selecionados.")
        
        # --- RENDERIZAÇÃO DA PÁGINA SELECIONADA (TODAS AS OPÇÕES RESTAURADAS) ---
        if selected_page == "Informações":
            page_informacao()
        elif selected_page == "Score Prestador":
            page_score_prestador(df_filtrado, df_nps_prestador)
        elif selected_page == "Capilaridade":
            page_capilaridade(df_filtrado)
        elif selected_page == "Financeiro":
            page_financeiro(df_filtrado)
        elif selected_page == "Qualidade":
            page_qualidade_nps(df_filtrado, df_nps_cidade_full, df_nps_prestador)


# --- Execução Principal ---
if __name__ == "__main__":
    # Supondo que a função inject_css() é chamada aqui
    # inject_css() 
    main()