import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import textwrap

# Configuration de la page Streamlit avec le mode centré
st.set_page_config(layout="centered", page_title="Diag360 - Visualisation")

# Titre de la page
st.title("Diag360 - Visualisation des besoins territoriaux")

# Fonction pour tronquer et ajouter des retours à la ligne aux textes trop longs sans couper les mots
def format_label(text, max_len=120, line_len=40):
    if len(text) > max_len:
        text = text[:max_len] + '...'
    wrapped_text = textwrap.fill(text, width=line_len, break_long_words=False)
    return wrapped_text.replace('\n', '<br>')

def add_to_radar(df, groupe, s_groupe):
    df_grouped = df.groupby([groupe, s_groupe])["valeur_indice"].mean().reset_index()
    df_grouped["Indice"] = (df_grouped[groupe] != df_grouped[groupe].shift()).cumsum().map(lambda x: chr(64 + x))
    values = df_grouped["valeur_indice"].values

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    labels = df_grouped[s_groupe].values
    values = df_grouped["valeur_indice"].values
    angles = np.linspace(0, 2 * np.pi, len(df_grouped), endpoint=False)
    num_vars = len(labels)

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('RdYlGn')
    colors = cmap(norm(values))

    ax.spines['polar'].set_color('white')
    ax.set_theta_offset(np.pi / 2)
    ax.set_thetagrids(np.degrees(angles), labels)

    ax.set_facecolor('#FAFAFA')

    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('right')
        else:
            label.set_horizontalalignment('left')

    ax.set_axisbelow(False)

    labels = ["\n".join(textwrap.wrap(r[:38] + ('...' if len(r) > 38 else ''), 22, break_long_words=False)) for r in labels]
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelsize=16-0.25*num_vars, labelcolor='#222', grid_color='#555', grid_alpha=0.1, pad=3)
    ax.bar(angles, values, color=colors, alpha=1, width=2*np.pi/num_vars, zorder=1, linewidth=3, edgecolor='white')

    ax.tick_params(axis='y', labelcolor='white', labelsize=0, grid_color='#FFF', grid_alpha=0, width=0)
    ax.set_ylim(0, 1.05)

    st.pyplot(fig)

# Fonction pour tronquer la description
def truncate_text(text, max_length=400):
    return text if len(text) <= max_length else text[:max_length] + ' [...]'


uploaded_file = st.file_uploader("", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name='Export')
    df_besoins = pd.read_excel(uploaded_file, sheet_name="Besoins_Infos")
    df_indicateurs = pd.read_excel(uploaded_file, sheet_name="Indicateurs_Infos")

    choice_enjeux = st.selectbox("Types de besoins :", ['Vitaux', 'Essentiels', 'Induits', 'Vitaux, Essentiels et Induits'])
    choice_objectifs = st.multiselect("Objectifs territoriaux :", ['Transformation', 'Subsistance', 'Soutenabilité', 'Gestion de crise'], default=["Transformation", "Subsistance", "Soutenabilité", "Gestion de crise"])
    choice_plot = st.selectbox("Type de représentation :", ['Panorama 360°', 'Synthèse'])

    if choice_enjeux != 'Vitaux, Essentiels et Induits':
        df = df[df['type_besoins'] == choice_enjeux]

    if choice_objectifs:
        df = df[df['objectif'].isin(choice_objectifs)]

    df = df[pd.to_numeric(df['valeur_indice'], errors='coerce').notnull()]
    df['valeur_indice'] = pd.to_numeric(df['valeur_indice'])
    df = df[df['valeur_indice'] > 0]
    df['uniform_value'] = 1

    st.markdown("## Tour d'horizon de votre résilience territoriale")
    st.markdown(f"### {choice_plot} - {choice_enjeux}")
    st.markdown(f"#### sur objectif de : {', '.join(choice_objectifs)}")

    if choice_plot == 'Panorama 360°':
        path_value = ['type_besoins', 'besoins', 'objectif', 'designation_indicateur']
        colorscale = [[0, '#CA0D00'], [0.5, '#FCF68E'], [1, '#05892F']]
        fig = px.sunburst(df, path=path_value, values='uniform_value', color='valeur_indice', color_continuous_scale=colorscale)
        fig.update_layout(showlegend=False, margin=dict(t=0, l=100, r=0, b=0))
        st.plotly_chart(fig)
    else:
        add_to_radar(df, 'type_besoins', 'besoins')

    html_content = '<hr style="margin: 50px 0 !important;"><div class="ksln-grid">'
    if choice_plot == 'Synthèse':
        if choice_objectifs == 'besoins':
            for index, row in df_besoins.iterrows():
                if row['besoins'] in df['besoins'].values:
                    html_content += f"""
                    <a style="color: black; text-decoration: none;" href="{row['lien']}" target="_blank">
                        <div class="ksln-cards" style="margin: 0px auto auto 3px;">
                            <p style="text-align: center; font-weight: bold; padding: 10px; background-color: #ebebeb;">{row['besoins']}</p>
                                <p style="text-align: center; margin: 10px;">{truncate_text(row['description'])}</p>
                        </div>
                    </a>
                    """
        else:
            for index, row in df_indicateurs.iterrows():
                if row['designation_indicateur'] in df['designation_indicateur'].values:
                    html_content += f"""
                    <a style="color: black; text-decoration: none;" href="{row['lien']}" target="_blank">
                        <div class="ksln-cards" style="margin: 0px auto auto 3px;">
                            <p style="text-align: center; font-weight: bold; padding: 10px; background-color: #ebebeb;">{row['designation_indicateur']}</p>
                                <p style="text-align: center; margin: 10px;">{truncate_text(row['description'])}</p>
                        </div>
                    </a>
                    """
    else:
        for index, row in df_besoins.iterrows():
            if row['besoins'] in df['besoins'].values:
                html_content += f"""
                <a style="color: black; text-decoration: none;" href="{row['lien']}" target="_blank">
                    <div class="ksln-cards" style="margin: 0px auto auto 3px;">
                        <p style="text-align: center; font-weight: bold; padding: 10px; background-color: #ebebeb;">{row['besoins']}</p>
                            <p style="text-align: center; margin: 10px;">{truncate_text(row['description'])}</p>
                    </div>
                </a>
                """

    html_content += '</div>'
    st.html(html_content)
else:
    st.html("""
    <hr>
    <div style="margin-top: 25px; background-color: #FAFAFA; max-width: 100%">
        <div class="ksln-cards" style="margin: 2px; padding: 0px;">
            <h2>Vous débutez avec Diag360 ?</h2>
            <p>
                1. Télécharger ce fichier et suivez les instructions, nous vous retrouvons ici juste après 🤗 :<br>
                <a href="https://github.com/Konsilion/diag360/raw/master/mkdocs/media/Diag360_Indicateurs.xlsx">Télécharger le tableur de données</a>
            </p>
            <p>
                2. Une fois le fichier remplis, cliquer sur le <strong>bouton 'Browse' à gauche</strong>, et sélectionnez votre tableur de travail.
                Pour plus d'information, rendez-vous sur le site suivant :
                <a target="_blank" href="https://konsilion.github.io/diag360/">Documentation Diag360</a>
            </p>
        </div>
    </div>
    """)
