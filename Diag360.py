import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
import textwrap
import requests
from io import BytesIO

st.set_page_config(layout="centered", page_title="Diag360 - Visualisation")
st.title("Diag360 - Visualisation des besoins territoriaux")

def truncate_text(text, max_length=400):
    return text if len(text) <= max_length else text[:max_length] + ' [...]'

def format_label(label, truncate=False, max_len=50, line_len=20):
    if truncate and len(label) > max_len:
        label = label[:max_len] + '...'
    wrapped_text = textwrap.fill(label, width=line_len, break_long_words=False)
    return wrapped_text.replace('\n', '\n')

def add_to_radar(df, groupe, s_groupe, font_size, truncate_labels):
    df_grouped = df.groupby([groupe, s_groupe])["valeur_indice"].mean().reset_index()
    
    labels = df_grouped[s_groupe].values
    values = df_grouped["valeur_indice"].values

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    df_grouped["angle"] = angles
    df_grouped = df_grouped.sort_values(by="angle")
    
    labels = df_grouped[s_groupe].values
    values = df_grouped["valeur_indice"].values
    angles = df_grouped["angle"].values

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

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

    labels = [format_label(label, truncate_labels) for label in labels]

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size, labelcolor='#222', grid_color='#555', grid_alpha=0.1, pad=3)
    ax.bar(angles, values, color=colors, alpha=1, width=2*np.pi/num_vars, zorder=1, linewidth=3, edgecolor='white')

    ax.tick_params(axis='y', labelcolor='white', labelsize=0, grid_color='#FFF', grid_alpha=0, width=0)
    ax.set_ylim(0, 1.05)

    st.pyplot(fig)

    return list(zip(df_grouped[s_groupe], colors))





df = None

with st.expander("Téléchargement du fichier"):

    if df is None:
        st.markdown("""
        <div>
            <h2>Vous débutez avec Diag360 ?</h2>
            <p>
                1. Télécharger ce fichier et suivez les instructions, nous vous retrouvons ici juste après 🤗 :<br>
                <a href="https://github.com/Konsilion/diag360/raw/master/mkdocs/media/Diag360_Indicateurs.xlsx">Télécharger le tableur de données</a>
            </p>
            <p>
                2. Une fois le fichier rempli, collez le lien vers votre fichier Excel hébergé en lien "raw" (GitHub, Gitlab) dans le champ prévu, ou alors téléchargez le fichier depuis votre ordinateur directement.
                <br>--<br>Pour plus d'informations, rendez-vous sur le site suivant :
                <a target="_blank" href="https://konsilion.github.io/diag360/">Documentation Diag360</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    
    # Choix entre lien ou fichier local
    upload_option = st.radio("Choisir la méthode de téléchargement", ["Lien vers le fichier", "Télécharger depuis l'ordinateur"])

    if upload_option == "Lien vers le fichier":
        # Champ pour l'URL
        excel_url = st.text_input(
            "Collez le lien de votre fichier Excel :",
            value="https://github.com/Konsilion/konsilion-drive/raw/refs/heads/main/Diag360/Diag360_Indicateurs%20(1).xlsx"
        )
        df = None
        if excel_url:
            try:
                response = requests.get(excel_url)
                response.raise_for_status()
                file_in_memory = BytesIO(response.content)

                df = pd.read_excel(file_in_memory, sheet_name='Export')
                df_besoins = pd.read_excel(file_in_memory, sheet_name="Besoins_Infos")
                df_indicateurs = pd.read_excel(file_in_memory, sheet_name="Indicateurs_Infos")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")

    elif upload_option == "Télécharger depuis l'ordinateur":
        # Champ pour le téléchargement local
        uploaded_file = st.file_uploader("Choisir un fichier Excel", type=["xlsx"])
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file, sheet_name='Export')
                df_besoins = pd.read_excel(uploaded_file, sheet_name="Besoins_Infos")
                df_indicateurs = pd.read_excel(uploaded_file, sheet_name="Indicateurs_Infos")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")










# 👉 Affichage seulement si le fichier a bien été chargé
if df is not None:
    st.markdown("## Synthèse de l'ensemble des besoins")
    with st.expander("Paramètres graphique"):
        st.markdown("### Graphique général")
        font_size_global = st.slider("Taille de la police des étiquettes (Global)", min_value=5, max_value=12, value=9, key="font_size_global")
        truncate_labels_global = st.checkbox("Tronquer les étiquettes (Global)", value=True, key="truncate_labels_global")
        st.markdown("### Graphique focus besoin")
        font_size_specific = st.slider("Taille de la police des étiquettes (Spécifique)", min_value=5, max_value=12, value=9, key="font_size_specific")
        truncate_labels_specific = st.checkbox("Tronquer les étiquettes (Spécifique)", value=True, key="truncate_labels_specific")

    add_to_radar(df, 'type_besoins', 'besoins', font_size_global, truncate_labels_global)

    st.markdown("## Focus sur un besoin")
    besoins_list = df['besoins'].unique()
    selected_besoin = st.selectbox("", besoins_list)

    if selected_besoin:
        df_selected = df[df['besoins'] == selected_besoin]
        values_dict = dict(zip(df_selected["designation_indicateur"], df_selected["valeur_indice"]))

        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap('RdYlGn')

        ordered_indicators = add_to_radar(df_selected, 'besoins', 'designation_indicateur', font_size_specific, truncate_labels_specific)

        html_content = '<hr style="margin: 50px 0 !important;"><div class="ksln-grid">'
        rotated_ordered_indicators = ordered_indicators[::-1][-1:] + ordered_indicators[::-1][:-1]
        for label, color in rotated_ordered_indicators:
            val = values_dict.get(label, 0)
            if val > 0.9 or val < 0.1:
                ft_color = "#FFF"
            else:
                ft_color = "#000"
                
            if val == 0:
                bg_color = "#D3D3D3"
                ft_color = "#000"
            else:
                bg_color = to_hex(cmap(norm(val)))
            lien = df_indicateurs.loc[df_indicateurs["designation_indicateur"] == label, "lien"].values
            href = lien[0] if len(lien) > 0 else "#"
            html_content += f"""
            <a style="color: {ft_color}; text-decoration: none;" href="https://konsilion.github.io/diag360/" target="_blank">
                <div class="ksln-cards" style="margin: 0px auto auto 3px;">
                    <p style="
                        text-align: left;
                        padding: 10px;
                        background-color: {bg_color};
                        border-radius: 12px;
                        transition: 0.3s ease;
                        font-weight: 500;
                        margin-bottom: 6px;
                    ">
                        {label} ➜
                    </p>
                </div>
            </a>
            """
        html_content += '</div>'
        st.html(html_content)
else:
    st.warning("Aucun fichier n’a été chargé. Veuillez en charger un pour commencer.")
