import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
import textwrap
import requests
from io import BytesIO


st.markdown("## Diag360 - Visualisation des besoins territoriaux")
        
st.html("<br><img src='https://github.com/Konsilion/diag360/blob/master/mkdocs/media/Bandeau_Diag360.png?raw=true' alt='Bandeau de diag360' style='max-width: 100%'>")

def truncate_text(text, max_length=400):
    return text if len(text) <= max_length else text[:max_length] + ' [...]'

def format_label(label, truncate=False, max_len=50, line_len=25):
    if truncate and len(label) > max_len:
        label = label[:max_len] + '...'
    wrapped_text = textwrap.fill(label, width=line_len, break_long_words=False)
    return wrapped_text.replace('\n', '\n')










def add_to_radar(df, groupe, s_groupe, truncate_labels=True, df_ref=None):

    # Calculer la taille de la police en fonction du nombre d'indicateurs
    num_indicators = len(df[s_groupe].unique())
    font_size = 18 + (num_indicators) * (-2/3)  # Linéaire entre 10 et 18 indicateurs

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

    formatted_labels = [format_label(label, truncate_labels) for label in labels]
    
    if df_ref is not None:
        # Aligner les indicateurs du fichier de référence avec ceux du fichier principal
        df_ref_grouped = df_ref.groupby([groupe, s_groupe])["valeur_indice"].mean().reset_index()
        ref_values = []
        for label in labels:  # Utiliser les labels d'origine pour la correspondance
            if label in df_ref_grouped[s_groupe].values:
                ref_values.append(df_ref_grouped.loc[df_ref_grouped[s_groupe] == label, "valeur_indice"].values[0])
            else:
                ref_values.append(0)  # Valeur par défaut si l'indicateur est manquant
        ax.bar(angles, ref_values, color='none', alpha=1, width=2*np.pi/num_vars, zorder=2.5, linewidth=1, edgecolor='lightcoral', linestyle='dashed')

    
    # Tracer les barres principales après les barres de référence
    ax.bar(angles, values, color=colors, alpha=1, width=2*np.pi/num_vars, zorder=2, linewidth=3, edgecolor='white')
    ax.tick_params(axis='y', labelcolor='white', labelsize=0, grid_color='#FFF', grid_alpha=0, width=0)
    ax.set_ylim(0, 1.05)

    ax.set_xticks(angles)
    ax.set_xticklabels(formatted_labels, fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size, labelcolor='#222', grid_color='#555', zorder=0, grid_alpha=0.1, pad=5)

    if(groupe == "type_besoins"):
        df_affiche = df_grouped.iloc[:, [0, 1, 2]].set_axis(['Type de besoin', 'Besoin', 'Indice'], axis=1)
        df_affiche = df_affiche.iloc[[0] + [i for i in range(-1, -len(df_affiche), -2)] + [i for i in range(-2, -len(df_affiche), -2)]]
        df_affiche = df_affiche.set_index("Type de besoin")  # <-- assignation ici
    else:
        df_affiche = df_grouped.iloc[:, [1, 2]].set_axis(['Besoin', 'Indice'], axis=1)
        df_affiche = df_affiche.iloc[[0] + [i for i in range(-1, -len(df_affiche), -2)] + [i for i in range(-2, -len(df_affiche), -2)]]
        df_affiche = df_affiche.set_index("Besoin")  # <-- assignation ici

    if df_ref is not None:
        st.html(f"référence : {df_ref_info.iloc[1, 1]}<br>")

    st.pyplot(fig)

    st.write(df_affiche.round(2))

    return list(zip(df_grouped[s_groupe], colors))












df = None
df_ref = None
uploaded_files = []


































with st.expander("Paramètres"):

    uploaded_files = st.file_uploader("Choisir un ou plusieurs fichiers Excel", type=["xlsx"], accept_multiple_files=True)
    if uploaded_files:
        file_names = []
        for uploaded_file in uploaded_files:
            try:
                df_info = pd.read_excel(uploaded_file, sheet_name="Informations")
                file_name = df_info.iloc[1, 1]  # Lire la cellule B3 de l'onglet "Informations"
                file_names.append(file_name)
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")

        selected_file_name = st.selectbox("Sélectionnez l'EPCI à étudier : ", file_names)
        selected_file = next(file for file in uploaded_files if pd.read_excel(file, sheet_name="Informations").iloc[1, 1] == selected_file_name)

        reference_file_name = st.selectbox("Sélectionnez un EPCI de référence :", [None] + file_names)
        reference_file = next((file for file in uploaded_files if pd.read_excel(file, sheet_name="Informations").iloc[1, 1] == reference_file_name), None)

        try:
            df = pd.read_excel(selected_file, sheet_name='Export')
            df_besoins = pd.read_excel(selected_file, sheet_name="Besoins_Infos")
            df_indicateurs = pd.read_excel(selected_file, sheet_name="Indicateurs_Infos")
            df_info = pd.read_excel(selected_file, sheet_name="Informations")

            if reference_file:
                df_ref = pd.read_excel(reference_file, sheet_name='Export')
                df_ref_info = pd.read_excel(reference_file, sheet_name='Informations')
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")


# 👉 Affichage seulement si le fichier a bien été chargé
if df is not None:

    st.markdown(f"# {df_info.iloc[1, 1]} - Les onze besoins")

    
    # truncate_labels_global = st.checkbox("Tronquer les étiquettes", value=True, key="truncate_labels_global")
    add_to_radar(df, 'type_besoins', 'besoins', True, df_ref)


    besoins_list = df['besoins'].unique()
    selected_besoin = st.selectbox("", besoins_list)

    st.markdown(f"# {df_info.iloc[1, 1]} - {selected_besoin}")

    if selected_besoin:
        df_selected = df[df['besoins'] == selected_besoin]
        values_dict = dict(zip(df_selected["designation_indicateur"], df_selected["valeur_indice"]))

        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap('RdYlGn')


        truncate_labels_specific = st.checkbox("Tronquer les étiquettes", value=True, key="truncate_labels_specific")
        ordered_indicators = add_to_radar(df_selected, 'besoins', 'designation_indicateur', truncate_labels_specific, df_ref)


        
        html_content = '<h3>Cliquez pour en savoir plus :</h3><div>'
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
            <a style="color: {ft_color}; text-decoration: none;" href="{href}" target="_blank">
                <div class="ksln-cards" style="margin: 0px auto auto 3px;">
                    <p style="
                        text-align: center;
                        padding: 10px;
                        background-color: {bg_color};
                        border-radius: 12px;
                        transition: 0.3s ease;
                        font-weight: 500;
                        margin-bottom: 6px;
                    ">
                        {label}
                    </p>
                </div>
            </a>
            """
        html_content += '</div>'
        st.html(html_content)
else:
    st.warning("Aucun fichier n’a été chargé. Veuillez en charger un pour commencer.")
