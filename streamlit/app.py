import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
import pickle
import os
from dotenv import load_dotenv
import joblib
import io

# Page config
st.set_page_config(page_title="GetAround Pricing", page_icon="🚗")

# Load environment variables
load_dotenv()

# Configuration explicite des credentials AWS
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

# Load model from S3
bucket = os.getenv('S3_BUCKET')
model_path = "mlflow/models/xgboost_model_4e1d09c075954401b4323321c1c84fc5.joblib"
loaded_model = None

s3 = boto3.client('s3')

try:
    response = s3.get_object(Bucket=os.getenv('S3_BUCKET'), Key=model_path)
    model_bytes = io.BytesIO(response['Body'].read())
    loaded_model = joblib.load(model_bytes)
    st.write("Modèle chargé avec succès")
except Exception as e:
    st.error(f"Erreur chargement modèle: {e}")

# Configuration explicite des credentials AWS
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

# Titre et description
st.title('GetAround Price Prediction 🚗')
st.write('Estimez le prix de location de votre véhicule')

# Formulaire principal en colonnes
col1, col2 = st.columns(2)

with col1:
    st.header('Caractéristiques')
    model_key = st.selectbox('Marque', ['Citroën', 'Renault', 'BMW', 'Peugeot', 'Audi'])
    mileage = st.number_input('Kilométrage', min_value=0)
    engine_power = st.number_input('Puissance moteur (ch)', min_value=0)
    fuel = st.selectbox('Carburant', ['diesel', 'essence', 'hybride', 'électrique'])
    car_type = st.selectbox('Type', ['berline', 'SUV', 'citadine'])
    paint_color = st.selectbox('Couleur', ['beige', 'black', 'blue', 'brown', 'green', 'grey', 'orange', 'red', 'silver', 'white'])

with col2:
    st.header('Options')
    has_gps = st.checkbox('GPS')
    has_air_conditioning = st.checkbox('Climatisation')
    automatic_car = st.checkbox('Boîte automatique')
    has_getaround_connect = st.checkbox('GetAround Connect')
    private_parking = st.checkbox('Parking privé')
    has_speed_regulator = st.checkbox('Régulateur de vitesse')
    winter_tires = st.checkbox('Pneus hiver')


# Bouton de prédiction au centre
if st.button('Calculer le prix recommandé'):
    # Préparation des données pour la prédiction
    data = {
        'model_key': [model_key],
        'mileage': [mileage],
        'engine_power': [engine_power],
        'fuel': [fuel],
        'car_type': [car_type],
        'private_parking_available': [int(private_parking)],
        'has_gps': [int(has_gps)],
        'has_air_conditioning': [int(has_air_conditioning)],
        'automatic_car': [int(automatic_car)],
        'has_getaround_connect': [int(has_getaround_connect)],
        'has_speed_regulator': [int(has_speed_regulator)],
        'paint_color': [paint_color],
        'winter_tires': [int(winter_tires)]
    }
    
    df_pred = pd.DataFrame(data)
    
    try:
        # Directement prédire avec le pipeline chargé
        prediction = loaded_model.predict(df_pred)[0]
        
        # Affichage du résultat
        st.success(f'Prix recommandé : {prediction:.2f}€ par jour')
        
        # Message contextuel
        if prediction < 50:
            st.info('💡 Ce prix est relativement bas. Vérifiez que toutes les options sont bien renseignées.')
        elif prediction > 200:
            st.info('💡 Ce prix est relativement élevé, votre véhicule dispose probablement d\'équipements premium.')
            
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.error(f"Détails de l'erreur : {str(e)}")

# Charger les données sur le marché
@st.cache_data
def load_data():
    df = pd.read_csv('streamlit/get_around_pricing_project_clean.csv') #si test en ajoutant streamlit/..si pb en local supprimer
    return df

df = load_data()

# Ajout visualisations
st.header('Analyses du marché')
tab1, tab2, tab3 = st.tabs(['Prix par marque', 'Impact des options', 'Analyse des retards'])

with tab1:
    # Prix moyen par marque
    fig1 = px.bar(
        df.groupby('model_key')['rental_price_per_day'].mean().reset_index(),
        x='model_key',
        y='rental_price_per_day',
        title='Prix moyen par marque',
        labels={'rental_price_per_day': 'Prix par jour ($)', 'model_key': 'Marque'}
    )
    st.plotly_chart(fig1)

with tab2:
    features = ['has_gps', 'has_air_conditioning', 'automatic_car', 
                'has_getaround_connect', 'has_speed_regulator', 'winter_tires']
    
    impact_data = []
    for feature in features:
        avg_with = df[df[feature] == True]['rental_price_per_day'].mean()
        avg_without = df[df[feature] == False]['rental_price_per_day'].mean()
        impact_data.append({
            'feature': feature,
            'with_option': avg_with,
            'without_option': avg_without
        })
    
    impact_df = pd.DataFrame(impact_data)
    fig2 = px.bar(
        impact_df,
        x='feature',
        y=['with_option', 'without_option'],
        barmode='group',
        title='Impact des options sur le prix',
        labels={'value': 'Prix par jour ($)', 'feature': 'Option'}
    )
    st.plotly_chart(fig2)
    
with tab3:
    # Charger les données des retards
    @st.cache_data
    def load_delay_data():
        return pd.read_excel('streamlit/get_around_delay_analysis.xlsx') #si test en ajoutant streamlit/..si pb en local supprimer)
    
    
    # Distribution des retards
    delay_df = load_delay_data()

    fig_delay = px.histogram(
    delay_df,
    x='delay_at_checkout_in_minutes',
    histnorm=None,
    opacity=0.75,
    title='Distribution des retards au check-out',
    labels={'delay_at_checkout_in_minutes': 'Retard (minutes)', 'count': 'Nombre de locations'},
    range_x=[-200, 200]
    )

    fig_delay.update_traces(xbins=dict(size=10))
    fig_delay.add_vline(x=0, line_dash="dash", line_color="red", line_width=1)
    st.plotly_chart(fig_delay)
    
    # Impact par type de check-in
    delay_by_checkin = delay_df.groupby('checkin_type')['delay_at_checkout_in_minutes'].agg(['mean', 'count']).reset_index()
    fig_checkin = px.bar(
        delay_by_checkin,
        x='checkin_type',
        y='mean',
        title='Retard moyen par type de check-in',
        labels={'mean': 'Retard moyen (min)', 'checkin_type': 'Type de check-in'}
    )
    st.plotly_chart(fig_checkin)