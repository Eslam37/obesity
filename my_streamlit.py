import streamlit as st
import pandas as pd
import seaborn as sns
import hydralit_components as hc
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

st.set_page_config(
    page_title="Obesity",
    layout='wide'
)

# Creating Navigation bar
menu_data = [
    {'label': 'Predictive Model', 'icon': '🧪'}
]
menu_id = hc.nav_bar(menu_definition=menu_data, sticky_mode='sticky', override_theme={'menu_background': '#3A9BCD',
                                                                                       'option_active': 'white'}
                     )

# Life expectancy page
if menu_id == "Predictive Model":
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    st.header("Let us predict the Obesity Level!")

    # Create the data for the table
    table = [
        ["FAVC", "Frequent consumption of high caloric food"],
        ["FCVC", "Frequency of consumption of vegetables"],
        ["NCP", "Number of main meals"],
        ["CAEC", "Consumption of food between meals"],
        ["CH20", "Consumption of water daily "],
        ["CALC", "Consumption of alcohol"],
        ["SCC", "Calories consumption monitoring"],
        ["FAF", "Physical activity frequency"],
        ["TUE", "Time using technology devices"],
        ["MTRANS", "Transportation used"],
    ]

    # Create a DataFrame from the data
    d = pd.DataFrame(table, columns=["Feature", "Description"])

    # Add CSS classes for styling
    styles = [
        {"selector": "table", "props": [("border-collapse", "collapse")]},
        {"selector": "th, td", "props": [("border", "1px solid #ddd"), ("padding", "8px")]},
        {"selector": "th", "props": [("background-color", "#f2f2f2")]},
    ]


    # Display the styled table
    style_tags = "".join(f"{style['selector']}{{{';'.join([f'{prop[0]}:{prop[1]}' for prop in style['props']])}}}" for
                         style in styles)
    st.markdown(f'<style>{style_tags}</style>', unsafe_allow_html=True)
    st.table(d)

    if st.checkbox('Show data sample'):
        st.subheader('Raw data')
        st.write(df)

    # Split the dataset into features (X) and target (y)
    X = df.drop("NObeyesdad", axis=1)
    y = df["NObeyesdad"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the categorical and numerical features
    categorical_features = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
    numerical_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OrdinalEncoder(), categorical_features)])

    # Create the final pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier()),
    ])

    # Train the pipeline on the training data
    model = pipeline.fit(X_train, y_train)

    # Collect user inputs
    col1, _, col2 = st.columns([1, 0.1, 1]) 
    with col1:
        age = st.slider('Age (years)', 14, 61, 20)
    with col2:
        height = st.slider('Height (cm)', 140, 210, 160)

    # Collect user inputs
    col1, _, col2 = st.columns([1, 0.1, 1]) 
    with col1:
        weight = st.slider('Weight (kg)', 30, 150, 60)
    with col2:
        fcvc = st.selectbox('Do you usually eat vegetables in your meals? (1 = Never, 2 = Sometimes, 3 = Always)', [1, 2, 3])

    # Collect user inputs
    col1, _, col2 = st.columns([1, 0.1, 1]) 
    with col1:
        ncp = st.selectbox('Number of main meals (per day)', [1, 2, 3, 4])
    with col2:
        ch2o = st.slider('How much water do you drink daily? (L)', 0.0, 3.0, 1.0, 0.1)

    # Collect user inputs
    col1, _, col2 = st.columns([1, 0.1, 1]) 
    with col1:
        faf = st.selectbox('How often do you have physical activity? (days)', [1, 2, 3])
    with col2:
        tue = st.selectbox('How much time do you use technological devices such as cell phone, videogames, television, computer and others?', ['0 to 2 hrs', '3 to 5 hrs', '5+ hrs'])


    # Collect user inputs
    col1, _, col2 = st.columns([1, 0.1, 1]) 
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
    with col2:
        scc = st.selectbox('Do you monitor the calories you eat daily?', ['no', 'yes'])

    # Collect user inputs
    col1, _, col2 = st.columns([1, 0.1, 1]) 
    with col1:
        calc = st.selectbox('How often do you drink alcohol?', ['no', 'Sometimes', 'Frequently', 'Always'])
    with col2:
        caec = st.selectbox('Do you eat any food between meals?', ['no', 'Sometimes', 'Frequently', 'Always'])

    # Collect user inputs
    col1, _, col2 = st.columns([1, 0.1, 1]) 
    with col1:
        favc = st.selectbox('Do you eat high caloric food frequently?', ['yes', 'no'])
    with col2:
        mtrans = st.selectbox('Which transportation do you usually use?', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])

    # Collect user inputs
    col1, _, col2 = st.columns([1, 0.1, 1]) 
    with col1:
        smoke = st.selectbox('Do you smoke?', ['no', 'yes'])
    with col2:
        family = st.selectbox('Do you have a family history with overweight', ['no', 'yes'])

    if tue == '0 to 2 hrs':
        tue = 1
    elif tue == '3 to 5 hrs':
        tue = 2
    else:
        tue = 3
    # Apply Prediction
    if st.button("Apply Prediction"):
        # Create the input data for prediction
        model = joblib.load("obesity_model.pkl")
        # Create the input data for prediction
        input_data = pd.DataFrame({
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'CH2O': [ch2o],
            'FAF': [faf],
            'TUE': [tue],
            'Gender': [gender],
            'SCC': [scc],
            'CALC': [calc],
            'CAEC': [caec],
            'MTRANS': [mtrans],
            'FAVC': [favc],
            'SMOKE': [smoke],
            'family_history_with_overweight': [family]
        })

        # Make a prediction using the model
        predictions = model.predict(input_data)

        # Display the predicted results
        st.subheader("Prediction Results")
        st.markdown("<h2 style='color: #3A9BCD;'>{}</h2>".format(predictions[0]), unsafe_allow_html=True)
