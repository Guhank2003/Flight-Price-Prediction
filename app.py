import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
try:
    model = pickle.load(open('fligh_price_prediction.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'fligh_price_prediction.pkl' not found. Please ensure the model is trained and saved.")
    st.stop()

# Load the original raw dataset to get category names and their corresponding encoded values
try:
    raw_df = pd.read_csv("Clean_Dataset.csv")
    # Re-apply the encoding logic to raw_df to create `encoded_df`
    # This ensures we get the exact same category codes as used during training.
    encoded_df = raw_df.copy()
    for col in ['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']:
        encoded_df[col] = encoded_df[col].astype('category').cat.codes

except FileNotFoundError:
    st.error("Original dataset file '/content/Clean_Dataset.csv' not found. Cannot create category mappings.")
    st.stop()


# Create mapping dictionaries from original string values to encoded integer values
def create_reverse_mapping(original_col_series, encoded_col_series):
    temp_df = pd.DataFrame({'original': original_col_series, 'encoded': encoded_col_series})
    unique_mappings = temp_df.drop_duplicates().set_index('original')['encoded'].to_dict()
    return unique_mappings

airline_map = create_reverse_mapping(raw_df['airline'], encoded_df['airline'])
source_city_map = create_reverse_mapping(raw_df['source_city'], encoded_df['source_city'])
destination_city_map = create_reverse_mapping(raw_df['destination_city'], encoded_df['destination_city'])
departure_time_map = create_reverse_mapping(raw_df['departure_time'], encoded_df['departure_time'])
arrival_time_map = create_reverse_mapping(raw_df['arrival_time'], encoded_df['arrival_time'])
stops_map = create_reverse_mapping(raw_df['stops'], encoded_df['stops'])
class_map = create_reverse_mapping(raw_df['class'], encoded_df['class'])
flight_map = create_reverse_mapping(raw_df['flight'], encoded_df['flight'])


st.title('Flight Price Prediction')
st.write('Predict the price of your flight!')

# Input widgets
selected_airline = st.selectbox('Airline', sorted(list(airline_map.keys())))
selected_source_city = st.selectbox('Source City', sorted(list(source_city_map.keys())))
selected_destination_city = st.selectbox('Destination City', sorted(list(destination_city_map.keys())))
selected_departure_time = st.selectbox('Departure Time', sorted(list(departure_time_map.keys())))
selected_arrival_time = st.selectbox('Arrival Time', sorted(list(arrival_time_map.keys())))
selected_stops = st.selectbox('Stops', sorted(list(stops_map.keys())))
selected_class = st.selectbox('Class', sorted(list(class_map.keys())))

selected_flight = st.text_input('Flight Number (e.g., SG-8709)', 'SG-8709')

duration = st.number_input('Duration (hours)', min_value=0.5, max_value=50.0, value=5.0, step=0.1)
days_left = st.slider('Days Left for Departure', min_value=1, max_value=50, value=15)

# Prediction button
if st.button('Predict Price'):
    # Get encoded values
    airline_code = airline_map.get(selected_airline)
    source_city_code = source_city_map.get(selected_source_city)
    destination_city_code = destination_city_map.get(selected_destination_city)
    departure_time_code = departure_time_map.get(selected_departure_time)
    arrival_time_code = arrival_time_map.get(selected_arrival_time)
    stops_code = stops_map.get(selected_stops)
    class_code = class_map.get(selected_class)

    flight_code = flight_map.get(selected_flight)

    # Check if any mapping failed (should not happen for selectboxes, but good for text_input)
    if None in [airline_code, source_city_code, destination_city_code, departure_time_code,
                arrival_time_code, stops_code, class_code]:
        st.error("An error occurred with categorical mapping. Please check inputs.")
    elif flight_code is None:
        st.error(f"Flight number '{selected_flight}' not recognized. Please ensure it's a valid flight from the training data.")
    else:
        # Prepare input features for prediction, ensuring column order matches training data
        input_data = pd.DataFrame([[
            0, # Unnamed: 0 (index, using a placeholder 0 as it was present in training X)
            airline_code,
            flight_code,
            source_city_code,
            departure_time_code,
            stops_code,
            arrival_time_code,
            destination_city_code,
            class_code,
            duration,
            days_left
        ]], columns=[
            'Unnamed: 0', 'airline', 'flight', 'source_city', 'departure_time', 'stops',
            'arrival_time', 'destination_city', 'class', 'duration', 'days_left'
        ])

        try:
            predicted_price = model.predict(input_data)[0]
            st.success(f'The predicted flight price is: ₹{predicted_price:,.2f}')
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Please ensure all inputs are valid and the model is correctly loaded.")
