import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Define the LabelEncoder object
label_encoder = LabelEncoder()
MODEL = ['Alto', 'Grand', 'i20', 'Ecosport', 'Wagon R', 'i10', 'Venue',
         'Swift', 'Verna', 'Duster', 'Cooper', 'Ciaz', 'C-Class', 'Innova',
         'Baleno', 'Swift Dzire', 'Vento', 'Creta', 'City', 'Bolero',
         'Fortuner', 'KWID', 'Amaze', 'Santro', 'XUV500', 'KUV100', 'Ignis',
         'RediGO', 'Scorpio', 'Marazzo', 'Aspire', 'Figo', 'Vitara',
         'Tiago', 'Polo', 'Seltos', 'Celerio', 'GO', '5', 'CR-V',
         'Endeavour', 'KUV', 'Jazz', '3', 'A4', 'Tigor', 'Ertiga', 'Safari',
         'Thar', 'Hexa', 'Rover', 'Eeco', 'A6', 'E-Class', 'Q7', 'Z4', '6',
         'XF', 'X5', 'Hector', 'Civic', 'D-Max', 'Cayenne', 'X1', 'Rapid',
         'Freestyle', 'Superb', 'Nexon', 'XUV300', 'Dzire VXI', 'S90',
         'WR-V', 'XL6', 'Triber', 'ES', 'Wrangler', 'Camry', 'Elantra',
         'Yaris', 'GL-Class', '7', 'S-Presso', 'Dzire LXI', 'Aura', 'XC',
         'Ghibli', 'Continental', 'CR', 'Kicks', 'S-Class', 'Tucson',
         'Harrier', 'X3', 'Octavia', 'Compass', 'CLS', 'redi-GO', 'Glanza',
         'Macan', 'X4', 'Dzire ZXI', 'XC90', 'F-PACE', 'A8', 'MUX',
         'GTC4Lusso', 'GLS', 'X-Trail', 'XE', 'XC60', 'Panamera', 'Alturas',
         'Altroz', 'NX', 'Carnival', 'C', 'RX', 'Ghost', 'Quattroporte',
         'Gurkha']


def main():
    # st.title('Used Car Price Prediction')
    st.markdown("<h1 style='text-align: center;'>Used Car Price Prediction</h1>", unsafe_allow_html=True)
    img = Image.open('images\car.jpg')
    st.image(img, width=None, use_column_width=True)

    # Input fields
    car_model = st.selectbox('Car Model', MODEL)
    
    col4, col5 = st.columns(2)

    with col4:
        vehicle_age = st.number_input('Vehicle Age', min_value=0)
    with col5:
        km_driven = st.number_input('Total km driven', min_value=0, step=1000)

    # mapping
    seller_type_list = {'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2}
    fuel_type_list = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3,
                        'Electric': 4}
    transmission_type_list = {'Manual': 0, 'Automatic': 1}

    col1, col2, col3 = st.columns(3)
    with col1:
    
        option1 = st.radio('Seller Type',
                           ['Individual', 'Dealer', 'Trustmark Dealer'])
    seller_type = seller_type_list[option1]

    with col2:
        option2 = st.radio('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG',
                           'Electric'])
    fuel_type = fuel_type_list[option2]

    with col3:
        option3 = st.radio('Transmission Type', ['Manual', 'Automatic'])
    transmission_type = transmission_type_list[option3]

    col7, col8 = st.columns(spec=2, gap='large')
    with col7:
        mileage = st.number_input('Mileage', min_value=0.0, max_value=None, step=1.0)
    with col8:
        engine = st.number_input('Engine', min_value=0)

    col9, col10 = st.columns(spec=2, gap='large')
    with col9:
        max_power = st.number_input('Maximum Power', step=1.0)
    with col10:
        seats = st.selectbox('Number of Seats', [0, 2, 4, 5, 6, 7, 8, 9])

    if st.button('Predict Price Range'):
        # Validate input data
        if validate_inputs([car_model, vehicle_age, km_driven, seller_type,
                            fuel_type, transmission_type, mileage,
                            engine, max_power, seats]):
            # Load model and scaler
            try:
                model = pickle.load(open('models\model.sav', 'rb'))
                scaler = pickle.load(open('models\scaler.sav', 'rb'))
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return

            # Label encode car_model
            try:
                label_encoder.fit(MODEL)
                car_model_encoded = label_encoder.transform([car_model])[0]

            except Exception as e:
                st.error(f"Error encoding car model: {e}")
                return

            # Predict price range
            try:
                prediction = model.predict(
                    scaler.transform([[
                        car_model_encoded, vehicle_age, km_driven,
                        seller_type, fuel_type, transmission_type,
                        float(mileage), engine, max_power, seats]]))[0]
            except Exception as e:
                st.error(f"Error predicting price range: {e}")
                return

            # Display predicted price range
            st.success(f'Estimate price : {int(prediction)}')
        else:
            st.error('Please enter valid values for all input features.')


def validate_inputs(inputs):
    for i in inputs:
        if i == '':
            return False
    return True


if __name__ == "__main__":
    main()
