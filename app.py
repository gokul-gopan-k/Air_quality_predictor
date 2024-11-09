import numpy as np
import pandas as pd
import joblib
import gradio as gr
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load data and model files
try:
    X = np.load('X_data.npy')
    X_mean = np.load('X_mean.npy')
    X_std = np.load('X_std.npy')
    Y = np.load('Y_data.npy')
    Y_mean = np.load('Y_mean.npy')
    Y_std = np.load('Y_std.npy')
    model = joblib.load('ridge_model.joblib')
    logging.info("Data and model loaded successfully.")
except Exception as e:
    logging.error("Error loading data or model: %s", e)

# Define AQI categories for PM2.5 concentration levels
aqi_ranges = [(0, 12), (12.1, 35.4), (35.5, 55.4), (55.5, 150.4), (150.5, 250.4), (250.5, float('inf'))]
rho = 0.99  # Discount factor for model
rho_LS = 0.99  # Discount factor for line search
alpha = 0.1  # Confidence level threshold

# Define mapping for locations with their respective coordinates
locations = {
    'Aotizhongxin': '39.989444, 116.409722',
    'Changping': '40.218086, 116.235908',
    'Dingling': '40.290168, 116.220278'
}

def denormalize(Y_normalized, Y_mean, Y_std):
    """
    Converts normalized data back to original scale.
    
    Args:
        Y_normalized (array): Normalized data array.
        Y_mean (float): Mean value of the original dataset.
        Y_std (float): Standard deviation of the original dataset.
    
    Returns:
        array: Denormalized data array.
    """
    return Y_normalized * Y_std + Y_mean

def pm_to_aqi(aqi_ranges, concentration_range):
    """
    Determines AQI category based on PM2.5 concentration range.
    
    Args:
        aqi_ranges (list): List of AQI ranges as tuples.
        concentration_range (tuple): Start and end of concentration range.
    
    Returns:
        int: Index of the AQI category in the aqi_ranges list.
    """
    start_conc, end_conc = concentration_range
    largest_index = None
    largest_start = -1

    # Identify the AQI category that the concentration range falls into
    for idx, (aqi_start, aqi_end) in enumerate(aqi_ranges):
        if aqi_end == float('inf'):
            if start_conc >= aqi_start:
                if largest_index is None or aqi_start > largest_start:
                    largest_index = idx
                    largest_start = aqi_start
        elif start_conc < aqi_end and end_conc > aqi_start:
            if aqi_start > largest_start:
                largest_index = idx
                largest_start = aqi_start

    logging.info("AQI category index selected: %s", largest_index)
    return largest_index

def get_outputs(index):
    """
    Returns AQI level, color, associated health risk, and suggested actions based on AQI category index.
    
    Args:
        index (int): AQI category index.
    
    Returns:
        tuple: (AQI level, color code, health risk message, and suggested actions).
    """
    levels = [
        ("Good", "lightgreen", "Air quality is satisfactory, and air pollution poses little or no risk.",
         "No specific actions are needed. Enjoy normal outdoor activities."),
        ("Moderate", "yellow", "Air quality is acceptable; however, some pollutants may be a concern for a very small number of people.",
         "People who are unusually sensitive to air pollution may consider limiting prolonged outdoor exertion."),
        ("Unhealthy for sensitive groups", "orange", "Health effects may start to impact sensitive groups.",
         "Sensitive groups should limit prolonged outdoor exertion."),
        ("Unhealthy", "red", "Health effects may be experienced by everyone, with sensitive individuals at greater risk.",
         "Everyone should reduce prolonged outdoor exertion. Sensitive individuals should avoid outdoor activities."),
        ("Very unhealthy", "darkred", "Health warnings of emergency conditions; the entire population may experience health effects.",
         "Avoid outdoor activities if possible. Indoor air purifiers are recommended."),
        ("Hazardous", "magenta", "Serious health effects likely for the entire population.",
         "Stay indoors and avoid physical activity. Use air purifiers and tightly seal indoors.")
    ]
    
    if index is not None and 0 <= index < len(levels):
        logging.info("Selected AQI level and health guidance for index: %s", index)
        return levels[index]
    else:
        logging.warning("Invalid AQI index: %s. Defaulting to 'Good' level.", index)
        return levels[0]

# Additional parameter descriptions
parameter_info = {
    "SO2": "Sulfur Dioxide (SO2) is a toxic gas from combustion and volcanic activity. Range: 0 to 100 µg/m³.",
    "NO2": "Nitrogen Dioxide (NO2) is a reddish-brown gas from combustion processes. Range: 0 to 200 µg/m³.",
    "CO": "Carbon Monoxide (CO) is a harmful, colorless gas from incomplete combustion. Range: 0 to 10000 µg/m³.",
    "O3": "Ozone (O3) at ground level is an air pollutant affecting lung function. Range: 0 to 180 µg/m³.",
    "Temperature": "Temperature indicates the warmth or coldness of the environment. Range: -20 to 40 °C.",
    "Pressure": "Atmospheric pressure affects weather prediction. Range: 900 to 1050 hPa.",
    "Dew Point": "Dew Point measures air saturation with moisture. Range: -20 to 30 °C.",
    "Rain": "Rainfall measures precipitation volume. Range: 0 to 300 mm.",
    "WSPM": "Wind Speed per Minute (WSPM) indicates the rate of air movement. Range: 0 to 10 m/s.",
    "Wind Direction": "Wind Direction specifies where the wind originates from.",
    "Location": "Select a location to specify where the air quality parameters are measured."
}

def show_info(parameter):
    """
    Provides description of the selected parameter.
    
    Args:
        parameter (str): Name of the parameter to show info about.
    
    Returns:
        str: Description text of the parameter.
    """
    info = parameter_info.get(parameter, "Information not available.")
    logging.info("Displaying info for parameter: %s", parameter)
    return info

def process_inputs(num1, num2, num3, num4, num5, num6, num7, num8, num9, wind, location):
    """
    Processes environmental inputs and predicts PM2.5 concentration along with AQI classification,
    as well as generates an OpenStreetMap embed for the selected location.

    Args:
        num1 to num9: Numerical environmental parameters (e.g., SO2, NO2, CO, etc.)
        wind: Wind direction as a string.
        location: Location name for the map.

    Returns:
        A tuple containing:
        - PM2.5 concentration prediction range.
        - OpenStreetMap iframe for the location.
        - AQI classification (with color).
        - Actionable insights based on the AQI.
    """
    
    # Wind direction mappings for horizontal and vertical adjustments
    wind_horizontal = {
        "N": 0, "S": 0, "NNW": -0.5, "SSE": -0.5, "SW": -0.7, "NW": -0.7,
        "WNW": -0.86, "WSW": -0.86, "W": -1, "NNE": 0.5, "SSE": 0.5, 
        "NE": 0.7, "SE": 0.7, "ENE": 0.86, "ESE": 0.86, "E": 1, "Unknown": 0
    }
    
    wind_vertical = {
        "W": 0, "E": 0, "WSW": -0.5, "ESE": -0.5, "SW": -0.7, "SE": -0.7, 
        "SSW": -0.86, "SSE": -0.86, "S": -1, "WNW": 0.5, "ENE": 0.5, 
        "NW": 0.7, "NE": 0.7, "NNW": 0.86, "NNE": 0.86, "N": 1, "Unknown": 0
    }

    # Get wind horizontal and vertical factors based on wind direction
    wind_h = wind_horizontal.get(wind, 0)
    wind_v = wind_vertical.get(wind, 0)

    # Prepare the input data as a dictionary
    data = {
        "SO2": [num1], "NO2": [num2], "CO": [num3], "O3": [num4], "Temp": [num5], 
        "Pres": [num6], "DW": [num7], "Rain": [num8], "WSPM": [num9], 
        "wind_h": [wind_h], "wind_v": [wind_v]
    }

    # Create DataFrame from input data
    df = pd.DataFrame(data)
    X_test = df.to_numpy().astype('float64')
    X_norma = (X_test - X_mean) / X_std  # Normalize the input data

    # Set model parameters
    n = len(Y)
    weights = rho ** (np.arange(n, 0, -1))
    tags = rho_LS ** (np.arange(n, -1, -1))
    loss_beta = 1.0  # Loss upper bound
    lambdas = np.arange(0, 2, 0.01)  # Lambda range

    # Split the data into calibration and test sets
    inds_odd = np.arange(1, int(np.ceil(n / 2) * 2 - 1), 2)
    inds_even = np.arange(2, int(np.floor(n / 2) * 2), 2)

    X_calib = X[inds_even]
    y_calib = Y[inds_even]

    # Predict on calibration set
    Y_pred = model.predict(X_calib)
    residuals = np.abs(y_calib - Y_pred)

    # Calculate the weighted empirical risk
    losses = np.zeros((len(lambdas), len(residuals)))
    n_w = np.sum(weights[inds_even])
    r_hats = np.zeros(len(lambdas))

    for li, l in enumerate(lambdas):
        losses[li, :] = np.maximum(np.zeros((residuals.shape)), residuals - (l / 2))
        ws = np.sum(weights[inds_even] * losses[li])
        r_hats[li] = (1 / n_w) * ws

    calib_lambdas = (r_hats * n_w / (n_w + 1)) + loss_beta / (n_w + 1)
    lambda_chosen = np.max(lambdas)

    # Select optimal lambda based on calibration
    for i, li in enumerate(calib_lambdas):
        if li <= alpha:
            lambda_chosen = lambdas[i]
            break

    # Predict PM2.5 concentration with uncertainty bounds
    x_1 = X_norma.reshape(1, -1)
    y_PI = np.array([denormalize(model.predict(x_1), Y_mean, Y_std) - (lambda_chosen / 2), 
                     denormalize(model.predict(x_1), Y_mean, Y_std) + (lambda_chosen / 2)])

    # Get coordinates for the selected location
    coordinates = locations.get(location, "0,0")  # Default to "0,0" if not found
    lat, lng = map(float, coordinates.split(","))
    
    # Generate OpenStreetMap iframe embed code
    osm_map_html = f"""
    <iframe width="100%" height="400" frameborder="0" scrolling="no" marginheight="0" marginwidth="0"
      src="https://www.openstreetmap.org/export/embed.html?bbox={lng - 0.05}%2C{lat - 0.05}%2C{lng + 0.05}%2C{lat + 0.05}&layer=mapnik&marker={lat}%2C{lng}" 
      style="border: 1px solid black"></iframe>
    """

    # Calculate AQI level and health risk based on predicted PM2.5 range
    result = pm_to_aqi(aqi_ranges, (np.exp(y_PI[0][0]), np.exp(y_PI[1][0])))
    level, color, aq, coe = get_outputs(result)

    # Prepare AQI classification output in HTML format
    out_2_html = f"<span style='color: {color}; font-size: 20px; text-transform: uppercase;'>{level}</span>"

    # Return PM2.5 range, map iframe, AQI classification, and health advice
    return (f"{round(np.exp(float(y_PI[0][0])), 2)} to {round(np.exp(float(y_PI[1][0])), 2)}", 
            osm_map_html, out_2_html, aq, coe)


# Define the Gradio interface
with gr.Blocks() as demo:
    # Center-aligned heading
    gr.Markdown("<h2 style='text-align: center;'>PM2.5 Air Quality Prediction </h2>")
    gr.Markdown("<p style='text-align: center;'>Enter air quality metrics and weather parameters to predict PM2.5 concentration, AQI level and suggested course of action.</p>")
    # Arrange inputs in two columns
    with gr.Row():
        with gr.Column():
    # Create the input components
            num1 = gr.Textbox(label="SO2", value=4.0)
            gr.Button("Info").click(lambda: show_info("SO2"), None, gr.Markdown(), show_progress=False)
            num2 = gr.Textbox(label="NO2", value=7.0)
            gr.Button("Info").click(lambda: show_info("NO2"), None, gr.Markdown(), show_progress=False)
            
            num3 = gr.Textbox(label="CO", value=300.0)
            gr.Button("Info").click(lambda: show_info("CO"), None, gr.Markdown(), show_progress=False)
            
            num4 = gr.Textbox(label="O3", value=77.0)
            gr.Button("Info").click(lambda: show_info("O3"), None, gr.Markdown(), show_progress=False)
            
            num5 = gr.Textbox(label="Temperature", value=-0.7)
            gr.Button("Info").click(lambda: show_info("Temperature"), None, gr.Markdown(), show_progress=False)
        with gr.Column():
            num6 = gr.Textbox(label="Pressure", value=1023.0)
            gr.Button("Info").click(lambda: show_info("Pressure"), None, gr.Markdown(), show_progress=False)
            
            num7 = gr.Textbox(label="Dew point", value=-18.8)
            gr.Button("Info").click(lambda: show_info("Dew Point"), None, gr.Markdown(), show_progress=False)
            
            num8 = gr.Textbox(label="Rain", value=0.0)
            gr.Button("Info").click(lambda: show_info("Rain"), None, gr.Markdown(), show_progress=False)
            
            num9 = gr.Textbox(label="WSPM", value=4.4)
            gr.Button("Info").click(lambda: show_info("WSPM"), None, gr.Markdown(), show_progress=False)
            
            category1 = gr.Dropdown(
                choices=["N", "S", "NNW", "SSE", "SW", "NW", "WNW", "WSW", "W", "NNE", "SSE",
                         "NE", "SE", "ENE", "ESE", "E", "Unknown"],
                label="Wind Direction", 
                value="NNW"
            )
            gr.Button("Info").click(lambda: show_info("Wind Direction"), None, gr.Markdown(), show_progress=False)
    
    
    
    location = gr.Dropdown(choices=list(locations.keys()), label="Select a Location")
    gr.Button("Info").click(lambda: show_info("Location"), None, gr.Markdown(), show_progress=False)
    
    
    # Map embed using iframe in Gradio
    location_html = gr.HTML(label="Selected Location on Map")
    
    # Create the output component
    output_1 = gr.Textbox(label="PM2.5 concentration range in µg/m", interactive=False)
    
    gr.Markdown("### AQI level")
    output_html= gr.HTML(label="AQI scale")
    
    out_aq = gr.Textbox(label="Assosiated Health risk", interactive=False)
    out_coe = gr.Textbox(label="Suggested course of action", interactive=False)
    # Create the submit button and link to the function
    submit_button = gr.Button("Submit")
    
    submit_button.click(process_inputs, inputs=[num1, num2, num3, num4, num5, num6, num7, num8, num9, category1, location], outputs=[output_1,location_html, output_html, out_aq, out_coe])

# Launch the Gradio interface
demo.launch()
