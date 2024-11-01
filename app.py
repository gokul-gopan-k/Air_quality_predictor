import numpy as np
import pandas as pd
import joblib
import gradio as gr

X = np.load('X_data.npy')
X_mean = np.load('X_mean.npy')
X_std = np.load('X_std.npy')
Y = np.load('Y_data.npy')
Y_mean = np.load('Y_mean.npy')
Y_std = np.load('Y_std.npy')

model = joblib.load('ridge_model.joblib')

# Example AQI ranges and a concentration range
aqi_ranges = [(0, 12), (12.1, 35.4), (35.5, 55.4),(55.5 ,150.4),(150.5,250.4),(250.5,float('inf'))]
rho = 0.99; rho_LS = 0.99
alpha = 0.1

locations = {
        'Aotizhongxin': '39.989444, 116.409722',
        'Changping': '40.218086, 116.235908',
        'Dingling': '40.290168, 116.220278'
    }

def denormalize(Y_normalized, Y_mean, Y_std):
    Y_original = Y_normalized * Y_std + Y_mean
    return  Y_original

def pm_to_aqi(aqi_ranges, concentration_range):
    start_conc, end_conc = concentration_range
    largest_index = None
    largest_start = -1  # Track the largest starting point of overlapping ranges

    for idx, aqi_range in enumerate(aqi_ranges):
        aqi_start, aqi_end = aqi_range
        # Check for overlap
        if aqi_end == float('inf'):
            if start_conc >= aqi_start:
                # Infinite overlap size
                if largest_index is None or aqi_start > largest_start:
                    largest_index = idx
                    largest_start = aqi_start
        elif start_conc < aqi_end and end_conc > aqi_start:
            # Calculate overlap
            if aqi_start > largest_start:
                largest_index = idx
                largest_start = aqi_start
    
    return largest_index  # Return the index of the range with the highest starting point or None if no overlap found

def get_outputs(index):
    if index == 0:
        level = "Good"
        color = "lightgreen"
        aq = "Air quality is satisfactory, and air pollution poses little or no risk."
        coe = "No specific actions are needed. Enjoy normal outdoor activities."
    elif index == 1:
        level = "Moderate"
        color = "yellow"
        aq = "AIr quality is acceptable; however, some pollutants may be a concern for a very small number of people."
        coe = "People who are unusually sensitive to air pollution may consider limiting prolonged outdoor exertion."
    elif index == 2:
        level = "Unhealthy for sensitive people"
        color = "orange"
        aq ="Health effects may start to impact sensitive groups."
        coe = "Sensitive groups (children, elderly, people with respiratory issues) should limit prolonged outdoor exertion. Everyone else can continue outdoor activities but should remain cautious."
    elif index == 3:
        level = "Unhealthy"
        color = "red"
        aq = "Health effects may be experienced by everyone, with sensitive individuals at greater risk."
        coe = "Everyone should reduce prolonged outdoor exertion. Sensitive individuals should avoid outdoor activities. Indoor air purifiers can be beneficial."
    elif index == 4:
        level = "Very unhealthy"
        color = "darkred"
        aq = "Health warnings of emergency conditions; the entire population may experience health effects."
        coe = "Avoid outdoor activities if possible. All individuals, especially sensitive groups, should remain indoors and use air purifiers. Wear N95 masks if outdoor exposure is necessary."
    elif index == 5:
        level = "Hazardous"
        color = "Magenta"
        aq = "Serious health effects likely for the entire population."
        coe = " Everyone should stay indoors and keep windows and doors closed. Avoid physical activity and consider relocating temporarily if exposure is prolonged. Use air purifiers and tightly seal indoors."
    
    return level, color,aq, coe


# Info dictionary for each parameter
parameter_info = {
    "SO2": "Sulfur Dioxide (SO2) is a toxic gas produced by volcanic activity and industrial processes, notably the combustion of fossil fuels. Measured in micrograms per cubic meter (µg/m³). Typical Range: 0 to 100 µg/m",
    "NO2": "Nitrogen Dioxide (NO2) is a reddish-brown gas with a characteristic sharp, biting odor, a significant air pollutant from combustion processes. Measured in micrograms per cubic meter (µg/m³). Typical Range: 0 to 200 µg/m",
    "CO": "Carbon Monoxide (CO) is a colorless, odorless gas that can be harmful when inhaled in large amounts, often produced by incomplete combustion. Measured in micrograms per cubic meter (µg/m³). Typical Range: 0 to 10000 µg/m",
    "O3": "Ozone (O3) at ground level is an air pollutant that affects lung function and is formed from reactions between sunlight and other pollutants. Measured in micrograms per cubic meter (µg/m³). Typical Range: 0 to 180 µg/m",
    "Temperature": "Temperature in meteorology is a measure of the warmth or coldness of the environment. Measured in Degrees Celsius (°C). Typical Range: -20 to 40 °C; varies with seasons.",
    "Pressure": "Pressure refers to the atmospheric force exerted on a surface, essential in predicting weather patterns. Measured in hectopascals (hPa). Typical Range: 900 to 1050 hPa generally lower in summer and higher in winter.",
    "Dew Point": "The Dew Point is the temperature at which air becomes saturated with moisture, crucial for humidity measurement. Measured in degrees Celsius (°C) Typical Range: -20 to 30 °C; higher values indicate more humidity.",
    "Rain": "Rainfall measurement indicates the volume of precipitation,  measured in Millimeters (mm) .Typical Range: 0 to 300 mm; varies significantly by season and weather patterns.",
    "WSPM": "Wind Speed per Minute (WSPM) refers to the rate at which air moves past a certain point. Measured in meters per second (m/s). Typical Range: 0 to 10 m/s; can be higher during storms.",
    "Wind Direction": "Wind Direction indicates the direction from which the wind originates, essential for weather predictions. eg: NNW denotes North-Northwest",
    "Location": "Location selection allows you to specify the area where the air quality parameters are being recorded or predicted."
}

# Function to return info text based on the parameter selected
def show_info(parameter):
    return parameter_info.get(parameter, "Information not available.")



# Function to process the inputs and return a numerical output
def process_inputs(num1, num2, num3, num4, num5, num6, num7, num8, num9, wind, location):
    # Example logic: sum the numerical inputs and assign a multiplier based on the categorical input
    wind_horizontal = {
        "N": 0, "S":0,
        "NNW": -0.5,"SSE":-0.5,
        "SW":-0.7,"NW": -0.7,
        "WNW": -0.86,"WSW": -0.86,
        "W": -1,
        "NNE": 0.5, "SSE":0.5,
        "NE": 0.7, "SE":0.7,
        "ENE": 0.86,
        "ESE":0.86,
        "E": 1,
        "Unknown": 0
    }
    wind_vertical = {
        "W": 0, "E":0,
        "WSW": -0.5,"ESE":-0.5,
        "SW":-0.7,"SE": -0.7,
        "SSW": -0.86,"SSE": -0.86,
        "S": -1,
        "WNW": 0.5, "ENE":0.5,
        "NW": 0.7, "NE":0.7,
        "NNW": 0.86,
        "NNE":0.86,
        "N": 1,
        "Unknown": 0
    }
    w_h = wind_horizontal.get(wind, 0)
    w_v = wind_vertical.get(wind, 0)

    data = {
    "S02": [num1],"N02": [num2],"CO": [num3],"O3": [num4],"Temp": [num5],"Pres": [num6],
    "DW": [num7],"Rain": [num8],"WSPM": [num9], "wind_h":[w_h], "wind_v":[w_v]
    
    }

    # Create DataFrame from dictionary
    df = pd.DataFrame(data)
    
    X_test = df.to_numpy().astype('float64')
    X_norma = (X_test - X_mean) / X_std


    n = len(Y)
    weights=rho**(np.arange(n,0,-1))
    tags=rho_LS**(np.arange(n,-1,-1))
    loss_beta = 1.0 #loss upper bound
    lambdas = np.arange(0,2,0.01) #set lambdas range (lambda*2)

    # odd data points for training, even for calibration
    inds_odd = np.arange(1,int(np.ceil(n/2)*2-1),2)
    inds_even = np.arange(2,int(np.floor(n/2)*2),2)

    X_calib = X[inds_even]
    y_calib = Y[inds_even]


    # get predictions on test set
    Y_pred = model.predict(X_calib)

    # compute residuals
    residuals = np.abs(y_calib - Y_pred)

    # compute weighted empirical risk in the calibration set
    losses = np.zeros((len(lambdas), len(residuals)))
    n_w = np.sum(weights[inds_even] )
    r_hats = np.zeros(len(lambdas))

    for li,l in enumerate(lambdas):
        losses[li,:]=np.maximum(np.zeros((residuals.shape)),residuals-(l/2))
        ws = np.sum(weights[inds_even]*losses[li])
        r_hats[li] = (1/n_w) * ws

    calib_lambdas = (r_hats*n_w/(n_w+1)) + loss_beta/(n_w+1)
    lambda_chosen = np.max(lambdas)

    # find the infimum of lambdas
    for i,li in enumerate(calib_lambdas):
        if li<=alpha:
            lambda_chosen = lambdas[i]
            break


    x_1 = X_norma.reshape(1, -1)
    y_PI = np.array([denormalize(model.predict(x_1), Y_mean ,Y_std)-(lambda_chosen/2),denormalize(model.predict(x_1), Y_mean ,Y_std)+(lambda_chosen/2)])
    

    # Get coordinates for selected location
    coordinates = locations.get(location, "0,0")  # Default to "0,0" if location not found
    
    lat, lng = coordinates.split(",")
    
    # Generate the OpenStreetMap iframe embed code with the selected coordinates
    osm_map_html = f"""
    <iframe 
      width="100%" 
      height="400" 
      frameborder="0" 
      scrolling="no" 
      marginheight="0" 
      marginwidth="0" 
      src="https://www.openstreetmap.org/export/embed.html?bbox={float(lng)-0.05}%2C{float(lat)-0.05}%2C{float(lng)+0.05}%2C{float(lat)+0.05}&layer=mapnik&marker={lat}%2C{lng}" 
      style="border: 1px solid black"></iframe>
    """

    result = pm_to_aqi(aqi_ranges, (np.exp(y_PI[0][0]),np.exp(y_PI[1][0])))

    level, color,aq, coe = get_outputs(result)
    
    out_2_html = f"<span style='color: {color}; font-size: 20px; text-transform: uppercase;'>{level}</span>"


    return str(round(np.exp(float(y_PI[0][0])),2)) + " to " + str(round(np.exp(float(y_PI[1][0])),2)), osm_map_html, out_2_html, aq, coe
    


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
