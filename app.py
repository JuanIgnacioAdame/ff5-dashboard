# Import necessary libraries
import requests
import zipfile
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Function to download and process Fama-French data
def download_ff_data():
    """
    Downloads and processes Fama-French 5 Factors daily data.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with PeriodIndex and factor returns
    """
    # URL for the zipped CSV file (Fama/French 5 Factors (2x3) [Daily])
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    
    # Download the zip file
    response = requests.get(url)
    response.raise_for_status()  # ensure we notice bad responses
    
    # Use BytesIO to handle the downloaded content as a file-like object
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # List files in the zip archive (typically there's one CSV file)
        file_list = z.namelist()
        print("Files in zip:", file_list)
        
        # Open the first (or appropriate) CSV file from the archive
        with z.open(file_list[0]) as f:
            # Often the file may contain header rows that you might need to skip.
            # Adjust 'skiprows' based on the file's structure.
            df = pd.read_csv(f, skiprows=3)
    
    # Set index to properly formatted pd.PeriodIndex
    df.index = pd.PeriodIndex(pd.to_datetime(df[df.columns[0]], format='%Y%m%d'), freq='D', name='Day')
    df = df.drop(columns=df.columns[0])
    df = df.sort_index()
    
    return df

# Download and process the data only once at startup
print("Downloading Fama-French data...")
df = download_ff_data()
print("Data processing complete.")

# Initialize the Dash app
dash_app = dash.Dash(__name__)

# Configure for Render deployment
app = dash_app.server  # This is the Flask server that Dash uses

print("Past server")

# Convert PeriodIndex to DatetimeIndex for compatibility with Plotly
df_datetime = df.copy()
df_datetime.index = df_datetime.index.to_timestamp()

# Get date range boundaries for the input controls
min_date = df_datetime.index.min()
max_date = df_datetime.index.max()
min_date_str = min_date.strftime('%Y-%m-%d')
max_date_str = max_date.strftime('%Y-%m-%d')

# dash_app LAYOUT
dash_app.layout = html.Div([
    # Dashboard title
    html.H1(
        "Fama-French factor returns",
        style={'textAlign': 'center'}
    ),
    
    # User controls section
    html.Div([
        # Date range selection controls
        html.Div([
            html.Label("Start date:", style={'marginRight': '5px'}),
            dcc.Input(
                id="start-date-input", 
                type="text", 
                value=min_date_str,
                placeholder="YYYY-MM-DD",
                style={'marginRight': '20px'}
            ),
            html.Label("End date:", style={'marginRight': '5px'}),
            dcc.Input(
                id="end-date-input", 
                type="text", 
                value=max_date_str,
                placeholder="YYYY-MM-DD"
            ),
        ], style={'marginBottom': '10px'}),
        
        # Y-axis scale selection (linear vs logarithmic)
        html.Div([
            html.Label(
                "Log scale:", 
                style={'marginRight': '5px', 'display': 'inline-block'}
            ),
            dcc.Checklist(
                id="log-scale-checkbox", 
                options=[{'label': '', 'value': 'log'}], 
                value=[],
                style={'display': 'inline-block', 'verticalAlign': 'middle'}
            ),
        ], style={'marginBottom': '15px'}),
        
        # Container for validation warnings
        html.Div(id="warning-message", style={'color': 'red'}),
        
        # Main visualization
        dcc.Graph(id="cumulative-returns-plot", style={'height': '600px'}),
    ])
])

# CALLBACK FUNCTION
@dash_app.callback(
    [
        Output("cumulative-returns-plot", "figure"),
        Output("warning-message", "children")
    ],
    [
        Input("start-date-input", "value"),
        Input("end-date-input", "value"),
        Input("log-scale-checkbox", "value")
    ]
)
def update_graph(start_date_str, end_date_str, use_log_scale):
    """
    Updates the graph based on user inputs.
    
    Parameters:
    -----------
    start_date_str : str
        User-provided start date in string format
    end_date_str : str
        User-provided end date in string format
    use_log_scale : list
        Contains 'log' if logarithmic scale is selected, empty otherwise
        
    Returns:
    --------
    tuple
        (Plotly figure object, Warning messages div)
    """
    # Process checkbox value to boolean
    use_log_scale = 'log' in use_log_scale if use_log_scale else False
    warnings = []

    # Check for empty inputs and use defaults if needed
    if not start_date_str or start_date_str.strip() == "":
        start_date_str = min_date_str
        
    if not end_date_str or end_date_str.strip() == "":
        end_date_str = max_date_str
    
    # DATE VALIDATION
    # Validate and process start date
    try:
        start_date = pd.to_datetime(start_date_str)
        if start_date < min_date:
            warnings.append(f"Invalid start date. Please enter a date on or after {min_date_str}. Visualization proceeding with {min_date_str}.")
            start_date = min_date
    except:
        warnings.append(f"Invalid start date. Please enter a date on or after {min_date_str} in the format YYYY-MM-DD. Visualization proceeding with {min_date_str}.")
        start_date = min_date
    
    # Validate and process end date
    try:
        end_date = pd.to_datetime(end_date_str)
        if end_date > max_date:
            warnings.append(f"Invalid end date. Please enter an end date on or before {max_date_str}. Visualization proceeding with {max_date_str}.")
            end_date = max_date
    except:
        warnings.append(f"Invalid end date. Please enter and end date on or before {max_date_str} in the format YYYY-MM-DD. Visualization proceeding with {max_date_str}.")
        end_date = max_date
    
    # DATA FILTERING
    # Find closest available dates in the dataset
    valid_start_date = df_datetime.index.asof(start_date)
    valid_end_date = df_datetime.index.asof(end_date)
    
    # Filter data to selected date range
    filtered_df = df_datetime.loc[valid_start_date:valid_end_date]
    
    # CALCULATE RETURNS
    # Initialize DataFrames to store results
    raw_cum_returns = pd.DataFrame(index=filtered_df.index, columns=filtered_df.columns)
    
    # Calculate cumulative returns for each column (strategy/factor)
    for column in filtered_df.columns:
        # Convert percentage returns to cumulative raw returns
        # Formula: (1 + r₁/100) × (1 + r₂/100) × ... × (1 + rₙ/100)
        raw_cum_returns[column] = np.cumprod(1 + filtered_df[column] / 100)
    
    # CREATE VISUALIZATION
    fig = go.Figure()

    # For log scale, find the min and max of raw returns
    min_value = min([min(raw_cum_returns[col]) for col in raw_cum_returns.columns])
    max_value = max([max(raw_cum_returns[col]) for col in raw_cum_returns.columns])

    if use_log_scale:
        y_tick_values = 10.**np.linspace(start=np.log10(min_value), stop=np.log10(max_value), num=10, endpoint=True)
    else:
        y_tick_values = np.linspace(start=min_value, stop=max_value, num=10, endpoint=True)
    
    # Append 1 to the array
    y_tick_values = np.append(y_tick_values, 1)
    
    y_tick_values = np.sort(y_tick_values)
    y_tick_text = [f'{np.around((y-1)*100).astype(int)}%' for y in y_tick_values]

    # Add a trace for each column/strategy
    for column in raw_cum_returns.columns:
        fig.add_trace(go.Scatter(
            x=raw_cum_returns.index,
            y=raw_cum_returns[column],
            mode="lines",
            name=column
        ))
    
    # Configure layout and title
    title = f"Cumulative returns"
    title += f" from {valid_start_date.strftime('%Y-%m-%d')} to {valid_end_date.strftime('%Y-%m-%d')}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Percentage return (%)",
        yaxis=dict(
            type="log" if use_log_scale else "linear",
            tickmode="array",
            tickvals=y_tick_values,
            ticktext=y_tick_text
        ),
        hovermode="x unified",  # Show all values for same x coordinate
        legend={'orientation': 'h', 'y': 1.1},  # Horizontal legend above chart
    )
    
    # Add reference line at y=1 (log scale) or y=0 (linear scale)
    # This represents the break-even point (no gain, no loss)
    ref_y = 1
    fig.add_shape(
        type="line",
        x0=filtered_df.index[0],
        y0=ref_y,
        x1=filtered_df.index[-1],
        y1=ref_y,
        line=dict(color="#707070", width=1, dash="dash"),
    )
    
    # Return the figure and any warning messages
    return fig, html.Div([html.P(w) for w in warnings])

# Only run the server if this file is executed directly
if __name__ == '__main__':
    dash_app.run(debug=True)
