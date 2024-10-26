from django.shortcuts import render, redirect
from django.urls import reverse
from coordinate_systems.python_scripts.coordinate_systems.program_transformacija3D import transformacija_D48_D96
from coordinate_systems.python_scripts.coordinate_systems.kart2proj import kart2proj_fun
from coordinate_systems.python_scripts.coordinate_systems.proj2kart import proj2kart_fun, DolMer, GK2FLh
from .models import YourModel, TextFile
from django.http import HttpResponse
from .forms import CoordinateFileForm
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import json
import math



coordinate_systems = {
    "D48": "This is the previously used coordinate system we used in Slovenia.",
    "D96": "This is the coodinate system we use today.",
    "UTM": "This coordinate system is used in the NATO military.",
    "Transformations": "Convert coordinates from two different coordinate systems."
}



# Create your views here.
def starting_page(request):
    coordinate_system = list(coordinate_systems.keys())

    return render(request, "coordinate_systems/index.html", {
        "Coordinate_systems": coordinate_system
    })

def upload_files(request):
    form = CoordinateFileForm(request.POST or None, request.FILES or None)

    if request.method == 'POST':
        #request.FILES only works for the POST method
        form = CoordinateFileForm(request.POST, request.FILES)

        if form.is_valid():
            # Create a model instance and save the files if needed
            instance = YourModel(
                D48_file=request.FILES['D48'],
                D96_file=request.FILES['D96']
            )
            instance.save()

            # Get the file paths for your transformation function
            D48_file_path = instance.D48_file.path
            D96_file_path = instance.D96_file.path

            # Call your existing transformation function with the file paths
            transform_result = transformacija_D48_D96(D48_file_path, D96_file_path)

            return redirect(reverse('coordinates') + (f'?result={transform_result}' if transform_result is not None else ''))

    return render(request, 'coordinate_systems/input_template.html', {'form': form})


def success_view(request):
    file_path = "rezultati.txt" 
    
    with open(file_path, 'r') as file:
        #file.read() - fine for smaller files => use chunks for larger files
        content = file.read()

    text_file = TextFile.objects.create(content=content)

    return render(request, 'coordinate_systems/coordinates.html', {'text_file': text_file})

def download_text_file(request):
    file_path = "rezultati.txt"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            response = HttpResponse(file.read(), content_type='text/plain')
            #file.read() - fine for smaller files => use chunks for larger files
            response['Content-Disposition'] = 'attachment; filename="rezultati.txt"'
            return response
    else:
        return HttpResponse("The requested file does not exist.", status=404)

def D48_GK(request):
    return render(request, "coordinate_systems/D48_GK.html")

def D96_TM(request):
    return render(request, "coordinate_systems/D96_TM.html")

def UTM(request):
    return render(request, "coordinate_systems/UTM.html")

def differences_2d(request):
    with open('rezultati.json', 'r') as file:
        json_data = json.load(file)

    # Extract data for plotting
    input_data_e = [float(e) for e in json_data["INPUT_DATA_D96_17_TM"]["e[m]"]]
    input_data_n = [float(n) for n in json_data["INPUT_DATA_D96_17_TM"]["n[m]"]]
    transformed_data_e = [float(e) for e in json_data["Transformed_Coordinates_D96_TM"]["e[m]"]]
    transformed_data_n = [float(n) for n in json_data["Transformed_Coordinates_D96_TM"]["n[m]"]]

    # Create the figure
    fig = make_subplots(rows=1, cols=1)

    max_e = float(max(input_data_e + transformed_data_e))
    max_n = float(max(input_data_n + transformed_data_n))

    min_e = float(min(input_data_e + transformed_data_e))
    min_n = float(min(input_data_n + transformed_data_n))

    max_distance = math.sqrt((max_e - min_e)**2 + (max_n - min_n)**2)

    if max_distance > 2500:
        # Scale factor for connection vectors
        # Default scale factor for connection vectors (or read from request parameters)
        scale_factor_conn = float(request.GET.get('scale_factor', 200))  # Read the scaling factor from request or set default

        diff_e_li = []
        diff_n_li = []

        for i in range(len(input_data_e)):
            diff_e = transformed_data_e[i] - input_data_e[i]
            diff_n = transformed_data_n[i] - input_data_n[i]
            diff_e_li.append(diff_e)
            diff_n_li.append(diff_n)

        avg_e = (sum(diff_e_li))/len(input_data_e)
        avg_n = (sum(diff_n_li))/len(input_data_e)

        avg_diff = math.sqrt(avg_e**2 + avg_n**2)
        avg_diff = round(avg_diff, 3)

        # Scale the average difference based on scale factor
        scaled_avg_diff = avg_diff * scale_factor_conn

        # Calculate scaled connection vectors
        scaled_conn_e = [(transformed - original) * scale_factor_conn for original, transformed in zip(input_data_e, transformed_data_e)]
        scaled_conn_n = [(transformed - original) * scale_factor_conn for original, transformed in zip(input_data_n, transformed_data_n)]

        # Plot original points with labels above
        for i in range(len(input_data_e)):
            fig.add_trace(go.Scatter(
                x=[input_data_e[i]],
                y=[input_data_n[i]],
                mode='markers+text',
                text=f'Point {i+1}',
                textposition='top center',
                name='Input',
                showlegend=False
            ))

        # Plot transformed points with labels below
        for i in range(len(transformed_data_e)):
            fig.add_trace(go.Scatter(
                x=[transformed_data_e[i]],
                y=[transformed_data_n[i]],
                mode='markers',
                text=f'Transformed {i+1}',
                textposition='bottom center',
                name='Transformed',
                showlegend=False
            ))

        # Plot scaled connection vectors
        for i in range(len(input_data_e)):
            fig.add_trace(go.Scatter(x=[input_data_e[i], input_data_e[i] + scaled_conn_e[i]], 
                                    y=[input_data_n[i], input_data_n[i] + scaled_conn_n[i]], 
                                    mode='lines', 
                                    name=f'Connection {i+1}',
                                    line=dict(color='blue', width=1),
                                    showlegend=False))

        # Add a separate trace for scale_factor_conn
        fig.add_trace(go.Scatter(x=[min(input_data_e)], y=[min(input_data_n)], mode='lines', 
                                name=f'Scale for differences: {scaled_avg_diff}m',
                                marker=dict(color='blue', size=10),
                                showlegend=True))

        # Add the scale bar
        fig.add_shape(type="line",
            x0=min(input_data_e), y0=min(input_data_n) - 5, 
            x1=min(input_data_e) + scaled_avg_diff, y1=min(input_data_n) - 5,
            line=dict(color="black", width=3)
        )

        # Customize layout
        fig.update_layout(
            title='Original vs Transformed Points',
            xaxis_title='Easting (m)',
            yaxis_title='Northing (m)',
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                xanchor="right"
            )
        )

        # Convert the plot to JSON format
        plot_json = fig.to_json()
        return render(request, "coordinate_systems/coordinates.html", {'plot_json': plot_json})

    else:
        # Scale factor for connection vectors
        scale_factor_conn = 1  # Adjust as needed

        diff_e_li = []
        diff_n_li = []

        for i in range(len(input_data_e)):
            diff_e = transformed_data_e[i] - input_data_e[i]
            diff_n = transformed_data_n[i] - input_data_n[i]
            diff_e_li.append(diff_e)
            diff_n_li.append(diff_n)

        avg_e = (sum(diff_e_li))/len(input_data_e)
        avg_n = (sum(diff_n_li))/len(input_data_e)

        avg_diff = math.sqrt(avg_e**2 + avg_n**2)
        avg_diff = round(avg_diff, 3)

        # Calculate scaled connection vectors
        scaled_conn_e = [(transformed - original) * scale_factor_conn for original, transformed in zip(input_data_e, transformed_data_e)]
        scaled_conn_n = [(transformed - original) * scale_factor_conn for original, transformed in zip(input_data_n, transformed_data_n)]

        # Create the figure
        fig = go.Figure()

        # Plot original points with labels above
        for i in range(len(input_data_e)):
            fig.add_trace(go.Scatter(
                x=[input_data_e[i]],
                y=[input_data_n[i]],
                mode='markers+text',
                text=f'Point {i+1}',
                textposition='top center',
                name='Input',
                showlegend=False
            ))

        # Plot transformed points with labels below
        for i in range(len(transformed_data_e)):
            fig.add_trace(go.Scatter(
                x=[transformed_data_e[i]],
                y=[transformed_data_n[i]],
                mode='markers',
                text=f'Transformed {i+1}',
                textposition='bottom center',
                name='Transformed',
                showlegend=False
            ))

        # Plot scaled connection vectors
        for i in range(len(input_data_e)):
            fig.add_trace(go.Scatter(x=[input_data_e[i], input_data_e[i] + scaled_conn_e[i]], 
                                    y=[input_data_n[i], input_data_n[i] + scaled_conn_n[i]], 
                                    mode='lines', 
                                    name=f'Connection {i+1}',
                                    line=dict(color='blue', width=1),
                                    showlegend=False))

        # Add a separate trace for scale_factor_conn
        fig.add_trace(go.Scatter(x=[min(input_data_e)],
                                  y=[min(input_data_n)], 
                                  mode='lines', 
                                    name=f'Scale for differences: {avg_diff}',
                                    marker=dict(color='blue', size=10),
                                    showlegend=True))

        # Add the scale bar
        fig.add_shape(type="line",
            x0=min(input_data_e), y0=min(input_data_n) - 5, 
            x1=min(input_data_e) + avg_diff, y1=min(input_data_n) - 5,
            line=dict(color="black", width=3)
        )

        # Customize layout
        fig.update_layout(
            title='Original vs Transformed Points',
            xaxis_title='Easting (m)',
            yaxis_title='Northing (m)',
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                xanchor="right"
            )
        )

        # Convert the plot to JSON format
        plot_json = fig.to_json()
        return render(request, "coordinate_systems/coordinates.html", {'plot_json': plot_json})