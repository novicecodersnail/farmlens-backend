import azure.functions as func
import logging, os, requests, json, datetime,base64
import traceback, io, tempfile
from requests_toolbelt.multipart.encoder import MultipartEncoder
from inference_sdk import InferenceHTTPClient
from roboflow import Roboflow

from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
from functools import partial
import time
import cv2
import supervision as sv
from datetime import datetime, timedelta
import numpy as np

app = func.FunctionApp()


@app.function_name(name="HttpTriggerTest")
@app.route(route="httptriggertest", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def test_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Attempting HTTP Python Trigger: Checking Auth')

    try:
        # Correct the method to access the environment variable and split it into a list
        allowed_values = os.environ.get("AUTH_VALUES", "").split(",")

        # Get the 'Authorization' header value from the request
        header_value = req.headers.get("Authorization")

        # Check if the header value is in the list of allowed values
        if header_value in allowed_values:
            return func.HttpResponse("This HTTP triggered function executed successfully from Azure Function.", status_code=200)
        else:
            # Log and return an unauthorized access error if the header value is not allowed
            logging.error("API Auth Error: You are not authorized")
            return func.HttpResponse("You are not authorized to access this function.", status_code=401)

    except Exception as e:
        # Log the exception details and return an internal server error response
        logging.error(f"An error occurred: {str(e)}")
        return func.HttpResponse(f"An internal error occurred: {str(e)}", status_code=500)
    
@app.route(route="TestClassifyStrawberryImageRoboflowJSON", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def TestClassifyStrawberryImageRoboflowJSON(req: func.HttpRequest) -> func.HttpResponse:
    try:
        header_value= req.headers.get("Authorization")
        allowed_values= os.environ["AUTH_VALUES",""].split(",")
        if header_value not in allowed_values:
            logging.error("API Auth Error: Unauthorized access attempt.")
            return func.HttpResponse("Unauthorized access.", status_code=401)

        # This should be the binary data of the image
        image_data = req.get_body()  
    
        m = MultipartEncoder(
            fields={'file': ('imageToUpload', image_data, 'image/jpeg')}
        )

        headers = {
            'Content-Type': m.content_type,
        }

        response = requests.post(
            
            f"{os.environ['ROBOFLOW_API_URL']}/{os.environ['ROBOFLOW_WORKSPACE']}/{os.environ['ROBOFLOW_MODEL_VERSION']}?api_key={os.environ['ROBOFLOW_API_KEY']}",
            headers=headers,
            data=m
        )

        if response.status_code == 200:
            return func.HttpResponse(body=response.content, status_code=200, mimetype="application/json")
        else:
            logging.error(f"API Response Error: {response.content}")
            return func.HttpResponse(body=response.content, status_code=response.status_code)
        
    except KeyError as e:
        logging.error(f"Configuration error: {e}")
        return func.HttpResponse(f"Server configuration error: {str(e)}", status_code=500)
    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}")
        return func.HttpResponse(f"An internal error occurred: {str(e)}", status_code=500)
            
    
@app.route(route="TestClassifyStrawberryImageRoboflowModel", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def TestClassifyStrawberryImageRoboflowModel(req: func.HttpRequest) -> func.HttpResponse:
    try:
        header_value= req.headers.get("Authorization")
        allowed_values = os.environ.get("AUTH_VALUES", "").split(",")

        if header_value not in allowed_values:
            logging.error("API Auth Error: Unauthorized access attempt.")
            return func.HttpResponse("Unauthorized access.", status_code=401)
        
      
        # Read binary image data from request
        image_data = req.get_body()

        # Write image data to temporary file for prediction
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            temp_image.write(image_data)
            temp_image_path = temp_image.name

        # Instantiate Roboflow model
        rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
        project = rf.workspace().project(os.environ["ROBOFLOW_WORKSPACE"])
        model = project.version(os.environ["ROBOFLOW_MODEL_VERSION"]).model

        # Make prediction using temporary file
        prediction = model.predict(temp_image_path)

        # Save prediction result to another temporary file
        with tempfile.NamedTemporaryFile(mode='wb+', delete=False, suffix='.jpg') as temp_file:
            prediction.save(temp_file.name)
            temp_file.seek(0)
            image_content = temp_file.read()

        # Clean up temporary files
        os.unlink(temp_image_path)
        os.unlink(temp_file.name)

        # Return image content in HTTP response
        return func.HttpResponse(body=image_content, status_code=200, mimetype='image/jpeg')
    except KeyError as e:
        logging.error(f"Configuration error: {e}")
        return func.HttpResponse(f"Server configuration error: {str(e)}", status_code=500)
    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}")
        return func.HttpResponse(f"An internal error occurred: {str(e)}", status_code=500)
    
@app.route(route="TestClassifyStrawberryVideo", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def TestClassifyStrawberryVideo(req: func.HttpRequest) -> func.HttpResponse:
    try:
        header_value = req.headers.get("Authorization")
        allowed_values = os.environ.get("AUTH_VALUES", "").split(",")

        if header_value not in allowed_values:
            return func.HttpResponse("Unauthorized access.", status_code=401)

        # Read binary video data from the request and save to temporary file
        video_data = req.get_body()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_data)
            video_path = temp_video.name

        #Temporary file for storing result from model 
        temp_dir = tempfile.gettempdir()
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=temp_dir)
        output_path = output_file.name
        output_file.close()

        # Setup Roboflow model
        rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
        project = rf.workspace().project(os.environ["ROBOFLOW_WORKSPACE"])
        model = project.version(os.environ["ROBOFLOW_MODEL_VERSION"]).model

        # Start video processing
        job_id, signed_url, expire_time = model.predict_video(
            video_path, fps=5, prediction_type="batch-video"
        )
        logging.info(f"Video processing started: job_id={job_id}, signed_url={signed_url}")

        # Poll for results
        results = model.poll_until_video_results(job_id)
        if not results or 'strawberry-object-detection' not in results:
            return func.HttpResponse("Processing results are not available.", status_code=404)
        
        # Display contents of roboflow response which is dict of : frame offset, time offset, strawberry-object-detections
        logging.info(f"{print(results)}")


        # Annotate video
        annotate_video(video_path, results['strawberry-object-detection'],results['frame_offset'], output_path)
        logging.info("----finisihed processing results------------")

        # Serve processed video file directly
        with open(output_path, 'rb') as vid:
            video_content = vid.read()

        # Clean up temporary files
        os.unlink(video_path)
        os.unlink(output_path)

        return func.HttpResponse(body=video_content, status_code=200, mimetype='video/mp4')

    except KeyError as e:
        logging.error(f"Configuration error: {str(e)}")
        return func.HttpResponse(f"Server configuration error: {str(e)}", status_code=500)
    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}")
        return func.HttpResponse(f"An internal error occurred: {str(e)}", status_code=500)
    
def annotate_video(video_path, detections, frame_offsets, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

    x_offset = -10  # Adjust these values based on your observations
    y_offset = -10
    frame_count = 0
    current_detection_index = 0
    last_boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame index is one where detections are available
        if current_detection_index < len(frame_offsets) and frame_count == frame_offsets[current_detection_index]:
            frame_detections = detections[current_detection_index]['predictions']
            last_boxes = frame_detections
            current_detection_index += 1
        elif last_boxes:
            frame_detections = last_boxes  # Use last known boxes if no new data is available
        
        # Draw bounding boxes and labels on the frame
        for pred in frame_detections:
            x, y, w, h = int(pred['x'] + x_offset), int(pred['y'] + y_offset), int(pred['width']), int(pred['height'])
            color = (0, 255, 0) if pred['class'] == 'ripe' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{pred['class']} {pred['confidence']*100:.2f}%"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    return output_path