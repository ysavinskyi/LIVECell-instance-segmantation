from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import predict, preprocess_image
import numpy as np
import traceback
from PIL import Image
import io


@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        try:
            image_file = request.FILES['file']
            # Read the image file
            image = Image.open(image_file)
            image = np.array(image)

            # Make predictions
            predictions = predict(image)

            # Clip predictions to be within the range [0, 1]
            predictions = np.clip(predictions, 0, 1)

            # Scale predictions to the range [0, 255]
            predictions = (predictions * 255).astype(np.uint8)

            # Convert predictions to list for JSON serialization
            prediction_list = predictions[0].tolist()

            # Return the prediction list as JSON
            return JsonResponse({'predictions': prediction_list})
        except Exception as e:
            error_trace = traceback.format_exc()
            print(error_trace)  # Print the traceback to the server log
            return JsonResponse({'error': str(e), 'trace': error_trace}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)
