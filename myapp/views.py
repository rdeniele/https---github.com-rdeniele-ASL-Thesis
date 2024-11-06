import logging
from threading import Lock

from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse, HttpResponseServerError

from .forms import RegisterForm
from .detection import ASLDetector

logger = logging.getLogger(__name__)

# Use a dictionary to store detector instances per user to ensure each user has a separate instance of the detector,
# which helps in managing user-specific detection sessions and avoids conflicts between different users' data.
detectors = {}
detector_locks = {}


def get_detector(request):
    user_id = request.user.id
    if user_id not in detectors:
        detectors[user_id] = ASLDetector()
    if user_id not in detector_locks:
        detector_locks[user_id] = Lock()

    return detectors[user_id], detector_locks[user_id]


def home(request):
    return render(request, 'home.html')

def dashboard(request):
    return render(request, 'dashboard.html') #new


def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('myapp:dashboard')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('myapp:dashboard')  # Ensure this matches the URL pattern name    #new
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def user_logout(request):
    logout(request)
    return redirect('myapp:home')  # Ensure this matches the URL pattern name

@login_required
def video_feed(request):
    """
    Stream video frames from the ASL detector to the client.
    """
    detector, detector_lock = get_detector(request)
    logger.info("Video feed endpoint accessed")

    def generate_frames():
        try:
            while True:
                with detector_lock:
                    frame, prediction, confidence = detector.get_frame()
                    if frame is None:
                        logger.error("Received None frame")
                        break
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(f"Error in generate_frames: {e}")
        finally:
            with detector_lock:
                detector.cleanup()

    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@login_required
def cleanup_view(request):
    """
    Clean up the ASL detector instance for the current user and remove it from the detectors dictionary.
    """
    detector, detector_lock = get_detector(request)
    with detector_lock:
        detector.cleanup()
        del detectors[request.user.id]  # Remove detector instance
        del detector_locks[request.user.id]  # Remove corresponding lock
    return redirect('dashboard') #new


@login_required
def get_prediction(request):
    """
    Get the prediction and confidence from the ASL detector for the current frame.
    """
    detector, detector_lock = get_detector(request)

    with detector_lock:
        _, prediction, confidence = detector.get_frame()

    return JsonResponse({"prediction": prediction, "confidence": confidence})
