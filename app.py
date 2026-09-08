
"""
============================================================
DEEPFAKE DETECTION - FLASK BACKEND
============================================================

Available Detection:
1. Image Detection
2. Video Detection
3. Audio Detection

Image and Video detection use:
    model_utils.predict_image()
    model_utils.predict_video()

These functions are NOT changed.
============================================================
"""

import os
import time

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory
)

import model_utils


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path=""
)


# ============================================================
# MAXIMUM UPLOAD SIZE
# ============================================================

app.config["MAX_CONTENT_LENGTH"] = (
    100 * 1024 * 1024
)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXT = {

    "image": {
        "jpg",
        "jpeg",
        "png"
    },

    "video": {
        "mp4",
        "avi",
        "mov",
        "mkv",
        "webm"
    },

    "audio": {
        "mp3",
        "wav",
        "ogg"
    }
}


# ============================================================
# GET FILE EXTENSION
# ============================================================

def get_ext(filename):

    if not filename:
        return ""

    if "." not in filename:
        return ""

    return filename.rsplit(
        ".",
        1
    )[1].lower()


# ============================================================
# ERROR RESPONSE
# ============================================================

def error_response(
    message,
    status=400
):

    print()
    print("=" * 60)
    print("ERROR")
    print(message)
    print("=" * 60)

    return jsonify({

        "result": "error",

        "confidence": 0,

        "processing_time_ms": 0,

        "model": "Error",

        "error": message

    }), status


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def serve_home():

    return send_from_directory(
        STATIC_DIR,
        "index.html"
    )


# ============================================================
# STATIC FILES
# ============================================================

@app.route("/<path:filename>")
def serve_static_file(filename):

    return send_from_directory(
        STATIC_DIR,
        filename
    )


# ============================================================
# GENERIC FILE UPLOAD HANDLER
# ============================================================

def handle_upload_detection(
    kind,
    model_fn
):

    print()
    print("=" * 60)
    print("NEW DETECTION REQUEST")
    print("=" * 60)

    print(
        "Detection type:",
        kind
    )


    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "file" not in request.files:

        return error_response(
            "No file uploaded. "
            "The frontend must send the file "
            "using form field 'file'."
        )


    file = request.files["file"]


    if not file.filename:

        return error_response(
            "Empty filename."
        )


    # --------------------------------------------------------
    # GET EXTENSION
    # --------------------------------------------------------

    filename = file.filename

    ext = get_ext(
        filename
    )


    print(
        "Filename:",
        filename
    )

    print(
        "Extension:",
        ext
    )


    # --------------------------------------------------------
    # CHECK EXTENSION
    # --------------------------------------------------------

    if ext not in ALLOWED_EXT[kind]:

        allowed = ", ".join(
            sorted(
                ALLOWED_EXT[kind]
            )
        )

        return error_response(
            f"Unsupported file type "
            f"'.{ext}'. "
            f"Allowed types: {allowed}"
        )


    # --------------------------------------------------------
    # START TIMER
    # --------------------------------------------------------

    start_time = time.time()


    try:

        print()
        print(
            "Calling:",
            model_fn.__name__
        )


        # ====================================================
        # IMAGE / VIDEO / AUDIO MODEL
        # ====================================================

        result, confidence, model_name = (
            model_fn(file)
        )


        # ====================================================
        # PROCESSING TIME
        # ====================================================

        elapsed_ms = int(
            (time.time() - start_time)
            * 1000
        )


        print()
        print("=" * 60)
        print("DETECTION RESPONSE")
        print("=" * 60)

        print(
            "Result:",
            result
        )

        print(
            "Confidence:",
            confidence
        )

        print(
            "Model:",
            model_name
        )

        print(
            "Processing time:",
            elapsed_ms,
            "ms"
        )

        print("=" * 60)


        # ====================================================
        # JSON RESPONSE
        # ====================================================

        return jsonify({

            "result": result,

            "confidence": confidence,

            "processing_time_ms": elapsed_ms,

            "model": model_name

        })


    except Exception as e:

        print()
        print("=" * 60)
        print("MODEL ERROR")
        print("=" * 60)

        print(
            repr(e)
        )

        print("=" * 60)


        return error_response(
            "Detection failed: " + str(e),
            500
        )


# ============================================================
# IMAGE DETECTION
# ============================================================

@app.route(
    "/api/detect/image",
    methods=["POST"]
)
def detect_image():

    print()
    print(">>> IMAGE API CALLED")

    return handle_upload_detection(
        "image",
        model_utils.predict_image
    )


# ============================================================
# VIDEO DETECTION
# ============================================================

@app.route(
    "/api/detect/video",
    methods=["POST"]
)
def detect_video():

    print()
    print("=" * 60)
    print(">>> VIDEO API CALLED")
    print("=" * 60)

    return handle_upload_detection(
        "video",
        model_utils.predict_video
    )


# ============================================================
# AUDIO DETECTION
# ============================================================

@app.route(
    "/api/detect/audio",
    methods=["POST"]
)
def detect_audio():

    print()
    print(">>> AUDIO API CALLED")

    return handle_upload_detection(
        "audio",
        model_utils.predict_audio
    )


# ============================================================
# 404 ERROR
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({

        "result": "error",

        "error":
            "Page or API endpoint not found."

    }), 404


# ============================================================
# 413 ERROR - FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "result": "error",

        "error":
            "File is too large. "
            "Maximum size is 100 MB."

    }), 413


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("       DEEPFAKE DETECTION SERVER")
    print("=" * 60)

    print(
        "Base directory:",
        BASE_DIR
    )

    print(
        "Static directory:",
        STATIC_DIR
    )

    print(
        "Static directory exists:",
        os.path.exists(STATIC_DIR)
    )

    print()
    print("Available APIs:")

    print()
    print("Image:")
    print("POST /api/detect/image")

    print()
    print("Video:")
    print("POST /api/detect/video")

    print()
    print("Audio:")
    print("POST /api/detect/audio")


    print()
    print("=" * 60)
    print("Starting Flask server...")
    print("=" * 60)


    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
