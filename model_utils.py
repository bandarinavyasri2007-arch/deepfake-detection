# ============================================================
# model_utils.py
# DEEPFAKE DETECTION
# IMAGE + VIDEO + AUDIO
#
# IMAGE:
#   Uses image_model.pt
#
# VIDEO:
#   Uses the same image_model.pt
#
# AUDIO:
#   No trained audio model available
#
#
# IMAGE + VIDEO CODE IS KEPT UNCHANGED.
# ============================================================


import os
import tempfile

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms


# ============================================================
# SETTINGS
# ============================================================

# ------------------------------------------------------------
# IMAGE + VIDEO MODEL
# ------------------------------------------------------------

MODEL_PATH = "saved_models/image_model.pt"


# Maximum video frames

MAX_VIDEO_FRAMES = 40


# Video prediction batch size

VIDEO_BATCH_SIZE = 8


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 60)
print("       DEEPFAKE DETECTION MODEL")
print("=" * 60)

print(
    "Image/Video model path:",
    MODEL_PATH
)

print(
    "Using device:",
    device
)


# ============================================================
# CHECK IMAGE/VIDEO MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n\n"
        "Please run:\n"
        "python train_image_model.py"
    )


# ============================================================
# LOAD IMAGE/VIDEO MODEL
# ============================================================

print("Loading trained image/video model...")

try:

    image_model = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

except TypeError:

    image_model = torch.load(
        MODEL_PATH,
        map_location=device
    )


image_model = image_model.to(device)

image_model.eval()

print(
    "Image/video model loaded successfully!"
)


# ============================================================
# CLASS MAPPING
# ============================================================

# Confirmed from training:
#
# fake = 0
# real = 1

CLASS_NAMES = [
    "fake",
    "real"
]


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# IMAGE DETECTION
# ============================================================

# DO NOT CHANGE THE IMAGE DETECTION LOGIC.

def predict_image(file_storage):

    try:

        print()
        print("=" * 60)
        print("IMAGE ANALYSIS")
        print("=" * 60)

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            file_storage.stream
        ).convert("RGB")

        print(
            "Image size:",
            image.size
        )

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        image_tensor = image_transform(
            image
        )

        image_tensor = image_tensor.unsqueeze(
            0
        )

        image_tensor = image_tensor.to(
            device
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        image_model.eval()

        with torch.no_grad():

            outputs = image_model(
                image_tensor
            )

            probabilities = F.softmax(
                outputs,
                dim=1
            )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        fake_probability = (
            probabilities[0][0].item()
        )

        real_probability = (
            probabilities[0][1].item()
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        result = CLASS_NAMES[
            predicted_class
        ]

        confidence = max(
            fake_probability,
            real_probability
        ) * 100

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            "MODEL OUTPUTS:",
            outputs
        )

        print(
            "PROBABILITIES:",
            probabilities
        )

        print(
            f"Fake Probability: "
            f"{fake_probability * 100:.2f}%"
        )

        print(
            f"Real Probability: "
            f"{real_probability * 100:.2f}%"
        )

        print(
            "Predicted class:",
            predicted_class
        )

        print(
            "Predicted result:",
            result
        )

        print(
            f"Confidence: "
            f"{confidence:.2f}%"
        )

        print("=" * 60)

        return (
            result,
            round(
                confidence,
                1
            ),
            "Custom ResNet18 Deepfake Detector"
        )

    except Exception as e:

        print()
        print("=" * 60)
        print("IMAGE DETECTION ERROR")
        print("=" * 60)

        print(
            str(e)
        )

        print("=" * 60)

        return (
            "error",
            0.0,
            "Image detection failed"
        )


# ============================================================
# VIDEO FRAME SELECTION
# ============================================================

def _select_video_frames(
    total_frames,
    max_frames
):

    if total_frames <= 0:

        return []

    number_of_frames = min(
        total_frames,
        max_frames
    )

    if number_of_frames == 1:

        return [0]

    positions = []

    for i in range(
        number_of_frames
    ):

        position = int(
            i * (total_frames - 1)
            /
            (number_of_frames - 1)
        )

        positions.append(
            position
        )

    return list(
        dict.fromkeys(
            positions
        )
    )


# ============================================================
# VIDEO PREDICTION
# ============================================================

def predict_video(file_storage):

    import cv2

    temp_path = None
    video = None

    try:

        print()
        print("=" * 60)
        print("          VIDEO DEEPFAKE ANALYSIS")
        print("=" * 60)

        # ====================================================
        # SAVE TEMPORARY VIDEO
        # ====================================================

        filename = getattr(
            file_storage,
            "filename",
            ""
        )

        extension = os.path.splitext(
            filename
        )[1].lower()

        allowed_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm"
        }

        if extension not in allowed_extensions:

            extension = ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_path = temp_file.name

            file_storage.save(
                temp_path
            )

        print(
            "Temporary video:",
            temp_path
        )

        # ====================================================
        # OPEN VIDEO
        # ====================================================

        video = cv2.VideoCapture(
            temp_path
        )

        if not video.isOpened():

            return (
                "error",
                0.0,
                "Unable to open video"
            )

        # ====================================================
        # VIDEO INFORMATION
        # ====================================================

        total_frames = int(
            video.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = video.get(
            cv2.CAP_PROP_FPS
        )

        if fps is None or fps <= 0:

            fps = 25.0

        duration = (
            total_frames / fps
            if total_frames > 0
            else 0
        )

        print(
            "Total frames:",
            total_frames
        )

        print(
            "FPS:",
            round(
                fps,
                2
            )
        )

        print(
            "Duration:",
            round(
                duration,
                2
            ),
            "seconds"
        )

        # ====================================================
        # CHECK VIDEO
        # ====================================================

        if total_frames <= 0:

            return (
                "error",
                0.0,
                "Video contains no frames"
            )

        # ====================================================
        # SELECT FRAMES
        # ====================================================

        frame_positions = _select_video_frames(
            total_frames,
            MAX_VIDEO_FRAMES
        )

        print(
            "Frames selected:",
            len(frame_positions)
        )

        # ====================================================
        # EXTRACT FRAMES
        # ====================================================

        frame_tensors = []

        frame_numbers = []

        for frame_number in frame_positions:

            video.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            success, frame = video.read()

            if not success:

                print(
                    f"Frame {frame_number}: "
                    "could not be read"
                )

                continue

            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # RGB -> PIL
            # ------------------------------------------------

            image = Image.fromarray(
                frame_rgb
            ).convert("RGB")

            # ------------------------------------------------
            # SAME TRANSFORM AS IMAGE
            # ------------------------------------------------

            tensor = image_transform(
                image
            )

            frame_tensors.append(
                tensor
            )

            frame_numbers.append(
                frame_number
            )

        # ====================================================
        # RELEASE VIDEO
        # ====================================================

        video.release()

        video = None

        # ====================================================
        # CHECK
        # ====================================================

        if len(frame_tensors) == 0:

            return (
                "error",
                0.0,
                "No frames could be extracted"
            )

        print(
            "Successfully extracted:",
            len(frame_tensors),
            "frames"
        )

        # ====================================================
        # PREDICTION STORAGE
        # ====================================================

        fake_probabilities = []

        real_probabilities = []

        # ====================================================
        # BATCH PREDICTION
        # ====================================================

        for start in range(
            0,
            len(frame_tensors),
            VIDEO_BATCH_SIZE
        ):

            batch_tensors = frame_tensors[
                start:
                start + VIDEO_BATCH_SIZE
            ]

            batch = torch.stack(
                batch_tensors
            )

            batch = batch.to(
                device
            )

            with torch.no_grad():

                outputs = image_model(
                    batch
                )

                probabilities = F.softmax(
                    outputs,
                    dim=1
                )

            # ------------------------------------------------
            # SAVE FRAME PROBABILITIES
            # ------------------------------------------------

            for i in range(
                probabilities.shape[0]
            ):

                fake_probability = (
                    probabilities[i][0].item()
                )

                real_probability = (
                    probabilities[i][1].item()
                )

                fake_probabilities.append(
                    fake_probability
                )

                real_probabilities.append(
                    real_probability
                )

                frame_number = (
                    frame_numbers[
                        start + i
                    ]
                )

                if fake_probability >= real_probability:

                    frame_result = "FAKE"

                else:

                    frame_result = "REAL"

                print(
                    f"Frame {frame_number:5d} | "
                    f"Fake: "
                    f"{fake_probability * 100:6.2f}% | "
                    f"Real: "
                    f"{real_probability * 100:6.2f}% | "
                    f"{frame_result}"
                )

        # ====================================================
        # CHECK
        # ====================================================

        processed_frames = len(
            fake_probabilities
        )

        if processed_frames == 0:

            return (
                "error",
                0.0,
                "No frames were predicted"
            )

        # ====================================================
        # BASIC AVERAGE
        # ====================================================

        average_fake = (
            sum(fake_probabilities)
            /
            processed_frames
        )

        average_real = (
            sum(real_probabilities)
            /
            processed_frames
        )

        # ====================================================
        # FRAME VOTES
        # ====================================================

        fake_votes = 0
        real_votes = 0

        for fake_probability, real_probability in zip(
            fake_probabilities,
            real_probabilities
        ):

            if fake_probability >= real_probability:

                fake_votes += 1

            else:

                real_votes += 1

        # ====================================================
        # VOTE PERCENTAGES
        # ====================================================

        fake_vote_percentage = (
            fake_votes
            /
            processed_frames
        )

        real_vote_percentage = (
            real_votes
            /
            processed_frames
        )

        # ====================================================
        # STRONG FAKE FRAME ANALYSIS
        # ====================================================

        strong_fake_threshold = 0.70

        strong_real_threshold = 0.70

        strong_fake_frames = sum(
            1
            for p in fake_probabilities
            if p >= strong_fake_threshold
        )

        strong_real_frames = sum(
            1
            for p in real_probabilities
            if p >= strong_real_threshold
        )

        strong_fake_percentage = (
            strong_fake_frames
            /
            processed_frames
        )

        strong_real_percentage = (
            strong_real_frames
            /
            processed_frames
        )

        # ====================================================
        # TOP FAKE PROBABILITIES
        # ====================================================

        sorted_fake = sorted(
            fake_probabilities,
            reverse=True
        )

        top_count = max(
            1,
            int(
                len(sorted_fake) * 0.25
            )
        )

        top_fake_average = (
            sum(
                sorted_fake[:top_count]
            )
            /
            top_count
        )

        # ====================================================
        # TOP REAL PROBABILITIES
        # ====================================================

        sorted_real = sorted(
            real_probabilities,
            reverse=True
        )

        top_real_average = (
            sum(
                sorted_real[:top_count]
            )
            /
            top_count
        )

        # ====================================================
        # VIDEO SCORES
        # ====================================================

        fake_video_score = (

            average_fake * 0.50

            +

            strong_fake_percentage * 0.20

            +

            top_fake_average * 0.30
        )

        real_video_score = (

            average_real * 0.50

            +

            strong_real_percentage * 0.20

            +

            top_real_average * 0.30
        )

        # ====================================================
        # FINAL VIDEO RESULT
        # ====================================================

        if fake_video_score >= real_video_score:

            final_result = "fake"

            final_confidence = (
                fake_video_score * 100
            )

        else:

            final_result = "real"

            final_confidence = (
                real_video_score * 100
            )

        # ====================================================
        # PRINT ANALYSIS
        # ====================================================

        print()
        print("=" * 60)
        print("          VIDEO ANALYSIS COMPLETE")
        print("=" * 60)

        print(
            "Processed frames:",
            processed_frames
        )

        print()

        print(
            f"Average Fake Probability: "
            f"{average_fake * 100:.2f}%"
        )

        print(
            f"Average Real Probability: "
            f"{average_real * 100:.2f}%"
        )

        print()

        print(
            f"Fake Votes: "
            f"{fake_votes}/{processed_frames}"
        )

        print(
            f"Real Votes: "
            f"{real_votes}/{processed_frames}"
        )

        print()

        print(
            f"Strong Fake Frames: "
            f"{strong_fake_frames}/{processed_frames}"
        )

        print(
            f"Strong Real Frames: "
            f"{strong_real_frames}/{processed_frames}"
        )

        print()

        print(
            f"Top Fake Frame Average: "
            f"{top_fake_average * 100:.2f}%"
        )

        print(
            f"Top Real Frame Average: "
            f"{top_real_average * 100:.2f}%"
        )

        print()

        print(
            f"FINAL FAKE VIDEO SCORE: "
            f"{fake_video_score * 100:.2f}%"
        )

        print(
            f"FINAL REAL VIDEO SCORE: "
            f"{real_video_score * 100:.2f}%"
        )

        print()

        print(
            "FINAL VIDEO RESULT:",
            final_result.upper()
        )

        print(
            f"FINAL VIDEO CONFIDENCE: "
            f"{final_confidence:.2f}%"
        )

        print("=" * 60)

        # ====================================================
        # RETURN
        # ====================================================

        return (
            final_result,
            round(
                final_confidence,
                1
            ),
            "ResNet18 Video Frame Deepfake Detector"
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print()
        print("=" * 60)
        print("VIDEO DETECTION ERROR")
        print("=" * 60)

        print(
            str(e)
        )

        print("=" * 60)

        return (
            "error",
            0.0,
            "Video detection failed"
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        if video is not None:

            try:

                video.release()

            except Exception:

                pass

        if (
            temp_path
            and
            os.path.exists(temp_path)
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


# ============================================================
# AUDIO
# ============================================================

# ============================================================
# AUDIO DETECTION
# ============================================================

import tempfile
import os
import numpy as np
import librosa
import torch
import torch.nn as nn


# ------------------------------------------------------------
# AUDIO SETTINGS
# ------------------------------------------------------------

AUDIO_MODEL_PATH = "saved_models/audio_model.pt"

AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 5
AUDIO_NUM_SAMPLES = (
    AUDIO_SAMPLE_RATE * AUDIO_DURATION
)

AUDIO_N_MELS = 128
AUDIO_N_FFT = 1024
AUDIO_HOP_LENGTH = 256

AUDIO_CLASS_NAMES = [
    "fake",
    "real"
]


# ------------------------------------------------------------
# AUDIO CNN
# ------------------------------------------------------------

class AudioCNN(nn.Module):

    def __init__(self):

        super(AudioCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256 * 4 * 4,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ------------------------------------------------------------
# LOAD AUDIO MODEL
# ------------------------------------------------------------

audio_model = None

if os.path.exists(AUDIO_MODEL_PATH):

    try:

        audio_model = AudioCNN()

        audio_model.load_state_dict(
            torch.load(
                AUDIO_MODEL_PATH,
                map_location="cpu"
            )
        )

        audio_model.eval()

        print(
            "Audio model loaded successfully."
        )

    except Exception as e:

        print(
            "Error loading audio model:",
            e
        )

        audio_model = None

else:

    print(
        "Audio model not found:",
        AUDIO_MODEL_PATH
    )


# ------------------------------------------------------------
# AUDIO PREDICTION
# ------------------------------------------------------------

def predict_audio(file_storage):

    if audio_model is None:

        return (
            "error",
            0.0,
            "Audio model not available"
        )

    temp_path = None

    try:

        # --------------------------------------------
        # Get original file extension
        # --------------------------------------------

        filename = file_storage.filename or "audio.wav"

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in [
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a"
        ]:

            extension = ".wav"


        # --------------------------------------------
        # Save temporary audio
        # --------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_path = temp_file.name

            file_storage.save(temp_path)


        # --------------------------------------------
        # Load audio
        # --------------------------------------------

        audio, sr = librosa.load(
            temp_path,
            sr=AUDIO_SAMPLE_RATE,
            mono=True
        )


        # --------------------------------------------
        # Make audio exactly 5 seconds
        # --------------------------------------------

        if len(audio) < AUDIO_NUM_SAMPLES:

            audio = np.pad(
                audio,
                (
                    0,
                    AUDIO_NUM_SAMPLES - len(audio)
                )
            )

        else:

            audio = audio[
                :AUDIO_NUM_SAMPLES
            ]


        # --------------------------------------------
        # Mel Spectrogram
        # --------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=AUDIO_SAMPLE_RATE,
            n_fft=AUDIO_N_FFT,
            hop_length=AUDIO_HOP_LENGTH,
            n_mels=AUDIO_N_MELS
        )


        # --------------------------------------------
        # Convert to dB
        # --------------------------------------------

        mel = librosa.power_to_db(
            mel,
            ref=np.max
        )


        # --------------------------------------------
        # Normalize
        # --------------------------------------------

        mel = (
            mel - mel.mean()
        ) / (
            mel.std() + 1e-8
        )


        # --------------------------------------------
        # Convert to Tensor
        # --------------------------------------------

        tensor = torch.tensor(
            mel,
            dtype=torch.float32
        )


        # Add channel dimension
        tensor = tensor.unsqueeze(0)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)


        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        with torch.no_grad():

            outputs = audio_model(
                tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]


        fake_probability = (
            probabilities[0].item()
        )

        real_probability = (
            probabilities[1].item()
        )


        # --------------------------------------------
        # Final result
        # --------------------------------------------

        if fake_probability > real_probability:

            result = "fake"

            confidence = (
                fake_probability * 100
            )

        else:

            result = "real"

            confidence = (
                real_probability * 100
            )


        print()
        print("========== AUDIO DETECTION ==========")
        print(
            "Fake Probability:",
            f"{fake_probability * 100:.2f}%"
        )
        print(
            "Real Probability:",
            f"{real_probability * 100:.2f}%"
        )
        print(
            "Predicted Result:",
            result
        )
        print(
            "Confidence:",
            f"{confidence:.2f}%"
        )
        print("=====================================")


        return (
            result,
            confidence,
            "AudioCNN"
        )


    except Exception as e:

        print(
            "Audio prediction error:",
            e
        )

        return (
            "error",
            0.0,
            str(e)
        )


    finally:

        # --------------------------------------------
        # Delete temporary file
        # --------------------------------------------

        if temp_path is not None:

            try:

                if os.path.exists(temp_path):

                    os.remove(temp_path)

            except Exception:

                pass
# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("model_utils.py loaded successfully")
    print("=" * 60)

    print(
        "Image/Video model:",
        MODEL_PATH
    )

    print(
        "Device:",
        device
    )

    print(
        "Classes:",
        CLASS_NAMES
    )

    print(
        "Confirmed mapping:"
    )

    print(
        "fake = 0"
    )

    print(
        "real = 1"
    )

    print(
        "Maximum video frames:",
        MAX_VIDEO_FRAMES
    )

    print("=" * 60)