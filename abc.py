import streamlit as st
import numpy as np
from PIL import Image
#from tensorflow.keras.models import load_model

from io import BytesIO

# PDF
#from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered"
)


# ============================================================
# CUSTOM UI STYLE
# ============================================================

st.markdown("""
<style>

    .stApp {
        background-color: #0f1015;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 45px;
        padding-bottom: 60px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        color: white;
        font-size: 48px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 12px;
    }

    .subtitle {
        text-align: center;
        color: #dddddf;
        font-size: 20px;
        margin-bottom: 35px;
    }

    /* Section headings */
    .section-heading {
        color: white;
        font-size: 30px;
        font-weight: 750;
        margin-top: 28px;
        margin-bottom: 18px;
    }

    /* Prediction */
    .prediction-title {
        color: white;
        font-size: 23px;
        margin-bottom: 8px;
    }

    .prediction-value {
        color: white;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .confidence-value {
        color: #eeeeee;
        font-size: 22px;
        margin-bottom: 20px;
    }

    /* Probability text */
    .probability-text {
        color: white;
        font-size: 20px;
        margin-top: 12px;
        margin-bottom: 5px;
    }

    /* Buttons */
    .stButton > button {
        font-size: 20px;
        font-weight: 700;
        min-height: 55px;
        border-radius: 10px;
    }

    .stDownloadButton > button {
        font-size: 19px;
        font-weight: 700;
        min-height: 55px;
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_dr_model():
    return load_model("dr_model.h5")


model = load_dr_model()


# ============================================================
# CLASS LABELS
# KEEPING YOUR EXACT WORKING ORDER
# ============================================================

class_labels = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]


# ============================================================
# PREPROCESSING FUNCTIONS
# EXACTLY AS YOUR WORKING CODE
# ============================================================

def noise_reduction(image):
    return cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )


def contrast_enhancement(image):
    return cv2.equalizeHist(image)


def normalize_image(image):
    return cv2.normalize(
        image,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F
    )


# ============================================================
# FUNCTION FOR DISPLAYING PROCESSED IMAGES
# THIS DOES NOT CHANGE MODEL INPUT
# ============================================================

def prepare_display_image(image):

    if image.dtype != np.uint8:

        if image.max() <= 1:
            image = (
                image * 255
            ).clip(
                0,
                255
            ).astype(np.uint8)

        else:
            image = cv2.normalize(
                image,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

    return image


# ============================================================
# CONVERT IMAGE TO PDF IMAGE
# ============================================================

def array_to_pdf_image(image, width=6.5 * inch):

    display_image = prepare_display_image(image)

    if len(display_image.shape) == 2:
        pil_image = Image.fromarray(
            display_image,
            mode="L"
        )
    else:
        pil_image = Image.fromarray(
            display_image
        )

    image_buffer = BytesIO()

    pil_image.save(
        image_buffer,
        format="PNG"
    )

    image_buffer.seek(0)

    pdf_image = PDFImage(
        image_buffer,
        width=width,
        height=width * pil_image.height / pil_image.width
    )

    return pdf_image


# ============================================================
# CREATE PDF REPORT
# ============================================================

def create_pdf_report(
    original_image,
    grayscale,
    resized,
    noise_reduced,
    contrast,
    normalized,
    filename,
    image_width,
    image_height,
    image_format,
    file_size_kb,
    prediction,
    predicted_category,
    confidence
):

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )


    # ========================================================
    # STYLES
    # ========================================================

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=25,
        leading=31,
        alignment=TA_CENTER,
        spaceAfter=15,
        textColor=colors.HexColor("#222222")
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=13,
        leading=18,
        alignment=TA_CENTER,
        spaceAfter=25,
        textColor=colors.HexColor("#555555")
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=19,
        leading=24,
        spaceBefore=15,
        spaceAfter=12,
        textColor=colors.HexColor("#222222")
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#333333")
    )

    result_style = ParagraphStyle(
        "Result",
        parent=styles["Normal"],
        fontSize=21,
        leading=27,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#222222")
    )


    story = []


    # ========================================================
    # REPORT TITLE
    # ========================================================

    story.append(
        Paragraph(
            "Detection and Classification<br/>"
            "of Diabetic Retinopathy",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Diabetic Retinopathy Image Classification Report",
            subtitle_style
        )
    )

    story.append(
        Spacer(1, 10)
    )


    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "1. Image Information",
            heading_style
        )
    )

    image_info = [
        ["Information", "Details"],
        ["Image Name", filename],
        [
            "Image Dimensions",
            f"{image_width} × {image_height} pixels"
        ],
        ["Image Format", image_format],
        ["File Size", f"{file_size_kb:.2f} KB"],
    ]

    info_table = Table(
        image_info,
        colWidths=[2.2 * inch, 4.2 * inch]
    )

    info_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#eeeeee")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.black
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (0, 1),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.6,
                colors.HexColor("#bbbbbb")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(info_table)

    story.append(
        Spacer(1, 20)
    )


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    story.append(
        Paragraph(
            "2. Original Retinal Image",
            heading_style
        )
    )

    original_buffer = BytesIO()

    original_image.save(
        original_buffer,
        format="PNG"
    )

    original_buffer.seek(0)

    original_pdf_image = PDFImage(
        original_buffer,
        width=6.4 * inch,
        height=(
            6.4 * inch *
            original_image.height /
            original_image.width
        )
    )

    story.append(
        original_pdf_image
    )

    story.append(
        Spacer(1, 20)
    )


    # ========================================================
    # PREPROCESSING
    # ========================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "3. Image Preprocessing",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "The uploaded retinal image is processed through "
            "the same preprocessing pipeline used before "
            "model prediction.",
            normal_style
        )
    )

    story.append(
        Spacer(1, 15)
    )


    # --------------------------------------------------------
    # 1 GRAYSCALE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "<b>Step 1 — Grayscale Conversion</b>",
            normal_style
        )
    )

    story.append(
        Spacer(1, 5)
    )

    story.append(
        array_to_pdf_image(
            grayscale,
            width=6.2 * inch
        )
    )

    story.append(
        Spacer(1, 15)
    )


    # --------------------------------------------------------
    # 2 RESIZE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "<b>Step 2 — Resizing to 64 × 64</b>",
            normal_style
        )
    )

    story.append(
        Spacer(1, 5)
    )

    story.append(
        array_to_pdf_image(
            resized,
            width=4.5 * inch
        )
    )

    story.append(
        Spacer(1, 15)
    )


    # --------------------------------------------------------
    # 3 NOISE REDUCTION
    # --------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "<b>Step 3 — Noise Reduction</b>",
            normal_style
        )
    )

    story.append(
        Spacer(1, 5)
    )

    story.append(
        array_to_pdf_image(
            noise_reduced,
            width=4.5 * inch
        )
    )

    story.append(
        Spacer(1, 15)
    )


    # --------------------------------------------------------
    # 4 CONTRAST ENHANCEMENT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "<b>Step 4 — Contrast Enhancement</b>",
            normal_style
        )
    )

    story.append(
        Spacer(1, 5)
    )

    story.append(
        array_to_pdf_image(
            contrast,
            width=4.5 * inch
        )
    )

    story.append(
        Spacer(1, 15)
    )


    # --------------------------------------------------------
    # 5 NORMALIZATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "<b>Step 5 — Normalization</b>",
            normal_style
        )
    )

    story.append(
        Spacer(1, 5)
    )

    story.append(
        array_to_pdf_image(
            normalized,
            width=4.5 * inch
        )
    )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "4. Prediction Result",
            heading_style
        )
    )

    prediction_table = Table(
        [
            ["Prediction", predicted_category.replace("_", " ")],
            ["Confidence", f"{confidence:.2f}%"]
        ],
        colWidths=[2.5 * inch, 3.9 * inch]
    )

    prediction_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#eeeeee")
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (1, 0),
                (1, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                13
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#bbbbbb")
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                12
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                12
            )
        ])
    )

    story.append(
        prediction_table
    )

    story.append(
        Spacer(1, 30)
    )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    story.append(
        Paragraph(
            "5. Class Probabilities",
            heading_style
        )
    )


    probability_rows = [
        ["Class", "Probability", "Probability Bar"]
    ]


    for i, label in enumerate(class_labels):

        probability = (
            prediction[0][i] * 100
        )

        # Width of visual bar
        bar_width = max(
            0.05,
            min(
                2.7,
                2.7 * probability / 100
            )
        )

        bar = Table(
            [[""]],
            colWidths=[bar_width],
            rowHeights=[0.18 * inch]
        )

        bar.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#555555")
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#444444")
                )
            ])
        )

        probability_rows.append([
            label.replace("_", " "),
            f"{probability:.2f}%",
            bar
        ])


    probability_table = Table(
        probability_rows,
        colWidths=[
            1.8 * inch,
            1.2 * inch,
            3.0 * inch
        ]
    )

    probability_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#eeeeee")
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (0, 1),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#bbbbbb")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(
        probability_table
    )

    story.append(
        Spacer(1, 25)
    )


    # ========================================================
    # PREPROCESSING PIPELINE
    # ========================================================

    story.append(
        Paragraph(
            "6. Preprocessing Pipeline",
            heading_style
        )
    )

    pipeline_text = """
    Grayscale Conversion<br/>
    ↓<br/>
    Resize to 64 × 64 pixels<br/>
    ↓<br/>
    Gaussian Blur for Noise Reduction<br/>
    ↓<br/>
    Histogram Equalization for Contrast Enhancement<br/>
    ↓<br/>
    Min-Max Normalization<br/>
    ↓<br/>
    CNN Model Prediction
    """

    story.append(
        Paragraph(
            pipeline_text,
            normal_style
        )
    )


    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div class="main-title">
        👁️ Detection and Classification<br>
        of Diabetic Retinopathy
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Upload a retinal fundus image to detect the stage
        of diabetic retinopathy.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a retinal image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# AFTER IMAGE UPLOAD
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    original_array = np.array(
        original_image
    )


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Uploaded Image</div>',
        unsafe_allow_html=True
    )

    st.image(
        original_image,
        width=750
    )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button(
        "🔍 Predict"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        # ----------------------------------------------------
        # EXACT SAME PREPROCESSING AS YOUR WORKING CODE
        # ----------------------------------------------------

        image = np.array(
            original_image.convert("L")
        )

        # Resize
        resized = cv2.resize(
            image,
            (64, 64)
        )

        # Noise reduction
        noise_reduced = noise_reduction(
            resized
        )

        # Contrast enhancement
        contrast = contrast_enhancement(
            noise_reduced
        )

        # Normalization
        normalized = normalize_image(
            contrast
        )

        # Add channel dimension
        image_input = np.expand_dims(
            normalized,
            axis=-1
        )

        # Add batch dimension
        image_input = np.expand_dims(
            image_input,
            axis=0
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing retinal image..."
        ):

            prediction = model.predict(
                image_input,
                verbose=0
            )


        # ----------------------------------------------------
        # PREDICTED CLASS
        # ----------------------------------------------------

        predicted_index = np.argmax(
            prediction[0]
        )

        predicted_category = class_labels[
            predicted_index
        ]

        confidence = (
            prediction[0][predicted_index] * 100
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.markdown(
            '<div class="section-heading">'
            '🩺 Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="prediction-title">'
            'Predicted Category'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction-value">'
            f'{predicted_category.replace("_", " ")}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence-value">'
            f'<b>Confidence:</b> {confidence:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.markdown(
            '<div class="section-heading">'
            '📊 Class Probabilities'
            '</div>',
            unsafe_allow_html=True
        )


        for i, label in enumerate(
            class_labels
        ):

            probability = (
                prediction[0][i] * 100
            )

            st.markdown(
                f'<div class="probability-text">'
                f'<b>{label.replace("_", " ")}:</b> '
                f'{probability:.2f}%'
                f'</div>',
                unsafe_allow_html=True
            )

            st.progress(
                float(prediction[0][i])
            )


        # ====================================================
        # PDF REPORT
        # ====================================================

        image_width, image_height = (
            original_image.size
        )

        image_format = (
            uploaded_file.name
            .split(".")[-1]
            .upper()
        )

        file_size_kb = (
            uploaded_file.size / 1024
        )


        pdf_data = create_pdf_report(
            original_image=original_image,
            grayscale=image,
            resized=resized,
            noise_reduced=noise_reduced,
            contrast=contrast,
            normalized=normalized,
            filename=uploaded_file.name,
            image_width=image_width,
            image_height=image_height,
            image_format=image_format,
            file_size_kb=file_size_kb,
            prediction=prediction,
            predicted_category=predicted_category,
            confidence=confidence
        )


        # ====================================================
        # DOWNLOAD BUTTON
        # ====================================================

        st.markdown("<br>", unsafe_allow_html=True)

        st.download_button(
            label="📄 Download Full PDF Report",
            data=pdf_data,
            file_name="Diabetic_Retinopathy_Report.pdf",
            mime="application/pdf"
        )
