import re
import cv2
import numpy as np
from PIL import Image

try:
    import latex2mathml.converter
except ImportError:
    latex2mathml = None


def assemble_math_expression(symbol_details):
    """
    Intelligently combines recognized symbols into:
    1. LaTeX math string (e.g., '2x + 3 = 5', '\\sqrt{x} + 2')
    2. Plain-text representation (e.g., '2x + 3 = 5', 'sqrt(x) + 2')
    3. Microsoft Word MathML XML string for direct paste into Word/Office
    """
    if not symbol_details:
        return "", "", ""

    tokens = [s["latex"] for s in symbol_details]
    classes = [s["class_name"] for s in symbol_details]
    categories = [s.get("category", "") for s in symbol_details]

    i = 0
    assembled = []
    plain_tokens = []

    while i < len(tokens):
        tok = tokens[i]
        cat = categories[i]

        # Handle square root followed by operand
        if tok in (r"\sqrt{}", r"\sqrt") and (i + 1 < len(tokens)):
            next_tok = tokens[i + 1]
            if next_tok not in ("+", "-", "=", r"\times", r"\div", r"\le", r"\ge", r"\ne", "<", ">"):
                assembled.append(f"\\sqrt{{{next_tok}}}")
                plain_tokens.append(f"sqrt({next_tok})")
                i += 2
                continue
            else:
                assembled.append(r"\sqrt{\ }")
                plain_tokens.append("sqrt()")
                i += 1
                continue
        elif tok in (r"\sqrt{}", r"\sqrt"):
            assembled.append(r"\sqrt{\ }")
            plain_tokens.append("sqrt()")
            i += 1
            continue

        # Format operators with spaces
        if tok in ("+", "-", "=", r"\times", r"\div", r"\pm", r"\le", r"\ge", r"\ne", "<", ">", r"\rightarrow"):
            assembled.append(f" {tok} ")
            plain_tokens.append(f" {tok} ")
        elif cat == "digits" and i > 0 and categories[i - 1] == "digits":
            # Consecutive digits: '2' + '5' -> '25'
            assembled.append(tok)
            plain_tokens.append(tok)
        elif cat == "variables" and i > 0 and categories[i - 1] == "digits":
            # Digit followed by variable: '2' + 'x' -> '2x'
            assembled.append(tok)
            plain_tokens.append(tok)
        else:
            if assembled and not assembled[-1].endswith(" "):
                assembled.append(" ")
            assembled.append(tok)
            if plain_tokens and not plain_tokens[-1].endswith(" "):
                plain_tokens.append(" ")
            plain_tokens.append(tok)

        i += 1

    latex_expr = re.sub(r"\s+", " ", "".join(assembled)).strip()
    plain_expr = re.sub(r"\s+", " ", "".join(plain_tokens)).strip()

    # Generate MS Word compatible MathML
    word_mathml = ""
    if latex2mathml is not None and latex_expr:
        try:
            word_mathml = latex2mathml.converter.convert(latex_expr)
        except Exception:
            word_mathml = f'<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mtext>{latex_expr}</mtext></mrow></math>'

    return latex_expr, plain_expr, word_mathml


def merge_overlapping_boxes(boxes, x_overlap_thresh=0.4, max_vertical_gap_ratio=1.5):
    """
    Merges bounding boxes that belong to the same composite symbol
    (e.g., division sign dots and bar, equals sign two bars, plus-minus sign, i-dots).
    """
    if len(boxes) <= 1:
        return boxes

    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = boxes[i]
        curr_box = [x1, y1, x1 + w1, y1 + h1]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                
                # Check horizontal overlap
                overlap_x = max(0, min(curr_box[2], x2 + w2) - max(curr_box[0], x2))
                min_w = min(curr_box[2] - curr_box[0], w2)
                
                # Check vertical distance
                y_dist = max(0, max(curr_box[1], y2) - min(curr_box[3], y2 + h2))
                max_h = max(curr_box[3] - curr_box[1], h2)

                if (overlap_x / max(1, min_w) >= x_overlap_thresh) and (y_dist <= max_h * max_vertical_gap_ratio):
                    # Merge boxes
                    curr_box[0] = min(curr_box[0], x2)
                    curr_box[1] = min(curr_box[1], y2)
                    curr_box[2] = max(curr_box[2], x2 + w2)
                    curr_box[3] = max(curr_box[3], y2 + h2)
                    used[j] = True
                    changed = True

        merged.append((curr_box[0], curr_box[1], curr_box[2] - curr_box[0], curr_box[3] - curr_box[1]))

    return merged


def segment_and_recognize_expression(
    canvas_image,
    predictor,
    model_name="Lightweight 2D CNN",
    min_area=40
):
    """
    Extracts, segments, and recognizes handwritten mathematical symbols across a full canvas.
    
    Parameters
    ----------
    canvas_image : PIL.Image or np.ndarray
    predictor : SymbolPredictor instance
    model_name : str
    min_area : int
    
    Returns
    -------
    result : dict containing:
        - assembled_latex: str (formatted LaTeX)
        - plain_text: str (plain-text equation)
        - word_mathml: str (MS Word MathML XML)
        - total_symbols: int
        - symbol_details: list of dicts
        - annotated_image: PIL.Image
    """
    # 1. Convert to grayscale uint8
    if isinstance(canvas_image, Image.Image):
        # Handle transparency if RGBA
        if canvas_image.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", canvas_image.size, (255, 255, 255))
            bg.paste(canvas_image, mask=canvas_image.split()[-1])
            img_arr = np.array(bg.convert("L"))
        else:
            img_arr = np.array(canvas_image.convert("L"))
    else:
        if canvas_image.ndim == 3:
            img_arr = cv2.cvtColor(canvas_image, cv2.COLOR_RGB2GRAY)
        else:
            img_arr = canvas_image.copy()

    orig_color = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    h, w = img_arr.shape

    # 2. Check polarity: invert so strokes are bright (255) on black (0)
    corners = [img_arr[0, 0], img_arr[0, -1], img_arr[-1, 0], img_arr[-1, -1]]
    if np.mean(corners) > 127:
        inverted = 255 - img_arr
    else:
        inverted = img_arr.copy()

    # 3. Thresholding & Morphology
    blurred = cv2.GaussianBlur(inverted, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Contour detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if (bw * bh >= min_area) and (bw >= 4) and (bh >= 4):
            raw_boxes.append((x, y, bw, bh))

    if not raw_boxes:
        # Fallback to whole image if no distinct components
        raw_boxes = [(0, 0, w, h)]

    # 5. Merge composite vertical multi-stroke symbols
    merged_boxes = merge_overlapping_boxes(raw_boxes)

    # 6. Sort boxes left-to-right
    sorted_boxes = sorted(merged_boxes, key=lambda b: b[0])

    # 7. Recognize each symbol
    symbol_details = []

    annotated = orig_color.copy()

    for idx, (bx, by, bw, bh) in enumerate(sorted_boxes, 1):
        # Crop symbol with slight padding
        pad = max(2, int(min(bw, bh) * 0.15))
        cy1 = max(0, by - pad)
        cy2 = min(h, by + bh + pad)
        cx1 = max(0, bx - pad)
        cx2 = min(w, bx + bw + pad)

        symbol_crop = inverted[cy1:cy2, cx1:cx2]
        crop_pil = Image.fromarray(symbol_crop)

        # Run inference
        pred_res = predictor.predict(crop_pil, model_name=model_name, top_k=3)
        best = pred_res["best_prediction"]
        top_cands = pred_res["top_predictions"]

        # Draw bounding box and label on annotated canvas
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (37, 99, 235), 2)
        cv2.putText(
            annotated,
            f"#{idx}: {best['class_name']}",
            (bx, max(15, by - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 38, 38),
            2
        )

        symbol_details.append({
            "order": idx,
            "box": (bx, by, bw, bh),
            "class_name": best["class_name"],
            "latex": best["latex"],
            "confidence_percent": best["confidence_percent"],
            "category": best["category"],
            "top_candidates": top_cands,
            "crop_image": Image.fromarray(255 - symbol_crop)  # normal polarity for display
        })

    # Assemble formula
    assembled_latex, plain_text, word_mathml = assemble_math_expression(symbol_details)

    return {
        "assembled_latex": assembled_latex,
        "plain_text": plain_text,
        "word_mathml": word_mathml,
        "total_symbols": len(symbol_details),
        "symbol_details": symbol_details,
        "annotated_image": Image.fromarray(annotated)
    }
