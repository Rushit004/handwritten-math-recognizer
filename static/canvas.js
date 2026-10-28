// =========================================================
// Handwritten Math Symbol Recognizer - Canvas & App Logic
// Clean, student-friendly JavaScript
// =========================================================

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const canvas = document.getElementById("drawCanvas");
  const ctx = canvas.getContext("2d");
  const clearBtn = document.getElementById("clearBtn");
  const predictBtn = document.getElementById("predictBtn");
  const strokeButtons = document.querySelectorAll(".stroke-btn");
  const statusMessage = document.getElementById("statusMessage");

  // Output Elements
  const symbolDisplay = document.getElementById("symbolDisplay");
  const symbolName = document.getElementById("symbolName");
  const categoryBadge = document.getElementById("categoryBadge");
  const confidenceText = document.getElementById("confidenceText");
  const confidenceFill = document.getElementById("confidenceFill");
  const latexCode = document.getElementById("latexCode");
  const copyLatexBtn = document.getElementById("copyLatexBtn");
  const candidatesList = document.getElementById("candidatesList");
  const referenceToggle = document.getElementById("referenceToggle");
  const referenceContent = document.getElementById("referenceContent");

  // Drawing State Variables
  let isDrawing = false;
  let hasDrawn = false;
  let strokeWidth = 16; // Default medium thickness

  // -------------------------------------------------------
  // Canvas Initialization (White Background)
  // -------------------------------------------------------
  function initCanvas() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    hasDrawn = false;
  }

  initCanvas();

  // -------------------------------------------------------
  // Coordinate Calculation Helper
  // -------------------------------------------------------
  function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches && e.touches.length > 0) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY,
      };
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  // -------------------------------------------------------
  // Drawing Event Handlers
  // -------------------------------------------------------
  function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    hasDrawn = true;
    canvas.classList.add("active-draw");
    setStatus("", "");

    const pos = getCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const pos = getCoordinates(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  }

  function stopDrawing(e) {
    if (!isDrawing) return;
    e.preventDefault();
    isDrawing = false;
    ctx.closePath();
    canvas.classList.remove("active-draw");
  }

  // Mouse Listeners
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mouseleave", stopDrawing);

  // Touch Listeners (Mobile & Tablet Support)
  canvas.addEventListener("touchstart", startDrawing, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });
  canvas.addEventListener("touchend", stopDrawing, { passive: false });
  canvas.addEventListener("touchcancel", stopDrawing, { passive: false });

  // -------------------------------------------------------
  // Stroke Width Selector
  // -------------------------------------------------------
  strokeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      strokeButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      strokeWidth = parseInt(btn.getAttribute("data-size"), 10);
      ctx.lineWidth = strokeWidth;
    });
  });

  // -------------------------------------------------------
  // Reset / Clear Canvas
  // -------------------------------------------------------
  function clearCanvas() {
    initCanvas();
    setStatus("", "");

    // Reset UI displays to default state
    symbolDisplay.textContent = "?";
    symbolName.textContent = "Draw a symbol to begin";
    categoryBadge.textContent = "Awaiting Input";
    categoryBadge.classList.remove("active");
    confidenceText.textContent = "0.0%";
    confidenceFill.style.width = "0%";
    latexCode.textContent = "-";
    candidatesList.innerHTML = '<li class="candidate-empty">Top 3 model candidates will appear here.</li>';
  }

  clearBtn.addEventListener("click", clearCanvas);

  // -------------------------------------------------------
  // Predict Drawing via Flask API
  // -------------------------------------------------------
  async function predictDrawing() {
    if (!hasDrawn) {
      setStatus("Please draw a symbol before predicting!", "error");
      return;
    }

    const dataUrl = canvas.toDataURL("image/png");

    predictBtn.disabled = true;
    predictBtn.textContent = "⏳ Analyzing...";
    setStatus("Processing symbol with CNN...", "info");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: dataUrl }),
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || "Prediction request failed");
      }

      // Update UI with primary prediction
      const pred = result.prediction;
      symbolDisplay.textContent = pred.symbol;
      symbolName.textContent = pred.name;
      categoryBadge.textContent = pred.category;
      categoryBadge.classList.add("active");

      confidenceText.textContent = `${pred.confidence.toFixed(1)}%`;
      confidenceFill.style.width = `${pred.confidence}%`;
      latexCode.textContent = pred.latex;

      // Update alternative candidate list
      renderCandidates(result.top3);
      setStatus("Prediction complete!", "info");
    } catch (err) {
      console.error(err);
      setStatus(err.message, "error");
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "✨ Predict Symbol";
    }
  }

  predictBtn.addEventListener("click", predictDrawing);

  // -------------------------------------------------------
  // Render Top-3 Candidates
  // -------------------------------------------------------
  function renderCandidates(candidates) {
    if (!candidates || candidates.length === 0) return;

    candidatesList.innerHTML = "";
    candidates.forEach((item) => {
      const li = document.createElement("li");
      li.className = "candidate-item";
      li.innerHTML = `
        <div class="left">
          <span class="rank">#${item.rank}</span>
          <span class="symbol">${item.symbol}</span>
          <span class="name">${item.name}</span>
        </div>
        <span class="conf">${item.confidence.toFixed(1)}%</span>
      `;
      candidatesList.appendChild(li);
    });
  }

  // -------------------------------------------------------
  // Copy LaTeX Code
  // -------------------------------------------------------
  copyLatexBtn.addEventListener("click", () => {
    const text = latexCode.textContent;
    if (text === "-") return;

    navigator.clipboard.writeText(text).then(() => {
      const originalText = copyLatexBtn.textContent;
      copyLatexBtn.textContent = "✅ Copied!";
      setTimeout(() => {
        copyLatexBtn.textContent = originalText;
      }, 1500);
    });
  });

  // -------------------------------------------------------
  // Status Message Helper
  // -------------------------------------------------------
  function setStatus(msg, type) {
    statusMessage.textContent = msg;
    statusMessage.className = `status-message ${type}`;
  }

  // -------------------------------------------------------
  // Collapsible Supported Symbols Section
  // -------------------------------------------------------
  if (referenceToggle && referenceContent) {
    referenceToggle.addEventListener("click", () => {
      const isHidden = referenceContent.style.display === "none";
      referenceContent.style.display = isHidden ? "grid" : "none";
      const icon = referenceToggle.querySelector(".toggle-icon");
      if (icon) {
        icon.textContent = isHidden ? "▼" : "▶";
      }
    });
  }

  // -------------------------------------------------------
  // Keyboard Shortcuts
  // -------------------------------------------------------
  document.addEventListener("keydown", (e) => {
    // Prevent interfering if typing inside an input
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    if (e.key === "Enter") {
      e.preventDefault();
      predictDrawing();
    } else if (e.key === "Escape" || e.key === "c" || e.key === "C") {
      clearCanvas();
    }
  });
});
