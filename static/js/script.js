/**
 * ClaimGuard AI — Frontend Script
 * GSAP animations, smooth scrolling, Lucide icons,
 * async prediction workflow, and state management.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ═════════════════════════════════════════════════════════════
  // 1. Initialize Lucide Icons
  // ═════════════════════════════════════════════════════════════
  if (window.lucide) {
    lucide.createIcons();
  }

  // ═════════════════════════════════════════════════════════════
  // 2. DOM References
  // ═════════════════════════════════════════════════════════════
  const form = document.getElementById("claimPredictionForm");
  const btnPredict = document.getElementById("btnPredict");
  const btnReset = document.getElementById("btnReset");
  const btnFillApproved = document.getElementById("btnFillApproved");
  const btnFillRejected = document.getElementById("btnFillRejected");
  const navbar = document.getElementById("navbar");
  const navToggle = document.getElementById("navToggle");
  const navLinks = document.getElementById("navLinks");

  // System Status
  const systemStatusBadge = document.getElementById("systemStatusBadge");
  const systemStatusText = document.getElementById("systemStatusText");

  // State Panels
  const stateInitial = document.getElementById("stateInitial");
  const stateLoading = document.getElementById("stateLoading");
  const stateError = document.getElementById("stateError");
  const stateSuccess = document.getElementById("stateSuccess");

  // Success Elements
  const claimStatusBadge = document.getElementById("claimStatusBadge");
  const claimStatusText = document.getElementById("claimStatusText");
  const decisionIcon = document.getElementById("decisionIcon");
  const approvalScoreValue = document.getElementById("approvalScoreValue");
  const approvalProgressBar = document.getElementById("approvalProgressBar");
  const modelNameValue = document.getElementById("modelNameValue");

  // Error Elements
  const errorTitle = document.getElementById("errorTitle");
  const errorMessage = document.getElementById("errorMessage");

  // Form Field References
  const fields = {
    incident_severity: document.getElementById("incident_severity"),
    police_report_available: document.getElementById("police_report_available"),
    authorities_contacted: document.getElementById("authorities_contacted"),
    witnesses: document.getElementById("witnesses"),
    total_claim_amount: document.getElementById("total_claim_amount"),
    policy_deductable: document.getElementById("policy_deductable"),
    months_as_customer: document.getElementById("months_as_customer"),
    number_of_vehicles_involved: document.getElementById("number_of_vehicles_involved"),
    incident_hour_of_the_day: document.getElementById("incident_hour_of_the_day"),
  };


  // ═════════════════════════════════════════════════════════════
  // 3. GSAP Animations & Smooth Scrolling
  // ═════════════════════════════════════════════════════════════
  if (window.gsap && window.ScrollTrigger) {
    gsap.registerPlugin(ScrollTrigger, ScrollToPlugin);

    // Animate all `.anim-fade-up` elements on scroll
    const animElements = document.querySelectorAll(".anim-fade-up");
    animElements.forEach((el, i) => {
      gsap.to(el, {
        opacity: 1,
        y: 0,
        duration: 0.7,
        ease: "power3.out",
        scrollTrigger: {
          trigger: el,
          start: "top 88%",
          toggleActions: "play none none none",
        },
        delay: el.closest(".features-grid, .hero-content, .presets-bar")
          ? i * 0.06
          : 0,
      });
    });

    // Stagger feature cards
    gsap.utils.toArray(".feature-card").forEach((card, i) => {
      gsap.to(card, {
        opacity: 1,
        y: 0,
        duration: 0.6,
        ease: "power3.out",
        delay: i * 0.08,
        scrollTrigger: {
          trigger: card,
          start: "top 90%",
        },
      });
    });

    // Stagger workflow steps
    gsap.utils.toArray(".workflow-step, .workflow-connector").forEach((step, i) => {
      gsap.to(step, {
        opacity: 1,
        y: 0,
        duration: 0.55,
        ease: "power2.out",
        delay: i * 0.1,
        scrollTrigger: {
          trigger: step,
          start: "top 90%",
        },
      });
    });

    // Smooth scroll for anchor links using GSAP ScrollToPlugin
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
      anchor.addEventListener("click", (e) => {
        e.preventDefault();
        const target = document.querySelector(anchor.getAttribute("href"));
        if (target) {
          gsap.to(window, {
            scrollTo: { y: target, offsetY: 72 },
            duration: 0.9,
            ease: "power2.inOut",
          });
          // Close mobile menu if open
          if (navLinks) navLinks.classList.remove("open");
        }
      });
    });
  } else {
    // Fallback: just reveal everything immediately
    document.querySelectorAll(".anim-fade-up").forEach((el) => {
      el.style.opacity = "1";
      el.style.transform = "none";
    });
  }


  // ═════════════════════════════════════════════════════════════
  // 4. Navbar Scroll Effect
  // ═════════════════════════════════════════════════════════════
  let lastScroll = 0;
  window.addEventListener("scroll", () => {
    const scrollY = window.scrollY;
    if (scrollY > 10) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
    lastScroll = scrollY;
  }, { passive: true });

  // Mobile menu toggle
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("open");
    });
  }


  // ═════════════════════════════════════════════════════════════
  // 5. System Health Check
  // ═════════════════════════════════════════════════════════════
  async function checkApiHealth() {
    try {
      const response = await fetch("/api/health");
      if (response.ok) {
        const data = await response.json();
        if (data.model_loaded) {
          systemStatusBadge.classList.add("online");
          systemStatusText.textContent = `Online · ${data.model_name || "Model Loaded"}`;
          return;
        }
      }
      systemStatusText.textContent = "API Up (Model Missing)";
    } catch (err) {
      systemStatusText.textContent = "Connection Error";
      console.warn("Health check failed:", err);
    }
  }
  checkApiHealth();


  // ═════════════════════════════════════════════════════════════
  // 6. State Panel Management
  // ═════════════════════════════════════════════════════════════
  function showState(panelName) {
    const panels = {
      initial: stateInitial,
      loading: stateLoading,
      error: stateError,
      success: stateSuccess,
    };

    Object.keys(panels).forEach((name) => {
      panels[name].classList.toggle("active", name === panelName);
    });

    // Re-render Lucide icons for dynamically shown panels
    if (window.lucide) {
      setTimeout(() => lucide.createIcons(), 50);
    }
  }


  // ═════════════════════════════════════════════════════════════
  // 7. Demo Presets
  // ═════════════════════════════════════════════════════════════
  function populatePreset(preset) {
    Object.keys(preset).forEach((key) => {
      if (fields[key]) {
        fields[key].value = preset[key];
      }
    });
    showState("initial");
  }

  btnFillApproved.addEventListener("click", () => {
    populatePreset({
      incident_severity: "Minor Damage",
      police_report_available: "YES",
      authorities_contacted: "Police",
      witnesses: 2,
      total_claim_amount: 32000,
      policy_deductable: 500,
      months_as_customer: 160,
      number_of_vehicles_involved: 1,
      incident_hour_of_the_day: 11,
    });
  });

  btnFillRejected.addEventListener("click", () => {
    populatePreset({
      incident_severity: "Major Damage",
      police_report_available: "NO",
      authorities_contacted: "None Contacted",
      witnesses: 0,
      total_claim_amount: 85000,
      policy_deductable: 2000,
      months_as_customer: 12,
      number_of_vehicles_involved: 3,
      incident_hour_of_the_day: 3,
    });
  });

  btnReset.addEventListener("click", () => {
    form.reset();
    showState("initial");
  });


  // ═════════════════════════════════════════════════════════════
  // 8. Form Submission & Prediction
  // ═════════════════════════════════════════════════════════════
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Collect raw values
    const rawData = {
      incident_severity: fields.incident_severity.value,
      police_report_available: fields.police_report_available.value,
      authorities_contacted: fields.authorities_contacted.value,
      witnesses: fields.witnesses.value.trim(),
      total_claim_amount: fields.total_claim_amount.value.trim(),
      policy_deductable: fields.policy_deductable.value,
      months_as_customer: fields.months_as_customer.value.trim(),
      number_of_vehicles_involved: fields.number_of_vehicles_involved.value.trim(),
      incident_hour_of_the_day: fields.incident_hour_of_the_day.value.trim(),
    };

    // Client-side: check for empty fields
    for (const [key, val] of Object.entries(rawData)) {
      if (!val && val !== 0) {
        showState("error");
        errorTitle.textContent = "Incomplete Form";
        errorMessage.textContent = `Please provide a value for "${key.replace(/_/g, " ")}".`;
        return;
      }
    }

    // Build typed payload matching backend contract
    const payload = {
      incident_severity: rawData.incident_severity,
      police_report_available: rawData.police_report_available,
      authorities_contacted: rawData.authorities_contacted,
      witnesses: Number(rawData.witnesses),
      total_claim_amount: parseFloat(rawData.total_claim_amount),
      policy_deductable: Number(rawData.policy_deductable),
      months_as_customer: Number(rawData.months_as_customer),
      number_of_vehicles_involved: Number(rawData.number_of_vehicles_involved),
      incident_hour_of_the_day: Number(rawData.incident_hour_of_the_day),
    };

    // Client-side range check for claim amount
    if (isNaN(payload.total_claim_amount) || payload.total_claim_amount < 98 || payload.total_claim_amount > 116602) {
      showState("error");
      errorTitle.textContent = "Validation Error";
      errorMessage.textContent = "Total Claim Amount must be between $98.00 and $116,602.00.";
      return;
    }

    // Client-side check for witnesses
    if (!Number.isInteger(payload.witnesses) || payload.witnesses < 0 || payload.witnesses > 3) {
      showState("error");
      errorTitle.textContent = "Validation Error";
      errorMessage.textContent = "Witnesses must be a whole number between 0 and 3.";
      return;
    }

    // ── Set Loading State ──
    btnPredict.disabled = true;
    btnPredict.classList.add("loading");
    showState("loading");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.ok && data.status === "success") {
        // ── Render Success ──
        const isApproved = data.claim_status === "Approved";

        claimStatusText.textContent = data.claim_status;
        claimStatusBadge.className = `decision-badge ${isApproved ? "badge-approved" : "badge-rejected"}`;

        // Update icon
        if (decisionIcon) {
          decisionIcon.setAttribute("data-lucide", isApproved ? "check-circle-2" : "x-circle");
        }

        const scorePercent = (data.approval_score * 100).toFixed(1);
        approvalScoreValue.textContent = `${scorePercent}%`;

        // Animate score bar with GSAP if available
        const fillWidth = Math.min(Math.max(parseFloat(scorePercent), 0), 100);
        if (window.gsap) {
          gsap.to(approvalProgressBar, {
            width: `${fillWidth}%`,
            duration: 0.8,
            ease: "power2.out",
          });
        } else {
          approvalProgressBar.style.width = `${fillWidth}%`;
        }

        modelNameValue.textContent = data.model_name || "LogisticRegression";

        // Render Influencing Factors
        const factorsList = document.getElementById("factorsList");
        if (factorsList && Array.isArray(data.influencing_factors)) {
          factorsList.innerHTML = "";
          const topFactors = data.influencing_factors.slice(0, 5);
          topFactors.forEach((factor) => {
            const isPos = factor.direction === "positive";
            const isNeg = factor.direction === "negative";
            const item = document.createElement("div");
            item.className = `factor-item ${isPos ? "factor-positive" : (isNeg ? "factor-negative" : "factor-neutral")}`;

            const sign = factor.contribution > 0 ? "+" : "";
            const iconName = isPos ? "check" : (isNeg ? "alert-circle" : "minus");

            item.innerHTML = `
              <div class="factor-main">
                <span class="factor-badge ${isPos ? "badge-pos" : (isNeg ? "badge-neg" : "badge-neu")}">
                  <i data-lucide="${iconName}" style="width:13px;height:13px"></i>
                </span>
                <div class="factor-details">
                  <span class="factor-label">${factor.label}</span>
                  <span class="factor-expl">${factor.explanation}</span>
                </div>
              </div>
              <div class="factor-impact ${isPos ? "impact-pos" : (isNeg ? "impact-neg" : "impact-neu")}">
                ${sign}${factor.contribution.toFixed(2)}
              </div>
            `;
            factorsList.appendChild(item);
          });
          if (window.lucide) {
            lucide.createIcons();
          }
        }

        showState("success");
      } else {
        // Server-side validation error
        showState("error");
        errorTitle.textContent = data.error || "Submission Error";
        errorMessage.textContent = data.message || "An unexpected error occurred during evaluation.";
      }
    } catch (err) {
      showState("error");
      errorTitle.textContent = "Network Error";
      errorMessage.textContent = "Unable to reach the prediction server. Ensure Flask is running at /api/predict.";
      console.error("Prediction request failed:", err);
    } finally {
      btnPredict.disabled = false;
      btnPredict.classList.remove("loading");
    }
  });
});
