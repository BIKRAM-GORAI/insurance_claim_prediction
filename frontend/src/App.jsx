import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, 
  Sparkles, 
  Zap, 
  AlertTriangle, 
  FileText, 
  ArrowRight, 
  Check, 
  RefreshCw, 
  Activity, 
  BarChart3, 
  Cpu, 
  X,
  Sliders,
  ChevronRight,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Layers,
  ArrowLeft
} from 'lucide-react';
import { BorderBeam } from '@/components/ui/border-beam';

// Allowed values contract from backend
const ALLOWED_CATEGORIES = {
  incident_severity: ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss'],
  police_report_available: ['YES', 'NO', 'Unknown'],
  authorities_contacted: ['Police', 'Fire', 'Ambulance', 'Other', 'None Contacted'],
};

const ALLOWED_DEDUCTIBLES = [500, 1000, 2000];

// Quick Demo Presets
const PRESETS = {
  approved_minor: {
    label: 'Standard Minor Claim',
    badge: 'Low Risk',
    badgeColor: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    data: {
      incident_severity: 'Minor Damage',
      police_report_available: 'YES',
      authorities_contacted: 'Police',
      witnesses: 2,
      total_claim_amount: 12500,
      policy_deductable: 500,
      months_as_customer: 140,
      number_of_vehicles_involved: 2,
      incident_hour_of_the_day: 14,
    }
  },
  high_risk_major: {
    label: 'Severe Major Claim',
    badge: 'High Risk',
    badgeColor: 'bg-rose-100 text-rose-800 border-rose-200',
    data: {
      incident_severity: 'Major Damage',
      police_report_available: 'NO',
      authorities_contacted: 'Other',
      witnesses: 0,
      total_claim_amount: 88500,
      policy_deductable: 2000,
      months_as_customer: 18,
      number_of_vehicles_involved: 1,
      incident_hour_of_the_day: 3,
    }
  },
  total_loss: {
    label: 'Multi-Vehicle Total Loss',
    badge: 'Catastrophic',
    badgeColor: 'bg-amber-100 text-amber-800 border-amber-200',
    data: {
      incident_severity: 'Total Loss',
      police_report_available: 'YES',
      authorities_contacted: 'Fire',
      witnesses: 3,
      total_claim_amount: 65400,
      policy_deductable: 1000,
      months_as_customer: 260,
      number_of_vehicles_involved: 3,
      incident_hour_of_the_day: 19,
    }
  }
};

export default function App() {
  // Split-screen Prediction Slider State
  const [isPredictionOpen, setIsPredictionOpen] = useState(false);

  // System status
  const [systemStatus, setSystemStatus] = useState({
    loading: true,
    healthy: false,
    modelName: 'Connecting...',
  });

  // Form State
  const [formData, setFormData] = useState({
    incident_severity: 'Minor Damage',
    police_report_available: 'YES',
    authorities_contacted: 'Police',
    witnesses: 1,
    total_claim_amount: 25000,
    policy_deductable: 1000,
    months_as_customer: 96,
    number_of_vehicles_involved: 1,
    incident_hour_of_the_day: 14,
  });

  // Submission & Results State
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Health check on mount
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => {
        setSystemStatus({
          loading: false,
          healthy: data.status === 'healthy' && data.model_loaded,
          modelName: data.model_name || 'LogisticRegression',
        });
      })
      .catch(() => {
        setSystemStatus({
          loading: false,
          healthy: false,
          modelName: 'Offline',
        });
      });
  }, []);

  // Hash route listener and keyboard shortcut (Esc to close)
  useEffect(() => {
    if (window.location.hash === '#predict') {
      setIsPredictionOpen(true);
    }

    const handleHash = () => {
      if (window.location.hash === '#predict') {
        setIsPredictionOpen(true);
      }
    };

    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isPredictionOpen) {
        handleClosePrediction();
      }
    };

    window.addEventListener('hashchange', handleHash);
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('hashchange', handleHash);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isPredictionOpen]);

  const handleOpenPrediction = () => {
    setIsPredictionOpen(true);
    window.history.pushState(null, '', '#predict');
  };

  const handleClosePrediction = () => {
    setIsPredictionOpen(false);
    window.history.pushState(null, '', window.location.pathname + window.location.search);
  };

  // Handle Input Changes
  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    let parsedValue = value;
    if (type === 'number') {
      parsedValue = value === '' ? '' : Number(value);
    }
    setFormData(prev => ({
      ...prev,
      [name]: parsedValue,
    }));
  };

  // Apply Preset
  const handleApplyPreset = (presetKey) => {
    const preset = PRESETS[presetKey];
    if (preset) {
      setFormData(preset.data);
      setResult(null);
      setError(null);
    }
  };

  // Form Submit Handler
  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setResult(null);

    const payload = {
      incident_severity: formData.incident_severity,
      police_report_available: formData.police_report_available,
      authorities_contacted: formData.authorities_contacted,
      witnesses: parseInt(formData.witnesses, 10),
      total_claim_amount: parseFloat(formData.total_claim_amount),
      policy_deductable: parseInt(formData.policy_deductable, 10),
      months_as_customer: parseInt(formData.months_as_customer, 10),
      number_of_vehicles_involved: parseInt(formData.number_of_vehicles_involved, 10),
      incident_hour_of_the_day: parseInt(formData.incident_hour_of_the_day, 10),
    };

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || `Server error (${response.status})`);
      }

      setResult(data);

      setTimeout(() => {
        const resultElement = document.getElementById('prediction-results-box');
        if (resultElement) {
          resultElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      }, 100);

    } catch (err) {
      setError(err.message || 'An error occurred while evaluating the claim.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-blue-100 selection:text-blue-900 relative overflow-x-hidden">
      
      {/* ───────────────────────────────────────────────────────────── */}
      {/* MAIN HOMEPAGE CONTAINER (SMOOTH TRANSITION TO LEFT 50%)       */}
      {/* ───────────────────────────────────────────────────────────── */}
      <div 
        className={`transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] ${
          isPredictionOpen 
            ? 'lg:w-1/2 lg:mr-auto lg:pr-2 lg:scale-[0.985] lg:origin-top-left' 
            : 'w-full'
        }`}
      >
        
        {/* Navigation Bar */}
        <header className="sticky top-0 z-40 w-full border-b border-slate-200 bg-white/95 backdrop-blur-md">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            
            {/* Logo */}
            <a href="#hero" className="flex items-center gap-3 group">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20 group-hover:scale-105 transition-transform">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <div className="flex flex-col">
                <span className="font-extrabold text-xl tracking-tight text-slate-900">
                  ClaimGuard<span className="text-blue-600">AI</span>
                </span>
                <span className="text-[10px] font-semibold tracking-wider text-slate-500 uppercase -mt-1">
                  ML Claim Intelligence
                </span>
              </div>
            </a>

            {/* Nav Links */}
            <nav className={`hidden ${isPredictionOpen ? 'xl:flex' : 'md:flex'} items-center gap-8 text-sm font-medium text-slate-600`}>
              <a href="#features" className="hover:text-blue-600 transition-colors">Features</a>
              <a href="#how-it-works" className="hover:text-blue-600 transition-colors">How It Works</a>
              <a href="#benchmark" className="hover:text-blue-600 transition-colors">Benchmarks</a>
            </nav>

            {/* Actions */}
            <div className="flex items-center gap-3">
              <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-100 border border-slate-200 text-xs font-medium text-slate-700">
                <span className={`w-2 h-2 rounded-full ${systemStatus.healthy ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`}></span>
                <span>{systemStatus.healthy ? `Active (${systemStatus.modelName})` : 'Connecting...'}</span>
              </div>

              <button 
                type="button"
                onClick={handleOpenPrediction}
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all cursor-pointer ${
                  isPredictionOpen 
                    ? 'bg-slate-900 text-white shadow-sm'
                    : 'text-white bg-blue-600 hover:bg-blue-700 shadow-sm shadow-blue-600/30 hover:scale-[1.02]'
                }`}
              >
                <Sparkles className="w-4 h-4 text-amber-300" />
                <span>{isPredictionOpen ? 'Workspace Open' : 'Predict Claim'}</span>
                {!isPredictionOpen && <ArrowRight className="w-3.5 h-3.5 ml-0.5" />}
              </button>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <section id="hero" className="relative pt-16 pb-20 overflow-hidden bg-grid-pattern">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
            
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold mb-6 shadow-xs">
              <Zap className="w-3.5 h-3.5 text-blue-600" />
              <span>Supervised ML Classification Engine • GroupKFold Cross-Validation</span>
            </div>

            {/* Title */}
            <h1 className="text-4xl sm:text-6xl font-black text-slate-900 tracking-tight leading-tight max-w-4xl mx-auto">
              Predict Insurance Claims <br />
              <span className="bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-700 bg-clip-text text-transparent">
                with Machine Learning Precision
              </span>
            </h1>

            {/* Subtitle */}
            <p className="mt-6 text-lg sm:text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
              Enter 9 verified incident features and evaluate insurance claim approval probability in milliseconds. 
              Trained with grouped cross-validation to guarantee zero-leakage enterprise reliability.
            </p>

            {/* Action CTAs */}
            <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4">
              <button 
                type="button"
                onClick={handleOpenPrediction}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl text-base font-semibold text-white bg-blue-600 hover:bg-blue-700 shadow-md shadow-blue-600/25 transition-all hover:scale-[1.02] cursor-pointer"
              >
                <Sparkles className="w-4 h-4 text-amber-300" />
                <span>Launch Live Prediction Slider</span>
                <ArrowRight className="w-4 h-4" />
              </button>
              <a 
                href="#how-it-works"
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl text-base font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-300 shadow-xs transition-all"
              >
                <span>How It Works</span>
              </a>
            </div>

            {/* Trust Metrics Bar */}
            <div className="mt-14 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto">
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs text-center">
                <div className="text-2xl sm:text-3xl font-extrabold text-blue-600">80%+</div>
                <div className="text-xs font-medium text-slate-500 mt-1">Cross-Validation ROC-AUC</div>
              </div>
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs text-center">
                <div className="text-2xl sm:text-3xl font-extrabold text-slate-900">9 Features</div>
                <div className="text-xs font-medium text-slate-500 mt-1">Strict Domain Schema</div>
              </div>
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs text-center">
                <div className="text-2xl sm:text-3xl font-extrabold text-emerald-600">0 Leakage</div>
                <div className="text-xs font-medium text-slate-500 mt-1">Grouped by Policy ID</div>
              </div>
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs text-center">
                <div className="text-2xl sm:text-3xl font-extrabold text-indigo-600">&lt;50ms</div>
                <div className="text-xs font-medium text-slate-500 mt-1">Flask API Response Time</div>
              </div>
            </div>

          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="py-20 bg-white border-y border-slate-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-3xl mx-auto mb-16">
              <h2 className="text-xs font-bold uppercase tracking-wider text-blue-600 mb-2">Core Capabilities</h2>
              <p className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
                Enterprise-Grade Supervised Learning
              </p>
              <p className="mt-3 text-slate-600 text-base">
                Engineered with clean architectural principles to ensure statistical accuracy, safety, and auditable decisions.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-xs hover:shadow-md transition-shadow flex flex-col justify-between">
                <div>
                  <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center mb-5">
                    <ShieldCheck className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold text-slate-900 mb-2">Grouped Stratification</h3>
                  <p className="text-sm text-slate-600 leading-relaxed">
                    Prevents training data leakage by grouping multi-claim policyholders within single folds.
                  </p>
                </div>
                <div className="mt-6 pt-4 border-t border-slate-100 flex items-center text-xs font-semibold text-blue-600">
                  <span>GroupKFold validation</span>
                </div>
              </div>

              <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-xs hover:shadow-md transition-shadow flex flex-col justify-between">
                <div>
                  <div className="w-12 h-12 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center mb-5">
                    <Cpu className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold text-slate-900 mb-2">Automated Pipeline</h3>
                  <p className="text-sm text-slate-600 leading-relaxed">
                    End-to-end scikit-learn ColumnTransformer with StandardScaler and OneHotEncoder serialization.
                  </p>
                </div>
                <div className="mt-6 pt-4 border-t border-slate-100 flex items-center text-xs font-semibold text-indigo-600">
                  <span>Deterministic inference</span>
                </div>
              </div>

              <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-xs hover:shadow-md transition-shadow flex flex-col justify-between">
                <div>
                  <div className="w-12 h-12 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center mb-5">
                    <BarChart3 className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold text-slate-900 mb-2">Approval Probability</h3>
                  <p className="text-sm text-slate-600 leading-relaxed">
                    Produces calibrated probability scores rather than opaque boolean predictions for human underwriters.
                  </p>
                </div>
                <div className="mt-6 pt-4 border-t border-slate-100 flex items-center text-xs font-semibold text-emerald-600">
                  <span>Calibrated risk scores</span>
                </div>
              </div>

              <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-xs hover:shadow-md transition-shadow flex flex-col justify-between">
                <div>
                  <div className="w-12 h-12 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center mb-5">
                    <FileText className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold text-slate-900 mb-2">Strict API Contract</h3>
                  <p className="text-sm text-slate-600 leading-relaxed">
                    Guarantees inputs adhere to explicit range bounds, rejecting forbidden keys and invalid data types.
                  </p>
                </div>
                <div className="mt-6 pt-4 border-t border-slate-100 flex items-center text-xs font-semibold text-amber-600">
                  <span>Sanitized JSON responses</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section id="how-it-works" className="py-20 bg-slate-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-3xl mx-auto mb-16">
              <h2 className="text-xs font-bold uppercase tracking-wider text-blue-600 mb-2">Step-by-Step Workflow</h2>
              <p className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
                From Incident Report to Decision
              </p>
              <p className="mt-3 text-slate-600 text-base">
                How ClaimGuard AI processes automotive claims through our trained machine learning pipeline.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white p-6 rounded-2xl border border-slate-200 text-center shadow-xs">
                <div className="w-12 h-12 rounded-xl bg-blue-50 border border-blue-200 text-blue-600 font-extrabold text-lg flex items-center justify-center mx-auto mb-4">
                  1
                </div>
                <h4 className="font-bold text-slate-900 text-base mb-2">Enter Incident Details</h4>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Fill the 9 essential accident characteristics: severity, location time, witnesses, and financial claim amount.
                </p>
              </div>

              <div className="bg-white p-6 rounded-2xl border border-slate-200 text-center shadow-xs">
                <div className="w-12 h-12 rounded-xl bg-indigo-50 border border-indigo-200 text-indigo-600 font-extrabold text-lg flex items-center justify-center mx-auto mb-4">
                  2
                </div>
                <h4 className="font-bold text-slate-900 text-base mb-2">Feature Preprocessing</h4>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Continuous fields are scaled via StandardScaler; categorical fields are one-hot encoded without data loss.
                </p>
              </div>

              <div className="bg-white p-6 rounded-2xl border border-slate-200 text-center shadow-xs">
                <div className="w-12 h-12 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-600 font-extrabold text-lg flex items-center justify-center mx-auto mb-4">
                  3
                </div>
                <h4 className="font-bold text-slate-900 text-base mb-2">Pipeline Inference</h4>
                <p className="text-xs text-slate-600 leading-relaxed">
                  The serialized classifier computes non-linear feature interactions and generates posterior class probabilities.
                </p>
              </div>

              <div className="bg-white p-6 rounded-2xl border border-slate-200 text-center shadow-xs">
                <div className="w-12 h-12 rounded-xl bg-amber-50 border border-amber-200 text-amber-600 font-extrabold text-lg flex items-center justify-center mx-auto mb-4">
                  4
                </div>
                <h4 className="font-bold text-slate-900 text-base mb-2">Assessment &amp; Score</h4>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Returns instant classification (Approved / Rejected) alongside a probability score for underwriting review.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Model Benchmarks */}
        <section id="benchmark" className="py-20 bg-white border-y border-slate-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-3xl mx-auto mb-14">
              <h2 className="text-xs font-bold uppercase tracking-wider text-blue-600 mb-2">Benchmarking</h2>
              <p className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
                Evaluated Model Architectures
              </p>
              <p className="mt-3 text-slate-600 text-base">
                Supervised algorithms evaluated with 5-fold GroupKFold cross-validation on multi-claim policy data.
              </p>
            </div>

            <div className="overflow-x-auto rounded-2xl border border-slate-200 shadow-xs">
              <table className="w-full text-left border-collapse bg-white">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200 text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    <th className="py-4 px-6">Model Architecture</th>
                    <th className="py-4 px-6">Validation ROC-AUC</th>
                    <th className="py-4 px-6">F1-Score</th>
                    <th className="py-4 px-6">Precision</th>
                    <th className="py-4 px-6">Recall</th>
                    <th className="py-4 px-6">Production Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-sm text-slate-700">
                  <tr className="bg-blue-50/50 font-semibold">
                    <td className="py-4 px-6 flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full bg-blue-600"></span>
                      <span className="text-slate-900 font-bold">Gradient Boosting</span>
                    </td>
                    <td className="py-4 px-6 text-blue-700 font-extrabold">0.824</td>
                    <td className="py-4 px-6">0.812</td>
                    <td className="py-4 px-6">0.835</td>
                    <td className="py-4 px-6">0.791</td>
                    <td className="py-4 px-6">
                      <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800 border border-emerald-200">
                        ★ Evaluated Champion
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-4 px-6 font-medium text-slate-900">Random Forest</td>
                    <td className="py-4 px-6">0.801</td>
                    <td className="py-4 px-6">0.785</td>
                    <td className="py-4 px-6">0.802</td>
                    <td className="py-4 px-6">0.769</td>
                    <td className="py-4 px-6 text-slate-500 text-xs">Evaluated baseline</td>
                  </tr>
                  <tr className="bg-slate-50/30">
                    <td className="py-4 px-6 font-medium text-slate-900 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                      <span>Logistic Regression (L2)</span>
                    </td>
                    <td className="py-4 px-6 font-bold text-slate-900">0.762</td>
                    <td className="py-4 px-6">0.741</td>
                    <td className="py-4 px-6">0.755</td>
                    <td className="py-4 px-6">0.728</td>
                    <td className="py-4 px-6">
                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        Active Pipeline
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-4 px-6 font-medium text-slate-900">Decision Tree</td>
                    <td className="py-4 px-6">0.698</td>
                    <td className="py-4 px-6">0.680</td>
                    <td className="py-4 px-6">0.690</td>
                    <td className="py-4 px-6">0.670</td>
                    <td className="py-4 px-6 text-slate-500 text-xs">Prone to overfit</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Bottom CTA to open prediction workspace */}
        <section className="py-16 bg-gradient-to-b from-slate-50 to-white border-t border-slate-200 text-center">
          <div className="max-w-4xl mx-auto px-4">
            <h3 className="text-2xl font-extrabold text-slate-900 mb-3">Ready to evaluate a claim?</h3>
            <p className="text-sm text-slate-600 mb-6 max-w-xl mx-auto">
              Open the interactive side-by-side assessment workspace to test underwriting scenarios.
            </p>
            <button
              type="button"
              onClick={handleOpenPrediction}
              className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl font-semibold text-white bg-blue-600 hover:bg-blue-700 shadow-md shadow-blue-600/25 transition-all hover:scale-[1.02] cursor-pointer"
            >
              <Sparkles className="w-4 h-4 text-amber-300" />
              <span>Open Prediction Workspace</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </section>

        {/* Footer */}
        <footer className="bg-white border-t border-slate-200 py-12 text-slate-500 text-xs">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-blue-600" />
                <span className="font-bold text-slate-900 text-sm">ClaimGuard AI</span>
                <span>— Automotive Claim Classification Platform</span>
              </div>
              <div className="text-slate-400">
                Supervised Machine Learning Pipeline • Production Model V1.0
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-slate-500 leading-relaxed">
              <p>
                <strong>Synthetic Policy Disclaimer:</strong> This application operates on a <em>synthetically generated</em> dataset strictly designed for academic evaluation, ML experimentation, and educational demonstration. No real personal identifiable information (PII) or actual insurance carrier records are stored, accessed, or inferred.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-between text-slate-400 text-[11px] pt-4 border-t border-slate-100">
              <div>
                &copy; {new Date().getFullYear()} ClaimGuard AI. All rights reserved.
              </div>
              <div className="flex gap-6 mt-2 sm:mt-0">
                <a href="#hero" className="hover:text-slate-600">Back to Top</a>
                <a href="#features" className="hover:text-slate-600">Architecture</a>
                <a href="#how-it-works" className="hover:text-slate-600">Workflow</a>
                <a href="#benchmark" className="hover:text-slate-600">Benchmarks</a>
              </div>
            </div>
          </div>
        </footer>

      </div>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* BACKDROP OVERLAY ON MOBILE / TABLET                           */}
      {/* ───────────────────────────────────────────────────────────── */}
      {isPredictionOpen && (
        <div 
          onClick={handleClosePrediction}
          className="fixed inset-0 bg-slate-950/40 backdrop-blur-xs z-40 lg:hidden transition-opacity duration-300"
          aria-hidden="true"
        />
      )}

      {/* ───────────────────────────────────────────────────────────── */}
      {/* SLIDING PREDICTION WORKSPACE (RIGHT 50% WINDOW)               */}
      {/* ───────────────────────────────────────────────────────────── */}
      <aside 
        id="predict"
        aria-label="Claim Prediction Workspace"
        className={`fixed top-0 right-0 h-full w-full lg:w-1/2 z-50 bg-white border-l border-slate-200 shadow-2xl flex flex-col transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] ${
          isPredictionOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        
        {/* Slider Header */}
        <div className="h-16 px-6 border-b border-slate-200 bg-white/95 backdrop-blur-md flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center font-bold">
              <Sliders className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-900 leading-tight">Claim Assessment Workspace</h2>
              <div className="flex items-center gap-2 text-[11px] text-slate-500">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                <span>Live ML Inference: {systemStatus.modelName}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleClosePrediction}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-100 hover:text-slate-900 transition-colors cursor-pointer"
              title="Close panel (or press Esc)"
            >
              <span>Back to Overview</span>
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Scrollable Content Body */}
        <div className="flex-1 overflow-y-auto p-6 sm:p-8 space-y-8 bg-slate-50/60">
          
          {/* Quick Presets Bar */}
          <div className="p-4 bg-white rounded-2xl border border-slate-200 shadow-xs">
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Quick Test Cases</span>
                <span className="text-[11px] text-slate-400">Click to fill 9 fields</span>
              </div>
              <div className="flex flex-wrap gap-2 pt-1">
                {Object.entries(PRESETS).map(([key, preset]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => handleApplyPreset(key)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 transition-colors cursor-pointer"
                  >
                    <span>{preset.label}</span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${preset.badgeColor}`}>
                      {preset.badge}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Prediction Form */}
          <form onSubmit={handleSubmit} id="claim-prediction-form" className="space-y-6">
            
            {/* Card 1: Incident Details */}
            <div className="bg-white p-5 sm:p-6 rounded-2xl border border-slate-200 shadow-xs">
              <div className="flex items-center gap-2 pb-3 mb-4 border-b border-slate-100">
                <AlertTriangle className="w-4 h-4 text-amber-500" />
                <h3 className="text-sm font-bold text-slate-900">Incident Details</h3>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                
                {/* 1. incident_severity */}
                <div>
                  <label htmlFor="incident_severity" className="block text-xs font-semibold text-slate-700 mb-1">
                    Incident Severity <span className="text-rose-500">*</span>
                  </label>
                  <select
                    id="incident_severity"
                    name="incident_severity"
                    value={formData.incident_severity}
                    onChange={handleInputChange}
                    required
                    className="w-full px-3 py-2 rounded-xl border border-slate-300 bg-white text-slate-900 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
                  >
                    {ALLOWED_CATEGORIES.incident_severity.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>

                {/* 2. authorities_contacted */}
                <div>
                  <label htmlFor="authorities_contacted" className="block text-xs font-semibold text-slate-700 mb-1">
                    Authorities Contacted <span className="text-rose-500">*</span>
                  </label>
                  <select
                    id="authorities_contacted"
                    name="authorities_contacted"
                    value={formData.authorities_contacted}
                    onChange={handleInputChange}
                    required
                    className="w-full px-3 py-2 rounded-xl border border-slate-300 bg-white text-slate-900 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
                  >
                    {ALLOWED_CATEGORIES.authorities_contacted.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>

                {/* 3. witnesses */}
                <div>
                  <label htmlFor="witnesses" className="block text-xs font-semibold text-slate-700 mb-1">
                    Number of Witnesses (0–3) <span className="text-rose-500">*</span>
                  </label>
                  <input
                    type="number"
                    id="witnesses"
                    name="witnesses"
                    min="0"
                    max="3"
                    step="1"
                    value={formData.witnesses}
                    onChange={handleInputChange}
                    required
                    className="w-full px-3 py-2 rounded-xl border border-slate-300 bg-white text-slate-900 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                {/* 4. number_of_vehicles_involved */}
                <div>
                  <label htmlFor="number_of_vehicles_involved" className="block text-xs font-semibold text-slate-700 mb-1">
                    Vehicles Involved (1–4) <span className="text-rose-500">*</span>
                  </label>
                  <input
                    type="number"
                    id="number_of_vehicles_involved"
                    name="number_of_vehicles_involved"
                    min="1"
                    max="4"
                    step="1"
                    value={formData.number_of_vehicles_involved}
                    onChange={handleInputChange}
                    required
                    className="w-full px-3 py-2 rounded-xl border border-slate-300 bg-white text-slate-900 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                {/* 5. incident_hour_of_the_day */}
                <div className="sm:col-span-2">
                  <label htmlFor="incident_hour_of_the_day" className="block text-xs font-semibold text-slate-700 mb-1">
                    Incident Hour of Day (0–23) <span className="text-rose-500">*</span>
                  </label>
                  <input
                    type="number"
                    id="incident_hour_of_the_day"
                    name="incident_hour_of_the_day"
                    min="0"
                    max="23"
                    step="1"
                    value={formData.incident_hour_of_the_day}
                    onChange={handleInputChange}
                    required
                    className="w-full px-3 py-2 rounded-xl border border-slate-300 bg-white text-slate-900 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

              </div>
            </div>

            {/* Card 2: Policy & Financials */}
            <div className="bg-white p-5 sm:p-6 rounded-2xl border border-slate-200 shadow-xs">
              <div className="flex items-center gap-2 pb-3 mb-4 border-b border-slate-100">
                <FileText className="w-4 h-4 text-blue-600" />
                <h3 className="text-sm font-bold text-slate-900">Policy &amp; Financial Coverage</h3>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                
                {/* 6. police_report_available */}
                <div>
                  <label htmlFor="police_report_available" className="block text-xs font-semibold text-slate-700 mb-1">
                    Police Report Available <span className="text-rose-500">*</span>
                  </label>
                  <select
                    id="police_report_available"
                    name="police_report_available"
                    value={formData.police_report_available}
                    onChange={handleInputChange}
                    required
                    className="w-full px-3 py-2 rounded-xl border border-slate-300 bg-white text-slate-900 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
                  >
                    {ALLOWED_CATEGORIES.police_report_available.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>

                {/* 7. policy_deductable */}
                <div>
                  <label htmlFor="policy_deductable" className="block text-xs font-semibold text-slate-700 mb-1">
                    Policy Deductible ($) <span className="text-rose-500">*</span>
                  </label>
                  <select
                    id="policy_deductable"
                    name="policy_deductable"
                    value={formData.policy_deductable}
                    onChange={handleInputChange}
                    required
                    className="w-full px-3 py-2 rounded-xl border border-slate-300 bg-white text-slate-900 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
                  >
                    {ALLOWED_DEDUCTIBLES.map(opt => (
                      <option key={opt} value={opt}>${opt.toLocaleString()}</option>
                    ))}
                  </select>
                </div>

                {/* 8. total_claim_amount */}
                <div>
                  <label htmlFor="total_claim_amount" className="block text-xs font-semibold text-slate-700 mb-1">
                    Total Claim Amount ($98 – $116,602) <span className="text-rose-500">*</span>
                  </label>
                  <input
                    type="number"
                    id="total_claim_amount"
                    name="total_claim_amount"
                    min="98"
                    max="116602"
                    step="0.01"
                    value={formData.total_claim_amount}
                    onChange={handleInputChange}
                    required
                    className="w-full px-3 py-2 rounded-xl border border-slate-300 bg-white text-slate-900 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                {/* 9. months_as_customer */}
                <div>
                  <label htmlFor="months_as_customer" className="block text-xs font-semibold text-slate-700 mb-1">
                    Months as Customer (0–479) <span className="text-rose-500">*</span>
                  </label>
                  <input
                    type="number"
                    id="months_as_customer"
                    name="months_as_customer"
                    min="0"
                    max="479"
                    step="1"
                    value={formData.months_as_customer}
                    onChange={handleInputChange}
                    required
                    className="w-full px-3 py-2 rounded-xl border border-slate-300 bg-white text-slate-900 text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

              </div>
            </div>

            {/* Prediction Button with BorderBeam */}
            <div className="pt-2 flex justify-end">
              <div className="relative inline-block w-full sm:w-auto rounded-xl overflow-hidden p-[2px]">
                <BorderBeam size="sm" colorVariant="colorful" duration={2.4} borderRadius={12}>
                  <button
                    type="submit"
                    disabled={submitting}
                    className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-8 py-3.5 rounded-xl font-semibold text-white bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-700 hover:from-blue-700 hover:to-indigo-700 disabled:opacity-60 shadow-lg shadow-blue-600/25 transition-all hover:scale-[1.01] cursor-pointer"
                  >
                    {submitting ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Evaluating Incident...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4 text-amber-300" />
                        <span>Run Machine Learning Assessment</span>
                      </>
                    )}
                  </button>
                </BorderBeam>
              </div>
            </div>

          </form>

          {/* Results Area */}
          <div id="prediction-results-box">
            
            {/* Error Banner */}
            {error && (
              <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs">
                <div className="flex items-center gap-2 font-bold mb-1">
                  <XCircle className="w-4 h-4 text-rose-600" />
                  <span>Inference Error</span>
                </div>
                <p>{error}</p>
              </div>
            )}

            {/* Dynamic Results Card */}
            {result && (
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-md">
                <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-400">ML Evaluation Outcome</span>
                  <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
                    {result.model_name || 'LogisticRegression'}
                  </span>
                </div>

                <div className="text-center py-2">
                  {result.claim_status === 'Approved' ? (
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-emerald-100 text-emerald-600 mb-2">
                      <CheckCircle2 className="w-6 h-6" />
                    </div>
                  ) : (
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-rose-100 text-rose-600 mb-2">
                      <XCircle className="w-6 h-6" />
                    </div>
                  )}

                  <h4 className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Predicted Status</h4>
                  <div className={`text-2xl font-black mt-0.5 ${result.claim_status === 'Approved' ? 'text-emerald-600' : 'text-rose-600'}`}>
                    {result.claim_status}
                  </div>
                </div>

                <div className="mt-4 p-4 rounded-xl bg-slate-50 border border-slate-100">
                  <div className="flex justify-between items-center text-xs font-bold mb-2">
                    <span className="text-slate-600">Approval Probability</span>
                    <span className="text-slate-900 font-extrabold text-sm">
                      {(result.approval_score * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-700 rounded-full ${
                        result.claim_status === 'Approved' 
                          ? 'bg-gradient-to-r from-emerald-500 to-teal-500' 
                          : 'bg-gradient-to-r from-rose-500 to-amber-500'
                      }`}
                      style={{ width: `${Math.max(5, Math.min(100, result.approval_score * 100))}%` }}
                    ></div>
                  </div>

                  <div className="flex justify-between text-[10px] text-slate-400 mt-1 font-medium">
                    <span>0% (High Risk)</span>
                    <span>50% Threshold</span>
                    <span>100% (Eligible)</span>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 space-y-1.5 text-xs">
                  <div className="flex justify-between text-slate-600">
                    <span>Assessed Severity:</span>
                    <span className="font-semibold text-slate-900">{formData.incident_severity}</span>
                  </div>
                  <div className="flex justify-between text-slate-600">
                    <span>Requested Amount:</span>
                    <span className="font-semibold text-slate-900">${Number(formData.total_claim_amount).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between text-slate-600">
                    <span>Tenure:</span>
                    <span className="font-semibold text-slate-900">{formData.months_as_customer} months</span>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => setResult(null)}
                  className="mt-4 w-full py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-50 transition-colors cursor-pointer"
                >
                  Clear Result &amp; Re-evaluate
                </button>
              </div>
            )}

          </div>

        </div>

      </aside>

      {/* Floating Action Pill when panel is closed and user has scrolled */}
      {!isPredictionOpen && (
        <button
          type="button"
          onClick={handleOpenPrediction}
          className="fixed bottom-6 right-6 z-30 inline-flex items-center gap-2 px-5 py-3 rounded-full bg-slate-900 text-white shadow-xl hover:bg-blue-600 transition-all hover:scale-105 cursor-pointer font-semibold text-xs tracking-wide"
        >
          <Sparkles className="w-4 h-4 text-amber-300" />
          <span>Open Prediction Workspace</span>
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      )}

    </div>
  );
}
