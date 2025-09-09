# advanced-traffic-fingerprinting-analyzer
Professional traffic fingerprinting and privacy analysis suite with ML-based attack detection and defense simulation capabilities
.
├── assets
│   ├── icons
│   └── styles
├── configs
│   └── config.yaml
├── data
│   ├── datasets
│   ├── processed
│   │   ├── features.csv
│   │   └── labels.csv
│   └── raw_pcaps
├── generate_sample_data.py
├── launch_analyzer.py
├── launch_gui.py
├── logs
│   └── project.log
├── main.py
├── notebooks
│   └── traffic_analysis.ipynb
├── quick_setup.py
├── reports
│   ├── exports
│   └── templates
│       ├── evaluation_report.html
│       └── threat_model.html
├── requirements.txt
├── results
│   ├── models
│   │   ├── baseline_RandomForest_model.joblib
│   │   ├── baseline_SVM_model.joblib
│   │   ├── baseline_XGBoost_model.joblib
│   │   ├── defended_RandomForest_model.joblib
│   │   ├── defended_SVM_model.joblib
│   │   └── defended_XGBoost_model.joblib
│   ├── plots
│   │   ├── comparison_baseline_comparison.png
│   │   ├── comparison_baseline_confusion.png
│   │   ├── comparison_defended_comparison.png
│   │   ├── comparison_defended_confusion.png
│   │   └── comparison_defense_effectiveness.png
│   └── reports
│       ├── ANALYSIS_REPORT_20250909_210700.pdf
│       ├── AUTOMATED_ANALYSIS_www_bput_ac_in_20250829_215330.html
│       ├── AUTOMATED_ANALYSIS_www_bput_ac_in_20250829_215330.json
│       ├── AUTOMATED_ANALYSIS_www_bput_ac_in_20250829_215358.html
│       ├── AUTOMATED_ANALYSIS_www_bput_ac_in_20250829_215358.json
│       ├── baseline_training_results.joblib
│       ├── defended_training_results.joblib
│       ├── ENHANCED_ANALYSIS_tat_trident_ac_in_20250829_221436.html
│       ├── ENHANCED_ANALYSIS_tat_trident_ac_in_20250829_221436.json
│       ├── ENHANCED_ANALYSIS_www_lpu_in_20250829_221141.html
│       ├── ENHANCED_ANALYSIS_www_lpu_in_20250829_221141.json
│       ├── ENHANCED_ANALYSIS_www_lpu_in_20250829_221235.html
│       ├── ENHANCED_ANALYSIS_www_lpu_in_20250829_221235.json
│       ├── ENHANCED_ANALYSIS_www_pmindia_gov_in_20250829_222811.html
│       ├── ENHANCED_ANALYSIS_www_pmindia_gov_in_20250829_222811.json
│       ├── TECHNICAL_ANALYSIS_www_bput_ac_in_20250909_210729.html
│       ├── VULN_SCAN_bput_ac_in_20250909_214431.html
│       ├── VULN_SCAN_bput_ac_in_20250909_214529.html
│       ├── VULN_SCAN_bput_ac_in_20250909_214552.html
│       └── VULN_SCAN_bput_ac_in_20250909_214856.html
└── src
    ├── __init__.py
    ├── __pycache__
    │   └── __init__.cpython-313.pyc
    ├── collection
    │   ├── __init__.py
    │   ├── capture.py
    │   └── crawler.py
    ├── defense
    │   ├── __init__.py
    │   ├── padding_defense.py
    │   └── timing_defense.py
    ├── gui
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-313.pyc
    │   │   └── advanced_traffic_analyzer.cpython-313.pyc
    │   ├── advanced_traffic_analyzer.py
    │   ├── dialogs
    │   │   └── settings_dialog.py
    │   ├── main_window.py.backup
    │   ├── test_gui.py
    │   ├── utils
    │   │   └── gui_helpers.py
    │   └── widgets
    │       ├── __init__.py
    │       ├── classification_panel.py
    │       ├── defense_panel.py
    │       ├── evaluation_panel.py
    │       └── visualization_panel.py
    ├── models
    │   ├── __init__.py
    │   ├── classifier.py
    │   ├── deep_learning
    │   │   ├── __init__.py
    │   │   ├── cnn_classifier.py
    │   │   └── rnn_classifier.py
    │   ├── open_world
    │   │   ├── __init__.py
    │   │   └── ow_detector.py
    │   └── trainer.py
    ├── preprocessing
    │   ├── __init__.py
    │   ├── feature_extractor.py
    │   └── pcap_parser.py
    ├── simulation
    │   ├── __init__.py
    │   ├── network_conditions.py
    │   └── traffic_morphing.py
    ├── technical_analyzer.py
    └── visualization
        ├── __init__.py
        ├── metrics.py
        └── plots.py
