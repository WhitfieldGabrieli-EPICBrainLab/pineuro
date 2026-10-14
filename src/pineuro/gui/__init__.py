"""Flet-based graphical interface for pineuro neurofeedback research.

This package provides a comprehensive GUI system for managing and running
real-time fMRI neurofeedback studies. It uses the Flet framework for
cross-platform compatibility (Windows, macOS, Linux).

Architecture
------------
The GUI follows a two-tier architecture:

1. **Study Manager** ([app.py](app.py))
   - Create, edit, duplicate, and delete study configurations
   - Browse saved studies with metadata preview
   - Quick-start sessions from study cards

2. **Study Editor** ([screens/study_editor.py](screens/study_editor.py))
   - Configure all study parameters in a tabbed interface
   - Tabs: General, Paths, Preprocessing, Mask Extraction, Feedback, Sham

3. **Participant Workflow** ([screens/participant_workflow.py](screens/participant_workflow.py))
   - Execute pipeline steps per participant
   - Monitor real-time progress with visual feedback
   - Access QC reports and logs

Key Components
--------------
PyrtnfApp : Main application class ([app.py](app.py))
    Manages screen navigation and application state.

ConfigManager : Study configuration persistence ([config_manager.py](config_manager.py))
    Handles loading, saving, and listing study configurations.

StudyConfig, StudyMetadata : Data models ([models.py](models.py))
    Dataclasses for study configuration and metadata.

Workflow Components
-------------------
Components in the [components/](components/) directory provide reusable
UI elements for the participant workflow:

- CollectionProgress : DICOM/NIfTI collection progress display
- ActivationPlot : Real-time activation timeseries chart
- FDPlot : Framewise displacement timeseries chart
- TaskMirror : PsychoPy task screen mirroring
- QCViewer : Interactive QC report browser
- MaskSelectionDialog : ICA component selection for masks
- RunSelectionDialog : Feedback run selection and configuration
- SessionSelectionDialog : Multi-session session management
- FeedbackSummary : Post-run summary statistics

Usage
-----
The GUI can be launched from the command line:

    python -m pineuro.gui.app

Or programmatically:

    from pineuro.gui.app import main
    main()

Configuration Storage
---------------------
Study configurations are stored in:
    ~/.pineuro/studies/

Each study has its own directory with:
    - config.yaml: Study parameters
    - metadata.yaml: Name, description, timestamps

Dependencies
------------
flet : Cross-platform GUI framework
pyyaml : Configuration file parsing
flet-charts : Optional chart components

Notes
-----
The GUI is designed for use by experimenters during study setup and
data collection. It does NOT replace the command-line interface for
batch processing or scripted pipelines.

See Also
--------
pineuro.preprocessing : Core preprocessing functions
pineuro.feedback : Real-time feedback estimation
"""

from pathlib import Path

# Default study storage location
STUDIES_DIR = Path.home() / ".pineuro" / "studies"

__all__ = ["STUDIES_DIR"]
