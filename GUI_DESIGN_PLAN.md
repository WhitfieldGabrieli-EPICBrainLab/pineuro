# pyrtnf Flet GUI Application - Comprehensive Design Plan

## Executive Summary

**Framework: Flet (Python-native Flutter UI)**

Build a two-tier GUI for pyrtnf neurofeedback research:
1. **Study Configuration Manager** - Create/edit/load reusable study protocols
2. **Participant Workflow Dashboard** - Execute and monitor pipeline steps per participant

**Key Design Decisions:**
- Flet chosen for Python-native development, modern UI, and sufficient real-time performance
- Studies stored in `~/.pyrtnf/studies/` as YAML files with metadata
- Participant workflow uses manual step-by-step control with status dashboard
- Real-time monitoring shows live activation plots, progress indicators, and streaming logs
- PsychoPy runs fullscreen on monitor 2, GUI monitoring on monitor 1
- QC reports auto-generate and embed in GUI via HTML viewer

---

## Requirements Summary (From User Input)

### Study Configuration
- **Persistence**: Save/load study configs as YAML files in studies library (`~/.pyrtnf/studies/`)
- **Reusability**: Load existing study, modify parameters, save as new study or update existing
- **Metadata**: Each study has name, description, creation/modification dates
- **Parameters**: Full control over all SessionConfig fields (Scanner, Network/Mask, Task, Paths)
- **Validation**: Real-time validation of parameters before saving

### Participant Workflow
- **Entry**: Study selection screen first, then enter participant ID to start workflow
- **Structure**: Dashboard showing all pipeline steps with status indicators (pending/running/complete)
- **Control**: Manual button-per-step control (no auto-progression)
  - Setup: Collect 2vol, Collect Rest Runs, Preprocess, Extract Masks, Coregister
  - Feedback: Transfer Run 1, Feedback Runs 1-N, Transfer Run 2
- **Execution**: Single step at a time (GUI blocks during execution)
- **Resume**: Show completion status, allow re-running any step with overwrite confirmation

### Monitoring & QC
- **Real-time plots**: Dual-network activation timeseries (full run history)
- **Progress**: Volume counters, progress bars, hit counters, scale factor display
- **Logs**: Always-visible console panel with streaming pipeline logs
- **QC Reports**: Auto-generate after each step, embed HTML in GUI viewer
- **Completion**: Show summary dialog with hits/scale factor and "Run Next Task" prompt

### Technical Requirements
- **Simulator Toggle**: Study config includes "Use Simulator" checkbox
- **PsychoPy Display**: Fullscreen on monitor 2 (participant-facing), launched as subprocess
- **Thread Safety**: Integration with NeurofeedbackCoordinator background thread
- **Error Handling**: Error dialogs with log excerpts on step failure
- **BIDS Export**: Export session data to BIDS derivatives format

---

## Application Architecture

### Overall Structure

```
pyrtnf-gui (Flet app)
├── Study Manager Screen
│   ├── Study list (cards with name, description, last modified)
│   ├── Create New Study button
│   ├── Load Study → goes to Study Editor
│   └── Delete Study (with confirmation)
├── Study Editor Screen
│   ├── Study metadata (name, description)
│   ├── Tabbed parameter sections
│   │   ├── Scanner Parameters
│   │   ├── Network/Mask Parameters
│   │   ├── Task Parameters
│   │   └── File Paths
│   ├── Validation status indicator
│   ├── Save / Save As / Cancel buttons
│   └── "Start Participant Session" button
└── Participant Workflow Screen
    ├── Header: Study name, Participant ID, Use Simulator indicator
    ├── Step Dashboard (main area)
    │   ├── Setup Phase Card
    │   │   ├── Collect Initial 2vol [button] [status]
    │   │   ├── Collect Rest Run 1 [button] [status]
    │   │   ├── Collect Rest Run 2 [button] [status]
    │   │   ├── Preprocess Rest Data [button] [status]
    │   │   ├── Extract Network Masks [button] [status]
    │   │   └── Coregister to 2vol [button] [status]
    │   ├── Feedback Session Card
    │   │   ├── Collect Real-time 2vol [button] [status]
    │   │   ├── Transfer Run 1 [button] [status]
    │   │   ├── Feedback Run 1 [button] [status]
    │   │   ├── Feedback Run N [button] [status]
    │   │   ├── Transfer Run 2 [button] [status]
    │   │   └── "Run Next Task" (smart button) [status]
    │   └── QC Reports Card
    │       ├── Preprocessing QC [button]
    │       ├── Mask Extraction QC [button]
    │       ├── 2vol Coregistration QC [button]
    │       └── Per-run Feedback QC [buttons]
    ├── Parameter Preview Panel (right sidebar, shows when step button clicked)
    │   ├── Step name and description
    │   ├── Key parameters for this step
    │   ├── Expected outputs
    │   ├── Overwrite warning if outputs exist
    │   └── "Start" button
    ├── Monitoring Panel (replaces preview during execution)
    │   ├── Real-time activation plot (dual-network timeseries)
    │   ├── Progress bar + volume counter
    │   ├── Hit counters, scale factor display
    │   └── "Stop" button (emergency abort)
    └── Log Console (bottom panel, always visible)
        ├── Auto-scrolling log viewer
        ├── Log level filter (INFO/WARNING/ERROR)
        └── "Export Logs" button
```

### Data Flow

1. **Study Creation/Loading**
   - User selects or creates study → Study Editor
   - Editor populates form from YAML or defaults (SessionConfig dataclass)
   - Validation runs on field changes
   - Save button writes YAML to `~/.pyrtnf/studies/{study_name}.yaml`

2. **Participant Session Start**
   - Load study config from YAML
   - Prompt for participant ID
   - Create PathConfig with participant-specific directories
   - Initialize or load SessionState from `{session_dir}/session_state.json`
   - Detect completed steps by checking output file existence
   - Enable/disable step buttons based on dependencies and completion

3. **Step Execution Flow**
   ```
   User clicks step button
   → Parameter preview panel appears
   → User reviews and clicks "Start"
   → GUI blocks other steps, shows monitoring panel
   → Background thread runs pipeline function (e.g., run_preprocessing)
   → Flet page.update() called via thread-safe callback
   → Monitoring panel updates plots/progress in real-time
   → On completion: auto-generate QC, show summary dialog
   → Re-enable GUI, update step status
   ```

4. **Real-time Monitoring (for feedback runs)**
   - NeurofeedbackCoordinator runs in background thread
   - PsychoPy BallTask runs in separate subprocess (fullscreen monitor 2)
   - Coordinator writes to `feedback.log.tsv` after each volume
   - GUI polls TSV file at 1 Hz (Flet timer)
   - Parses latest data, appends to activation plot
   - Updates progress bar, hit counters, scale factor displays

5. **QC Report Generation**
   - After step completes, call QC function (e.g., `generate_preprocessing_qc`)
   - QC generates HTML report in `{output_dir}/qc/`
   - GUI detects new HTML file
   - "View QC" button becomes enabled
   - Click button → open Flet WebView or markdown viewer with embedded HTML

---

## Detailed Screen Specifications

### Screen 1: Study Manager

**Purpose**: Browse, create, and manage study configurations

**Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│ pyrtnf Study Manager                               [+ New]  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │ Dual Network Study  │  │ DMN-Only Feedback   │          │
│  │ ─────────────────── │  │ ─────────────────── │          │
│  │ 17 Yeo networks     │  │ Single network      │          │
│  │ ContA + DefaultA    │  │ DefaultA only       │          │
│  │                     │  │                     │          │
│  │ Modified: 2026-01-15│  │ Modified: 2026-01-10│          │
│  │                     │  │                     │          │
│  │ [Edit] [Duplicate]  │  │ [Edit] [Duplicate]  │          │
│  │ [Delete]            │  │ [Delete]            │          │
│  └─────────────────────┘  └─────────────────────┘          │
│                                                               │
│  (More study cards...)                                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Components**:
- **Header Bar**: App title + "New Study" button
- **Study Cards** (GridView):
  - Study name (large text, editable on click)
  - Description (2-line snippet, full text on hover)
  - Key parameters (network names, TR, n_runs)
  - Last modified timestamp
  - Action buttons: Edit, Duplicate, Delete
- **Empty State**: "No studies found. Create your first study!" + large Create button

**Interactions**:
- Click "New Study" → Study Editor (blank form with defaults)
- Click "Edit" → Study Editor (populated with study config)
- Click "Duplicate" → Create copy with "(Copy)" suffix, open in Editor
- Click "Delete" → Confirmation dialog, then delete YAML file
- Double-click card → Direct to Participant Workflow (prompts for participant ID)

**Data Storage**:
- Study configs: `~/.pyrtnf/studies/{study_name}.yaml`
- Metadata stored in YAML frontmatter:
  ```yaml
  _metadata:
    name: "Dual Network Study"
    description: "ContA vs DefaultA neurofeedback protocol"
    created: "2026-01-15T10:30:00"
    modified: "2026-01-15T14:22:00"
  paths:
    participant_id: ""  # Set per-participant
    base_dir: "/data/pyrtnf"
  scan:
    tr: 1.2
    ...
  ```

---

### Screen 2: Study Editor

**Purpose**: Create or modify study configuration parameters

**Layout** (Tabs for parameter sections):
```
┌─────────────────────────────────────────────────────────────┐
│ Study Editor: Dual Network Study                            │
│ [Scanner] [Networks] [Task] [Paths] [Advanced]              │
├─────────────────────────────────────────────────────────────┤
│ SCANNER PARAMETERS                                           │
│                                                               │
│ Study Name:        [Dual Network Study_____________]         │
│ Description:       [ContA vs DefaultA feedback_______________│
│                     protocol for executive control_________] │
│                                                               │
│ TR (seconds):      [1.2____] ✓                              │
│ Volumes per run:   [150____] ✓                              │
│ Baseline volumes:  [25_____] ✓                              │
│                                                               │
│ Rest Runs:         [2______] ✓                              │
│ Volumes per rest:  [200, 200___________________] ✓          │
│                                                               │
│ 2vol Volumes:      [2______] ✓                              │
│ Feedback Runs:     [2______] ✓                              │
│                                                               │
│ Use Simulator:     [✓] Enable DICOM simulator               │
│   Simulator Speed: [10.0___]x (collection)                  │
│                    [1.0____]x (feedback)                     │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                     [Cancel] [Save] [Save As] [Start Session]│
└─────────────────────────────────────────────────────────────┘
```

**Tab 1: Scanner Parameters** (shown above)
- Study name (TextField)
- Description (TextField, multiline)
- TR, n_volumes, n_baseline (NumberField with validation)
- Rest runs configuration
- 2vol configuration
- Simulator toggle + speed multipliers

**Tab 2: Networks Parameters**
```
NETWORK/MASK PARAMETERS

Yeo Atlas:          ( ) 7 Networks  (•) 17 Networks

Target Networks (select 2+):
  [✓] DefaultA       Default Network (anterior)
  [✓] ContA          Control Network (anterior)
  [ ] DefaultB       Default Network (posterior)
  [ ] ContB          Control Network (posterior)
  [ ] ContC          Control Network (cingulate)
  [ ] LimbicA        Limbic (temporal)
  [ ] LimbicB        Limbic (orbitofrontal)
  [ ] SalVentAttnA   Salience/Ventral Attention (anterior)
  ... (all 17 networks)

Mask Names (for display):
  Network 1: [DefaultA___________]
  Network 2: [ContA______________]

Spatial Smoothing:  [5.0___] mm FWHM
Highpass Filter:    [0.01__] Hz
Lowpass Filter:     [______(optional)]
```

**Tab 3: Task Parameters**
```
TASK PARAMETERS (BallTask)

Initial Scale Factor:     [10.0___]
Fullscreen:               [✓]

Adaptive Difficulty:
  Min Hits (too slow):    [3_____] → increase 25%
  Max Hits (too fast):    [5_____] → decrease 25%
  Adjust Factor:          [0.25__] (25%)

Keyboard Controls:
  Trigger Keys:           [t, +, 5, equal_________]
  Left Button:            [1_]
  Right Button:           [2_]
  Enter Button:           [0_]
```

**Tab 4: File Paths**
```
FILE PATHS

Base Data Directory:
  [/data/pyrtnf________________________] [Browse]

DICOM Sources (for simulator mode):

Rest Run 1:  [/data/source/rest_run1______] [Browse]
Rest Run 2:  [/data/source/rest_run2______] [Browse]

2vol (initial):   [/data/source/2vol_initial_] [Browse]
2vol (realtime):  [/data/source/2vol_realtime] [Browse]

Feedback Runs:    [/data/source/feedback_____] [Browse]

Output subdirectories will be created automatically:
  • {base_dir}/preprocessing
  • {base_dir}/mask_extraction
  • {base_dir}/feedback
  • {base_dir}/collection
```

**Validation**:
- Real-time validation on field change
- Visual indicators: ✓ (valid), ⚠ (warning), ✗ (error)
- Error messages below invalid fields
- Save button disabled until all required fields valid
- Warnings don't block save (e.g., "DICOM path doesn't exist - OK if using real scanner")

**Actions**:
- **Cancel**: Discard changes, return to Study Manager
- **Save**: Overwrite current study YAML (disabled if invalid)
- **Save As**: Prompt for new study name, create new YAML
- **Start Session**: Save (if dirty) → Participant Workflow screen

---

### Screen 3: Participant Workflow Dashboard

**Purpose**: Execute pipeline steps and monitor progress for a single participant

**Layout**:
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ pyrtnf: Dual Network Study | Participant: sub-01 | Simulator Mode  [← Studies] │
├─────────────────────────┬───────────────────────────────────────────────────────┤
│ SETUP PHASE             │ PARAMETER PREVIEW / MONITORING                        │
│                         │                                                       │
│ ┌─ Initial 2vol        │ (Empty until step button clicked)                     │
│ │  [Collect] ✓         │                                                       │
│ │                      │                                                       │
│ ┌─ Resting State       │                                                       │
│ │  [Collect Run 1] ✓   │                                                       │
│ │  [Collect Run 2] ✓   │                                                       │
│ │                      │                                                       │
│ ┌─ Preprocessing       │                                                       │
│ │  [Run Preprocessing] │                                                       │
│ │  ⏸ Pending           │                                                       │
│ │                      │                                                       │
│ ┌─ Mask Extraction     │                                                       │
│ │  [Extract Masks] ⏸   │                                                       │
│ │                      │                                                       │
│ ┌─ Coregistration      │                                                       │
│ │  [Coregister] ⏸      │                                                       │
│ │                      │                                                       │
│ FEEDBACK SESSION        │                                                       │
│                         │                                                       │
│ ┌─ Real-time 2vol      │                                                       │
│ │  [Collect] ⏸         │                                                       │
│ │                      │                                                       │
│ ┌─ Feedback Runs       │                                                       │
│ │  [Transfer Run 1] ⏸  │                                                       │
│ │  [Feedback Run 1] ⏸  │                                                       │
│ │  [Feedback Run 2] ⏸  │                                                       │
│ │  [Transfer Run 2] ⏸  │                                                       │
│ │                      │                                                       │
│ │  [▶ Run Next Task]   │                                                       │
│ │                      │                                                       │
│ QUALITY CONTROL         │                                                       │
│                         │                                                       │
│ ┌─ QC Reports          │                                                       │
│ │  [Preprocessing] ⏸   │                                                       │
│ │  [Masks] ⏸           │                                                       │
│ │  [2vol Coreg] ⏸      │                                                       │
│ │  [Feedback Runs] ⏸   │                                                       │
│                         │                                                       │
├─────────────────────────┴───────────────────────────────────────────────────────┤
│ LOG CONSOLE                                                                     │
│ [INFO] 2026-01-16 14:22:10 - Session loaded: sub-01                            │
│ [INFO] 2026-01-16 14:22:10 - Detected completed: 2vol_run-01                   │
│ [INFO] 2026-01-16 14:22:10 - Detected completed: rest_run-01, rest_run-02      │
│ ...                                                         [Clear] [Export]    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Left Panel: Step Dashboard**
- **Setup Phase** (collapsible section):
  - Each step = button + status icon
  - Status icons: ⏸ Pending, ⏳ Running, ✓ Complete, ✗ Failed
  - Button disabled if dependencies not met or already running
  - Green highlight if ready to run, gray if blocked

- **Feedback Session** (collapsible section):
  - Individual run buttons (Transfer 1, Feedback 1-N, Transfer 2)
  - "Run Next Task" smart button (determines next run from SessionState)
  - Highlights next recommended run

- **QC Reports** (collapsible section):
  - Buttons to view QC HTML for each phase
  - Disabled until QC report exists
  - Badge showing number of warnings/errors in QC

**Right Panel: Parameter Preview (step button clicked)**
```
┌───────────────────────────────────────────┐
│ COLLECT REST RUN 1                        │
├───────────────────────────────────────────┤
│ Step: DICOM Collection (Resting State)    │
│                                           │
│ Parameters:                               │
│  • Task: rest_run-01                      │
│  • Volumes: 200                           │
│  • Source: /data/source/rest_run1         │
│  • Output: /data/sub-01/collection/       │
│            rest_run-01/                   │
│  • Simulator Speed: 10.0x                 │
│                                           │
│ Status: Not started                       │
│                                           │
│ Expected Duration: ~2 minutes (simulated) │
│                                           │
│ ⚠ Output directory exists. Overwrite?     │
│   [ ] Skip if sufficient DICOMs exist     │
│   [ ] Force overwrite                     │
│                                           │
│             [Cancel] [▶ Start]            │
└───────────────────────────────────────────┘
```

**Right Panel: Monitoring View (during execution)**
```
┌───────────────────────────────────────────┐
│ FEEDBACK RUN 1 - RUNNING                  │
├───────────────────────────────────────────┤
│ ┌─ Activation (normalized) ─────────────┐ │
│ │   0.5┤                              /│ │
│ │      │                          /\ / │ │
│ │   0.0┤     DefaultA    /\/\   /  \/  │ │
│ │      │                /    \ /       │ │
│ │  -0.5┤───────────────────────────────│ │
│ │   0.5┤          /\                   │ │
│ │      │    ContA   \/\                │ │
│ │   0.0┤   /            \     /\       │ │
│ │      │  /              \   /  \      │ │
│ │  -0.5┤──────────────────────────────│ │
│ │      0   20   40   60   80  100 120 │ │
│ │            Volume                    │ │
│ └──────────────────────────────────────┘ │
│                                           │
│ Progress: [████████████░░░░░░] 75/150    │
│                                           │
│ Status: Feedback Active                   │
│ Hits: ContA: 3, DefaultA: 2               │
│ Scale Factor: 10.0                        │
│ Motion: 0.12 mm (current FD)              │
│                                           │
│                      [⏹ Stop Run]         │
└───────────────────────────────────────────┘
```

**Bottom Panel: Log Console**
- Auto-scrolling text viewer (last 100 lines visible)
- Color-coded by level: INFO (white), WARNING (yellow), ERROR (red)
- Timestamp + log level + message
- Filter buttons: [All] [Info] [Warnings] [Errors]
- Actions: Clear, Export to file

**Completion Dialog** (after run finishes):
```
┌─────────────────────────────────────────┐
│ Feedback Run 1 Complete!                │
├─────────────────────────────────────────┤
│ Results:                                │
│  • Volumes Processed: 150/150           │
│  • Hits: ContA: 4, DefaultA: 3          │
│  • New Scale Factor: 8.75               │
│  • Circle Radii: 0.081, 0.073           │
│                                         │
│ QC Report: ✓ Generated                  │
│                                         │
│ Next Step: Feedback Run 2               │
│                                         │
│     [View QC] [▶ Run Next Task] [Done] │
└─────────────────────────────────────────┘
```

**Status Indicators**:
- ⏸ **Pending**: Not yet run, dependencies not met
- ⏳ **Running**: Currently executing (spinner animation)
- ✓ **Complete**: Finished successfully (green)
- ✗ **Failed**: Error occurred (red, click for details)
- ⚠ **Warning**: Completed with warnings (yellow)

---

## Framework Comparison (Flet vs Alternatives)

### Option 1: Flet (RECOMMENDED FOR THIS PROJECT)

**Strengths:**
- **Native performance**: C++ backend, minimal overhead for real-time plotting
- **Mature ecosystem**: 25+ years of development, extremely stable
- **Rich widget library**: Built-in support for plots (via pyqtgraph), file dialogs, forms, tabs
- **Thread integration**: Excellent Qt signal/slot mechanism for thread-safe GUI updates
- **Professional appearance**: Native OS look-and-feel or custom styling via QSS (CSS-like)
- **Single dependency**: `pip install PyQt6 pyqtgraph` covers most needs
- **Proven in neuroscience**: Used by MNE-Python, AFNI, FSLeyes, and many neuroimaging tools

**Integration with pyrtnf:**
- Main window runs in main thread (Qt requirement)
- NeurofeedbackCoordinator runs in background thread (existing pattern)
- Qt signals bridge threads: coordinator emits signal → GUI slot updates plot
- PsychoPy runs in separate process (launched via subprocess) to isolate display
- CSV tailing uses `QFileSystemWatcher` for efficient monitoring

**Real-time plotting:**
- **pyqtgraph**: High-performance plotting library designed for Qt
  - Handles streaming data at 60+ Hz
  - Built-in scrolling plot widgets
  - Minimal CPU overhead (OpenGL acceleration available)
  - Example: Live activation timeseries with 200-point rolling window

**Example architecture:**
```python
# Qt signal for thread-safe updates
class FeedbackSignals(QObject):
    new_activation = pyqtSignal(dict)  # {volume, masks, motion}

# In NeurofeedbackCoordinator:
self.signals = FeedbackSignals()
# When volume processed:
self.signals.new_activation.emit(data_dict)

# In GUI main window:
coordinator.signals.new_activation.connect(self.update_plots)

def update_plots(self, data):
    # Runs in main thread, safe to update widgets
    self.activation_plot.append(data['masks'])
```

**Complexity:** Medium
- Learning curve for Qt paradigm (signals/slots, layouts)
- Extensive documentation and examples
- Designer tool available for visual layout creation (Qt Designer)

---

### Option 2: Tkinter + matplotlib

**Strengths:**
- **Included in Python**: No additional dependencies
- **Simple API**: Easiest to learn for Python developers
- **Good documentation**: Many examples online

**Weaknesses:**
- **Poor real-time performance**: matplotlib backends slow for streaming data
- **Limited widgets**: Lacks modern UI components (no built-in tab widget, limited styling)
- **Dated appearance**: Looks outdated without significant custom work
- **Threading awkward**: No built-in signal/slot mechanism, manual queue management required
- **Not ideal for neuroscience apps**: Rarely used in professional neuroimaging tools

**Integration with pyrtnf:**
- Would require manual threading with `queue.Queue`
- matplotlib FuncAnimation for updates (10-30 Hz max practical)
- Less efficient for multi-plot dashboards

**Complexity:** Low (to start), Medium (for real-time features)

---

### Option 3: Web-based (Flask/FastAPI + React/Plotly Dash)

**Strengths:**
- **Modern appearance**: Professional, customizable UI
- **Remote access**: Could monitor from another machine in future
- **Rich plotting**: Plotly provides interactive visualizations
- **Cross-platform**: Browser-based, no client installation

**Weaknesses:**
- **Additional complexity**: Requires web stack (frontend + backend)
- **Not Python-native**: Frontend in JavaScript (React) or limited Python (Dash)
- **Overhead**: WebSocket latency for real-time updates
- **Deployment complexity**: Managing web server alongside PsychoPy
- **Team preference mismatch**: Requires web development knowledge

**Integration with pyrtnf:**
- Backend API wraps NeurofeedbackCoordinator
- WebSocket stream for real-time data
- Browser on localhost for GUI

**Complexity:** High (unless using Dash framework, then Medium-High)

---

### Option 4: Dear PyGui

**Strengths:**
- **GPU-accelerated**: Very fast rendering
- **Simple API**: Python-native, immediate mode paradigm
- **Modern look**: Sleek, game-like interface

**Weaknesses:**
- **Young library**: Less mature than Qt (~3 years old)
- **Limited ecosystem**: Fewer third-party widgets/examples
- **Unconventional paradigm**: Immediate mode GUI unusual for traditional apps
- **Threading unclear**: Less established patterns for background threads

**Complexity:** Medium

---

## Recommendation: PyQt6 (or PySide6)

### Why PyQt6/PySide6?

1. **Best fit for requirements:**
   - Real-time performance via pyqtgraph
   - Native Python integration
   - Thread-safe by design (Qt signals/slots)
   - Professional, polished UI
   - Same-machine deployment (no web server)

2. **Proven in neuroscience:**
   - FSLeyes (FSL's viewer): PyQt5
   - MNE-Python visualizations: PyQt5 + pyqtgraph
   - AFNI GUI components: Qt-based
   - Nilearn plotting tools compatible with Qt backends

3. **Long-term maintainability:**
   - Stable API, excellent backwards compatibility
   - Large community, extensive documentation
   - Qt Company backing (PySide6) or Riverbank (PyQt6)

4. **Development velocity:**
   - Rapid prototyping with Qt Designer
   - Rich widget library reduces custom code
   - Strong integration with numpy/scipy ecosystem

### PyQt6 vs PySide6?

Both are nearly identical (PySide6 is Qt Company's official binding):

| Feature | PyQt6 | PySide6 |
|---------|-------|---------|
| License | GPL or Commercial | LGPL (more permissive) |
| Performance | Identical | Identical |
| API | Identical | Identical |
| Maturity | Slightly older | Newer but backed by Qt Co. |

**Recommendation:** Start with **PySide6** for licensing flexibility (LGPL allows commercial use without license fees). Switch to PyQt6 only if you need specific PyQt-only features.

---

## Implementation Plan Overview

### Phase 1: Setup Window
**Widgets:**
- QLineEdit for text inputs (subject ID, task, run number)
- QSpinBox for numeric inputs (n_volumes, n_baseline, TR)
- QFileDialog for directory selection (incoming_dir, output_dir, masks)
- QComboBox for dropdowns (task type, run type)
- QPushButton to launch session
- QTableWidget to display previous runs (from SessionState JSON)

**Layout:**
- QTabWidget with tabs: "Session Config", "Simulator", "Advanced", "History"
- QFormLayout for parameter forms
- QVBoxLayout/QHBoxLayout for button arrangements

**Files to create:**
- `src/pyrtnf/gui/setup_window.py` (main configuration dialog)
- `src/pyrtnf/gui/config_models.py` (Pydantic models for validation)

---

### Phase 2: Monitoring Dashboard
**Widgets:**
- pyqtgraph PlotWidget for timeseries:
  - Dual-network activations (2 lines, scrolling)
  - Motion parameters (6 lines, scrolling)
  - Ball position Y (1 line, scrolling)
- QLabel for numeric displays (current volume, hits, scale factor)
- QProgressBar for session progress (volume N of M)
- QTextEdit for log display (recent events)
- QPushButton for controls (pause, stop, manual trigger)

**Layout:**
- QGridLayout for multi-plot arrangement:
  - Top row: Activation plot (large)
  - Middle row: Motion plot + Ball position plot (side-by-side)
  - Bottom row: Controls + Status labels

**Real-time updates:**
- QTimer polling CSV files (feedback.log.tsv, ball_task.tsv) at 10 Hz
- Qt signals from NeurofeedbackCoordinator (if integrated directly)

**Files to create:**
- `src/pyrtnf/gui/monitor_window.py` (dashboard)
- `src/pyrtnf/gui/plots.py` (custom pyqtgraph plot widgets)
- `src/pyrtnf/gui/log_monitor.py` (CSV tailing logic)

---

### Phase 3: Integration with Existing Pipeline
**Architecture:**
- Main GUI app in `src/pyrtnf/gui/app.py`
- Launches PsychoPy task in subprocess (not same process)
- NeurofeedbackCoordinator runs in QThread (Qt-managed thread)
- Signals bridge coordinator → GUI

**Modifications to existing code:**
- Add Qt signals to NeurofeedbackCoordinator (backward compatible, optional)
- Add `--gui` flag to run_feedback_session()
- Keep CLI interface intact for non-GUI users

**Files to modify:**
- [feedback.py](src/pyrtnf/feedback.py) (add optional Qt signals)
- [session.py](src/pyrtnf/session.py) (add GUI launch option)

---

### Phase 4: NIfTI Visualization (Optional Enhancement)
**For mask preview in setup:**
- Use nibabel to load NIfTI files
- Display tri-planar slices as QPixmap in QLabel
- Reuse overlay logic from [qc.py](src/pyrtnf/qc.py) (matplotlib → QImage conversion)

**Files to create:**
- `src/pyrtnf/gui/nifti_viewer.py` (lightweight slice viewer)

---

## Dependencies

### Core GUI:
```toml
# Add to pyproject.toml [project.dependencies]
"PySide6>=6.6.0",  # Qt framework
"pyqtgraph>=0.13.3",  # Real-time plotting
```

### Optional (for NIfTI preview):
```toml
# Already have: nibabel, matplotlib
# Conversion: matplotlib fig → QPixmap via matplotlib Qt backend
```

**Total new dependencies:** 2 (PySide6, pyqtgraph)

---

## Testing Strategy

### Unit tests:
- Test configuration validation (Pydantic models)
- Test CSV parsing logic (log_monitor.py)
- Mock Qt signals for coordinator integration

### Integration tests:
- Use QTest for GUI interaction testing
- Test subprocess launch of PsychoPy task
- Verify thread-safe signal/slot communication

### Manual testing:
- Run simulator mode with GUI monitoring
- Verify plots update correctly at TR intervals
- Test pause/resume functionality

---

## Alternative Recommendations (if PyQt/PySide rejected)

### If simplicity is paramount:
**Tkinter + matplotlib** - Functional but limited, acceptable for basic monitoring

### If web stack is acceptable:
**Plotly Dash** - Python-only web framework, easier than Flask+React
- Example: Single Python file creates web dashboard
- Real-time updates via `dcc.Interval` component
- Deployment: `python app.py` → opens browser

### If cutting-edge preferred:
**Textual** - Terminal-based UI (TUI) library
- Surprisingly powerful for dashboards
- No GUI overhead
- Modern terminal features (rich text, plots via plotext)
- Best for remote SSH access scenarios

---

## Implementation Phases

### Phase 1: Core Infrastructure (Foundation)

**Goal**: Set up GUI package structure, config management, and basic navigation

**New Files to Create**:
1. `src/pyrtnf/gui/__init__.py` - Package exports
2. `src/pyrtnf/gui/app.py` - Main Flet app with navigation
3. `src/pyrtnf/gui/models.py` - StudyMetadata dataclass
4. `src/pyrtnf/gui/config_manager.py` - Study YAML save/load
5. `src/pyrtnf/gui/utils.py` - Validation, threading helpers

**Tasks**:
- Create `~/.pyrtnf/studies/` directory on first launch
- Implement `ConfigManager.list_studies()`, `load_study()`, `save_study()`
- Set up Flet app with screen routing (study manager → editor → workflow)
- Add basic logging configuration

**Testing**:
- Create dummy study config, save to YAML, reload
- Verify YAML format matches SessionConfig structure
- Test navigation between screens

---

### Phase 2: Study Manager Screen

**Goal**: Browse and manage study configurations

**New Files**:
- `src/pyrtnf/gui/screens/__init__.py`
- `src/pyrtnf/gui/screens/study_manager.py`

**Components to Build**:
- Study card grid (name, description, last modified)
- New study button → Study Editor
- Edit button → Study Editor (with loaded config)
- Duplicate button → Create copy
- Delete button → Confirmation dialog

**Testing**:
- Create multiple studies, verify card display
- Edit study, verify changes persist
- Duplicate study, verify copy created
- Delete study, verify YAML removed

---

### Phase 3: Study Editor Screen

**Goal**: Create/edit study configurations with full parameter control

**New Files**:
- `src/pyrtnf/gui/screens/study_editor.py`
- `src/pyrtnf/gui/components/__init__.py`
- `src/pyrtnf/gui/components/validators.py` - Field validation

**Tabs to Implement**:
1. **Scanner Tab**: TR, n_volumes, n_baseline, simulator settings
2. **Networks Tab**: Yeo network multi-select, mask names, filtering params
3. **Task Tab**: Scale factor, hit thresholds, button mappings
4. **Paths Tab**: File browser dialogs for DICOM sources

**Validation Rules**:
- TR > 0
- n_volumes > 0
- n_yeo ∈ {7, 17}
- At least 2 networks selected
- File paths exist (warning, not error)

**Testing**:
- Create new study with all parameters
- Edit existing study, verify pre-population
- Test validation (invalid TR, missing network selection)
- Save study, verify YAML structure correct
- "Start Session" button navigates to Participant Workflow

---

### Phase 4: Participant Workflow - Step Dashboard

**Goal**: Display pipeline steps with status tracking

**New Files**:
- `src/pyrtnf/gui/screens/participant_workflow.py`
- `src/pyrtnf/gui/components/step_button.py` - Reusable step widget
- `src/pyrtnf/gui/components/status_tracker.py` - Detect completed steps

**Step Buttons to Create**:
- Setup: 2vol initial, rest runs, preprocess, masks, coreg
- Feedback: 2vol realtime, transfer/feedback runs, "Run Next"
- QC: Per-phase report viewers

**Status Detection**:
- Check for output files (preproc.nii.gz, masks, etc.)
- Load SessionState.json for run history
- Enable/disable buttons based on dependencies

**Testing**:
- Load participant with partial completion, verify status
- Click step button, verify param preview appears
- Test button enable/disable logic

---

### Phase 5: Parameter Preview Panel

**Goal**: Show step details before execution

**New Files**:
- `src/pyrtnf/gui/components/param_preview.py`

**Features**:
- Display step name, description
- List key parameters (DICOM source, volumes, output dir)
- Show warnings if outputs exist
- Overwrite checkbox
- "Start" button

**Testing**:
- Click various step buttons, verify correct params shown
- Test overwrite warning logic
- Cancel button clears preview

---

### Phase 6: Pipeline Execution & Monitoring

**Goal**: Run pipeline steps in background, show progress

**New Files**:
- `src/pyrtnf/gui/components/pipeline_runner.py` - Thread wrapper
- `src/pyrtnf/gui/components/monitoring_panel.py` - Real-time view
- `src/pyrtnf/gui/components/log_console.py` - Log viewer

**Execution Flow**:
1. User clicks "Start" in param preview
2. PipelineRunner starts background thread
3. Monitoring panel shows progress
4. On completion: update status, generate QC, show summary

**Steps to Integrate**:
- DICOM collection (rest, 2vol)
- Preprocessing
- Mask extraction
- Coregistration
- Feedback runs (with special handling for PsychoPy)

**Testing**:
- Run each step in simulator mode
- Verify GUI doesn't freeze during execution
- Test error handling (invalid DICOM path)
- Verify status updates on completion

---

### Phase 7: Real-Time Monitoring (Feedback Runs)

**Goal**: Display live activation plots during feedback

**New Files**:
- `src/pyrtnf/gui/components/activation_plot.py` - Flet LineChart wrapper
- `src/pyrtnf/gui/components/csv_poller.py` - Monitor TSV files

**Features**:
- Dual-network activation timeseries
- Progress bar (volume N / total)
- Hit counters, scale factor display
- Motion parameter (FD) display
- Stop button (emergency abort)

**CSV Polling Logic**:
- Monitor `feedback.log.tsv` with Flet timer (1 Hz)
- Parse new rows, append to plot data
- Update progress indicators

**Testing**:
- Run feedback run in simulator mode
- Verify plots update in real-time
- Test Stop button (coordinator cleanup)
- Verify completion dialog appears

---

### Phase 8: Log Console

**Goal**: Always-visible streaming logs

**New Files**:
- `src/pyrtnf/gui/components/log_console.py`

**Features**:
- Auto-scrolling text viewer
- Color-coded by level (INFO, WARNING, ERROR)
- Filter buttons
- Export to file button

**Integration**:
- Capture pyrtnf logger output
- Thread-safe log appending
- Max 1000 lines in memory (rolling buffer)

**Testing**:
- Run pipeline step, verify logs appear
- Test filter buttons
- Export logs to file

---

### Phase 9: QC Report Integration

**Goal**: Auto-generate and view QC reports

**New Files**:
- `src/pyrtnf/gui/components/qc_viewer.py`

**QC Generation**:
- Call `pyrtnf.qc` functions after each step completes
- Preprocessing QC, Mask QC, Feedback QC

**QC Viewing**:
- Option A: Embedded HTML viewer (if Flet supports)
- Option B: Open in external browser
- Show in dialog or separate tab

**Testing**:
- Complete preprocessing, verify QC auto-generates
- Click QC button, verify report displays
- Test with multiple runs (per-run feedback QC)

---

### Phase 10: "Run Next Task" Smart Button

**Goal**: Simplify multi-run workflow

**Logic**:
- Load SessionState to determine completed runs
- Determine next run type: transfer → feedback × N → transfer
- Disable if session complete
- Highlight next recommended step

**Integration**:
- Reuse run_next_task() logic from session.py
- Update button text dynamically ("Next: Feedback Run 2")

**Testing**:
- Run full session using only "Run Next Task"
- Verify correct run order
- Test session completion detection

---

### Phase 11: BIDS Export

**Goal**: Export session data to BIDS derivatives

**New Files**:
- `src/pyrtnf/gui/utils/bids_export.py`

**Features**:
- Generate `dataset_description.json`
- Create BIDS derivatives structure
- Copy/symlink outputs to BIDS layout
- Export button in QC section

**Testing**:
- Run session, export to BIDS
- Validate with bids-validator (if available)

---

### Phase 12: Polish & Error Handling

**Goal**: Improve UX, handle edge cases

**Tasks**:
- Add confirmation dialogs (delete study, overwrite outputs)
- Improve error messages (friendly text, actionable suggestions)
- Add keyboard shortcuts (Ctrl+S to save study)
- Loading spinners during file I/O
- Tooltips for parameters
- About dialog (pyrtnf version, credits)

**Testing**:
- Deliberately trigger errors, verify user-friendly messages
- Test edge cases (empty studies directory, corrupted YAML)

---

## Critical Files

**Existing Files (read for integration)**:
- [session.py](src/pyrtnf/session.py):1-1203 - SessionConfig dataclasses, workflow functions
- [feedback.py](src/pyrtnf/feedback.py):1008-1779 - NeurofeedbackCoordinator threading
- [tasks/session.py](src/pyrtnf/tasks/session.py) - SessionState persistence
- [qc.py](src/pyrtnf/qc.py):1-1660 - QC report generation
- [preprocessing.py](src/pyrtnf/preprocessing.py) - collect_dicoms, preprocessing
- [mask_extraction.py](src/pyrtnf/mask_extraction.py) - mask_extraction_rest, coregister_2vol

**New Files to Create** (17 files):
```
src/pyrtnf/gui/
├── __init__.py
├── app.py
├── models.py
├── config_manager.py
├── utils.py
├── screens/
│   ├── __init__.py
│   ├── study_manager.py
│   ├── study_editor.py
│   └── participant_workflow.py
└── components/
    ├── __init__.py
    ├── step_button.py
    ├── param_preview.py
    ├── monitoring_panel.py
    ├── activation_plot.py
    ├── log_console.py
    ├── qc_viewer.py
    └── pipeline_runner.py
```

**Configuration Changes**:
- [pyproject.toml](pyproject.toml) - Add flet dependency

---

## Verification Plan

**Manual Testing Checklist**:

1. **Study Management**:
   - [ ] Create new study with all parameters
   - [ ] Save study to YAML
   - [ ] Load study from YAML
   - [ ] Edit study, save changes
   - [ ] Duplicate study
   - [ ] Delete study (with confirmation)

2. **Participant Workflow - Setup Phase**:
   - [ ] Enter participant ID
   - [ ] Collect initial 2vol (simulator mode)
   - [ ] Collect rest run 1 (simulator mode)
   - [ ] Collect rest run 2 (simulator mode)
   - [ ] Run preprocessing
   - [ ] Extract network masks
   - [ ] Coregister masks to 2vol
   - [ ] Verify QC reports auto-generate

3. **Participant Workflow - Feedback Session**:
   - [ ] Collect realtime 2vol
   - [ ] Run transfer run 1 (PsychoPy appears on monitor 2)
   - [ ] Run feedback run 1 with live monitoring
   - [ ] Verify activation plots update in real-time
   - [ ] Run feedback run 2
   - [ ] Run transfer run 2
   - [ ] Test "Run Next Task" button through full session

4. **Monitoring & QC**:
   - [ ] View preprocessing QC report
   - [ ] View mask extraction QC report
   - [ ] View feedback run QC reports
   - [ ] Check log console shows all events
   - [ ] Export logs to file

5. **Error Handling**:
   - [ ] Trigger error (invalid DICOM path), verify friendly message
   - [ ] Test resume after interruption
   - [ ] Test overwrite confirmation dialogs

6. **Integration**:
   - [ ] Verify PsychoPy doesn't interfere with GUI
   - [ ] Test with real scanner mode (toggle simulator off)
   - [ ] Export session to BIDS format

7. **Performance**:
   - [ ] GUI remains responsive during long steps (preprocessing)
   - [ ] Activation plots update smoothly (no lag)
   - [ ] Log console doesn't slow down with many messages

---

## Dependencies

**Add to [pyproject.toml](pyproject.toml)**:
```toml
[project]
dependencies = [
    # ... existing deps ...
    "flet>=0.23.0",
]

[project.optional-dependencies]
gui = ["flet>=0.23.0"]

[project.scripts]
pyrtnf-gui = "pyrtnf.gui.app:main"
```

**No changes needed** to existing pyrtnf code - GUI wraps existing functions.

---

## Recommended Study Config Storage Structure

```
~/.pyrtnf/
└── studies/
    ├── dual-network-feedback.yaml
    ├── dmn-only-feedback.yaml
    └── default-template.yaml
```

**Example YAML** (dual-network-feedback.yaml):
```yaml
_metadata:
  name: "Dual Network Feedback"
  description: "ContA vs DefaultA neurofeedback protocol for executive control"
  created: "2026-01-16T10:00:00"
  modified: "2026-01-16T14:30:00"

paths:
  participant_id: ""  # Set per participant
  base_dir: "/data/pyrtnf"

scan:
  tr: 1.2
  n_volumes: 150
  n_baseline: 25
  rest_source_dicoms:
    - "/data/source/rest_run1"
    - "/data/source/rest_run2"
  rest_n_volumes: [200, 200]
  rest_max_runs: 2
  twovol_initial_dicoms: "/data/source/2vol_initial"
  twovol_realtime_dicoms: "/data/source/2vol_realtime"
  twovol_n_volumes: 2
  feedback_dicoms: "/data/source/feedback"
  collection_speed: 10.0
  simulator_speed: 1.0

mask:
  n_yeo: 17
  yeo_targets: ["DefaultA", "ContA"]
  smooth_fwhm: 5.0
  highpass: 0.01
  lowpass: null
  mask_names: ["DefaultA", "ContA"]

feedback:
  n_baseline: 25
  n_volumes: 150
  use_efficiency_weights: true

task:
  initial_scale_factor: 10.0
  fullscreen: true
  trigger_keys: ["t", "+", "5", "equal"]
  left_button: "1"
  right_button: "2"
  enter_button: "0"
  min_hits_threshold: 3
  max_hits_threshold: 5
  scale_adjust_factor: 0.25
  max_feedback_runs: 2
```

---

## Key Design Decisions (Summary)

1. **Flet Framework**: Python-native, modern UI, sufficient performance for TR-interval monitoring
2. **Two-Tier Architecture**: Study configs (reusable) → Participant sessions (per-ID)
3. **Studies Library**: Centralized YAML configs in `~/.pyrtnf/studies/`
4. **Manual Step Control**: No auto-progression, experimenter controls each step
5. **Parameter Preview**: Review before execution, with overwrite warnings
6. **Real-Time Monitoring**: CSV polling at 1 Hz for activation plots
7. **PsychoPy Separation**: Subprocess on monitor 2, no direct integration
8. **Auto QC**: Generate reports after each step completes
9. **Embedded QC Viewer**: HTML reports in GUI (or external browser fallback)
10. **Thread-Safe Execution**: Background threads for pipeline, page.update() for UI
11. **Resume Support**: Detect completed steps, allow re-run with confirmation
12. **BIDS Export**: Post-session export to derivatives format

---

## Conclusion

This plan delivers a comprehensive Flet-based GUI for pyrtnf with:
- **Study configuration management** for protocol reusability
- **Participant workflow dashboard** for manual step-by-step control
- **Real-time monitoring** for feedback runs with live activation plots
- **Integrated QC** with auto-generation and embedded viewing
- **Full parameter control** via tabbed editor
- **BIDS export** for data sharing

The architecture leverages existing pyrtnf code without modifications, wrapping pipeline functions in a thread-safe GUI. Phased implementation allows incremental development and testing.
