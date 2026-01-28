# Documentation

This directory contains supplementary documentation for the pineuro library.

## Files

### [references.md](references.md)
Academic references and citations for the methods and algorithms used in pineuro:

- **Gentleman's Algorithm** - Incremental GLM updates for real-time feedback
- **Hinds et al. (2011)** - Moment-to-moment BOLD activation computation
- **Yeo et al. (2011)** - Functional network parcellation
- **FSL** - FMRIB Software Library for preprocessing
- **DICOM Standards** - Medical image format specifications

## Additional Resources

### Main Documentation
- [../README.md](../README.md) - Project overview, installation, and quick start
- [../CLAUDE.md](../CLAUDE.md) - Comprehensive architecture guide and development patterns
- [../src/pineuro/tasks/README.md](../src/pineuro/tasks/README.md) - Task framework integration guide

### Module Documentation
All Python modules include detailed docstrings following NumPy style:

```python
import pineuro
help(pineuro.preprocessing)  # View preprocessing documentation
help(pineuro.run_feedback_session)  # View function documentation
```

### Example Usage
See the main [README.md](../README.md) for complete workflow examples covering:
- DICOM collection and preprocessing
- Network mask extraction
- Real-time feedback sessions
- Offline testing with simulation mode

## Contributing Documentation

When adding new features or algorithms, please:
1. Add appropriate references to [references.md](references.md)
2. Update module docstrings with clear explanations
3. Include usage examples in docstrings when applicable
4. Update the main README.md if adding major functionality
