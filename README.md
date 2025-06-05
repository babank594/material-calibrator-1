# OpenSees uniaxialMaterial parameter calibration via Bayesian search 

A tool for calibrating OpenSees material models using experimental data. This application uses Bayesian optimization to find the best material parameters that match experimental hysteresis curves.

## Author

**S. Mohammad Hosseini V.**  
Shahid Beheshti University  
Civil, Water, and Environmental Engineering Faculty  
Email: smo.hosseini@mail.sbu.ac.ir

## Demo

Check out the demo of this application on LinkedIn: [View Demo](https://www.linkedin.com/posts/mohammad-hosseini-a0494a158_opensees-bayesianoptimization-materialmodeling-activity-7329334562998616064-lv7w?utm_source=share&utm_medium=member_desktop&rcm=ACoAACXe-SQBJUieOCLjfwkErb7fNiePjXDd4hY)
## Features

- Interactive web interface built with Streamlit
- Automatic parameter detection from OpenSees material commands
- Bayesian optimization using Optuna
- Real-time visualization of optimization progress
- Comparison plots between experimental and optimized model responses
- Support for various OpenSees uniaxial material models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FEMquake/material-calibrator.git
cd material-calibrator
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run src/app.py
```

2. Upload your experimental data:
   - Strain/deformation data file (text format)
   - Stress/force data file (text format)

3. Define your OpenSees material model command
4. Set parameter bounds for optimization
5. Run the calibration

## Example

```python
# Example material command
uniaxialMaterial('Steel02', 1, Fy, E0, b, R0, cR1, cR2, a1, 1.0, a3, 1.0, 0.0)
```

## Dependencies

- streamlit: Web interface
- numpy: Numerical computations
- matplotlib: Plotting
- optuna: Bayesian optimization
- hysteresis: Hysteresis curve analysis
- openseespy: OpenSees integration
- plotly: Interactive visualizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
