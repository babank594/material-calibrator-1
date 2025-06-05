"""
Author: S. Mohammad Hosseini V.
Affiliation: Shahid Beheshti University
Department: Civil, Water, and Environmental Engineering Faculty
Email: smo.hosseini@mail.sbu.ac.ir

This tool optimizes uniaxialMaterial model parameters to fit experimental data utilizing
Bayesian search and OpenSeespy.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import optuna
import hysteresis as hys
import openseespy.opensees as ops
import re
import optuna.visualization as vis
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ─────────────────────── Optimizer Class ─────────────────────── #

class MaterialOptimizer:
    """
    A class to handle material model optimization using experimental data.
    
    This class implements the core functionality for optimizing material model
    parameters to match experimental hysteresis curves using Bayesian optimization.
    """
    
    def __init__(self, variables, mat_command_template, strain, stress):
        """
        Initialize the MaterialOptimizer.
        
        Args:
            variables (dict): Dictionary of variable names and their bounds
            mat_command_template (str): OpenSees material command template
            strain (np.ndarray): Experimental strain data
            stress (np.ndarray): Experimental stress data
        """
        self.variables = variables
        self.mat_command_template = mat_command_template
        self.strain = strain
        self.expHys = hys.Hysteresis(np.column_stack([strain, stress]))
        self.study = None

    def build_command(self, params):
        """Build the OpenSees material command with optimized parameters."""
        cmd_filled = self.mat_command_template
        for key, value in params.items():
            cmd_filled = re.sub(rf"\b{key}\b", str(value), cmd_filled)
        return f"ops.{cmd_filled}"

    def get_material_stress(self, mat_command_str, load_protocol):
        """Execute OpenSees material command and get stress response."""
        ops.wipe()
        try:
            exec(mat_command_str, {"ops": ops})
        except Exception as e:
            raise ValueError(f"Material command failed: {e}")
        ops.testUniaxialMaterial(1)
        return [ops.setStrain(eps) or ops.getStress() for eps in load_protocol]

    def run_mat_stress_strain(self, params):
        """Run material model and get stress-strain response."""
        mat_command_str = self.build_command(params)
        stress = self.get_material_stress(mat_command_str, self.strain)
        return np.column_stack([self.strain, stress])

    def _create_params(self, trial):
        """Create parameter dictionary from trial suggestions."""
        return {
            name: trial.suggest_float(name, low, high)
            for name, (low, high) in self.variables.items() if low != high
        }

    def objective(self, trial):
        """Objective function for optimization."""
        params = self._create_params(trial)
        try:
            xy = self.run_mat_stress_strain(params)
            if np.all(xy == 0):
                return float('inf')
            modelHys = hys.Hysteresis(xy)
            diff, _ = hys.compareHys(modelHys, self.expHys)
            return diff
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')

    def optimize(self, n_trials=50):
        """Run the optimization process."""
        self.study = optuna.create_study(
            direction="minimize",
            pruner=MedianPruner(),
            sampler=TPESampler(multivariate=True)
        )
        self.study.optimize(self.objective, n_trials=n_trials)
        return self.study

    def plot_fit(self):
        """Plot comparison between experimental and optimized model hysteresis curves."""
        best_xy = self.run_mat_stress_strain(self.study.best_params)
        modelHys = hys.Hysteresis(best_xy)
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        
        self.expHys.plot(
            label="Experimental", 
            color='black', 
            linewidth=1.5,
            linestyle='-'
        )
        modelHys.plot(
            label="Optimized Model", 
            color='red', 
            linewidth=1.5,
            linestyle='--'
        )
        
        ax.set_xlabel("Strain (mm/mm)", fontsize=12)
        ax.set_ylabel("Stress (MPa)", fontsize=12)
        
        ax.legend(
            loc='best', 
            frameon=False, 
            fontsize=10,
            framealpha=1.0
        )
        
        ax.grid(True, linestyle=':', alpha=0.7)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(0.5)
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
    
        return fig

    def optuna_plots(self):
        """Generate Optuna visualization plots."""
        return (
            vis.plot_optimization_history(self.study),
            vis.plot_param_importances(self.study),
            vis.plot_parallel_coordinate(self.study),
            vis.plot_slice(self.study)
        )

# ─────────────────────── Utility Functions ─────────────────────── #

def extract_variables_from_command(cmd):
    """
    Extract variable names from OpenSees material command.
    
    Args:
        cmd (str): OpenSees material command string
        
    Returns:
        list: List of variable names found in the command
    """
    try:
        cmd = re.search(r"\((.*)\)", cmd, re.DOTALL).group(1)
        args = re.split(r",(?![^(]*\))", cmd)
        variables = []
        for arg in args[2:]:
            arg = arg.strip().strip("'\"")
            if arg.startswith('-'):
                continue
            try:
                float(arg)
                continue
            except ValueError:
                pass
            if re.match(r'^[a-zA-Z_]\w*$', arg):
                variables.append(arg)
        return sorted(set(variables))
    except Exception:
        return []

# ─────────────────────── Streamlit UI ─────────────────────── #

def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        "OpenSees Model and Material Calibration",
        layout="wide",
        page_icon="📐"
    )
    st.title("📐 OpenSees Material Model Calibrator")
    st.markdown(
        "This tool optimizes material model parameters to fit experimental data "
        "utilizing **Bayesian** search and **OpenSeespy**."
    )

def setup_sidebar():
    """Configure and handle sidebar file uploads."""
    with st.sidebar:
        st.header("📁 Upload Experimental Data")
        strain_file = st.file_uploader("Strain/deformation", type=["txt"])
        stress_file = st.file_uploader("Stress/Force", type=["txt"])
        st.markdown("---")
    return strain_file, stress_file

def load_experimental_data(strain_file, stress_file):
    """Load and validate experimental data files."""
    if strain_file and stress_file:
        try:
            strain = np.loadtxt(strain_file)
            stress = np.loadtxt(stress_file)
            st.success("✅ Experimental data loaded successfully.")
            return strain, stress
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
    return None, None

def setup_material_definition():
    """Configure material definition section."""
    st.subheader("🧠 Define Material Model")
    st.info(
        "Define your **uniaxialMaterial(matType, matTag, matArgs)** command here. "
        "Variables are detected automatically!"
    )
    
    st.markdown("""
    <h4 style='margin-bottom: 0.5rem;'>Material Definition Example</h4>
    """, unsafe_allow_html=True)
    
    st.code(
        "uniaxialMaterial('Steel02', 1, Fy, E0, b, R0, cR1, cR2, a1, 1.0, a3, 1.0, 0.0)",
        language='python'
    )
    
    return st.text_area(
        "Material Command",
        height=100,
        key="mat_template",
        help="Enter your OpenSees material command here"
    )

def setup_parameter_bounds(param_set):
    """Configure parameter bounds section."""
    variables = {}
    if param_set:
        st.subheader("⚙️ Parameter Bounds")
        st.markdown("Specify search bounds for each variable.")
        
        for param in param_set:
            col1, col2 = st.columns(2)
            with col1:
                low = st.number_input(
                    f"Lower bound for `{param}`",
                    key=f"{param}_low",
                    format="%.4f",
                    step=0.01
                )
            with col2:
                high = st.number_input(
                    f"Upper bound for `{param}`",
                    key=f"{param}_high",
                    format="%.4f",
                    step=0.01
                )
            if low != high:
                variables[param] = (low, high)
    else:
        st.warning("⚠️ No variable parameters detected. Ensure your command is valid.")
    
    return variables

def main():
    """Main application entry point."""
    setup_page()
    strain_file, stress_file = setup_sidebar()
    strain, stress = load_experimental_data(strain_file, stress_file)
    
    mat_command_template = setup_material_definition()
    param_set = extract_variables_from_command(mat_command_template)
    variables = setup_parameter_bounds(param_set)
    
    st.subheader("⚙️ Optimization Settings")
    n_trials = st.slider(
        "Number of Trials",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )
    
    run_button = st.button("🚀 Run Calibration")
    
    if run_button and strain is not None and stress is not None and variables:
        with st.spinner("Running optimization... please wait."):
            try:
                optimizer = MaterialOptimizer(
                    variables,
                    mat_command_template,
                    strain,
                    stress
                )
                study = optimizer.optimize(n_trials=int(n_trials))
                
                st.success("🎉 Optimization completed successfully!")
                
                st.subheader("✅ Best Parameters")
                st.json(study.best_params)
                st.markdown(
                    f"**Minimum Error (Hysteresis Distance):** `{study.best_value:.4e}`"
                )
                
                st.subheader("📉 Model Fit Comparison")
                st.pyplot(optimizer.plot_fit())
                
                st.subheader("📊 Optimization Diagnostics")
                for fig in optimizer.optuna_plots():
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")
    else:
        st.info("📎 Please upload data, define your model, and specify bounds before running.")

if __name__ == "__main__":
    main() 
