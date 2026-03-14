import argparse
import json
import os
import shutil
import textwrap

def create_project():
    parser = argparse.ArgumentParser(description="Initialize a new PyAntiGen project structure.")
    parser.add_argument("project_name", help="Name of the directory to create for the new project")
    args = parser.parse_args()

    project_dir = args.project_name

    if os.path.exists(project_dir):
        print(f"Error: Directory '{project_dir}' already exists.")
        return

    # Directories to scaffold (top-level .agents for Cursor/agent skills)
    directories = [
        "scripts",
        "modules",
        "modules/Basic",
        "data",
        "antimony_models",
        "generated",
        "results",
        "SBML_models",
        ".agents",
        ".agents/skills",
    ]

    print(f"Creating PyAntiGen project '{project_dir}'...")

    # Create directories
    for d in directories:
        path = os.path.join(project_dir, d)
        os.makedirs(path, exist_ok=True)
        # Create an __init__.py in modules so it's a python package
        if d == "modules":
            with open(os.path.join(path, "__init__.py"), "w") as f:
                pass
        elif d == "modules/Basic":
            with open(os.path.join(path, "__init__.py"), "w") as f:
                f.write("# Basic module\n")
            with open(os.path.join(path, "ma_reaction.py"), "w") as f:
                f.write(textwrap.dedent("""\
                    \"\"\"
                    Basic module with a single MA reaction A -> B.
                    \"\"\"

                    from framework.module_base import PyAntiGenModule

                    class BasicMAReaction(PyAntiGenModule):
                        \"\"\"
                        Creates a simple MA reaction A -> B.
                        \"\"\"
                        def build(self):
                            # Define reaction properties
                            Reaction_name = "Basic_A_to_B"
                            Reactants = "[A_comp1]"
                            Products = "[B_comp1]"
                            Rate_type = "MA"
                            Rate_eqtn_prototype = "k_A_to_B"
                            
                            # Add the reaction to the model
                            self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)
                """))
        print(f"  Created folder: {d}/")

    # PyAntiGen project settings (e.g. archive_with_timestamp: false by default to avoid large projects)
    settings_path = os.path.join(project_dir, "pyantigen_settings.json")
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump({"archive_with_timestamp": False}, f, indent=2)
    print(f"  Created file: pyantigen_settings.json")

    # Copy framework's .agents/skills into project top-level .agents (so agents run in project context)
    framework_dir = os.path.dirname(os.path.abspath(__file__))
    framework_agents = os.path.join(framework_dir, ".agents")
    project_agents_skills = os.path.join(project_dir, ".agents", "skills")
    if os.path.isdir(framework_agents):
        src_skills = os.path.join(framework_agents, "skills")
        if os.path.isdir(src_skills):
            for name in os.listdir(src_skills):
                src_sub = os.path.join(src_skills, name)
                if os.path.isdir(src_sub):
                    dst_sub = os.path.join(project_agents_skills, name)
                    shutil.copytree(src_sub, dst_sub)
                    print(f"  Copied .agents/skills/{name}/")
        else:
            print("  Created folder: .agents/skills/ (no template skills in this install)")
    else:
        print("  Created folder: .agents/skills/")

    # Example scripts live in scripts/Example/ with MODEL_NAME = 'Example'
    example_scripts_dir = os.path.join(project_dir, "scripts", "Example")
    os.makedirs(example_scripts_dir, exist_ok=True)
    example_antimony_dir = os.path.join(project_dir, "antimony_models", "Example")
    example_generated_dir = os.path.join(project_dir, "generated", "Example")
    example_results_dir = os.path.join(project_dir, "results", "Example")
    os.makedirs(example_antimony_dir, exist_ok=True)
    os.makedirs(example_generated_dir, exist_ok=True)
    os.makedirs(example_results_dir, exist_ok=True)

    generate_script_path = os.path.join(example_scripts_dir, "Example_generate.py")
    with open(generate_script_path, "w") as f:
        f.write(textwrap.dedent("""\
            \"\"\"
            Example model builder (MODEL_NAME = 'Example'). Outputs go to antimony_models/Example/ and generated/Example/.
            \"\"\"
            import os
            import sys

            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

            from framework.pyantigen import PyAntiGen
            from modules.Basic.ma_reaction import BasicMAReaction

            if __name__ == "__main__":
                Isotopes = ['']
                model = PyAntiGen(name="Example", isotopes=Isotopes)
                BasicMAReaction(model)

                print(f"Reactions generated: {model.counter}")
                print(f"Rules generated: {len(model.rules)}")
                model.generate(__file__)
                print("\\nModel generated successfully.")
                print("Next steps:")
                print("  1. Optionally edit parameters in antimony_models/Example/Example_parameters.csv")
                print("  2. From scripts/Example/, run: python Example_run.py")
        """))
    print("  Created file: scripts/Example/Example_generate.py")

    param_csv_path = os.path.join(example_antimony_dir, "Example_parameters.csv")
    with open(param_csv_path, "w") as f:
        f.write("Parameter,Value,Units,Comment\n")
        f.write("k_A_to_B,0.1,,Default rate constant for A to B\n")
        f.write("V_comp1,1.0,,Default compartment volume\n")
    print("  Created file: antimony_models/Example/Example_parameters.csv")

    init_cond_path = os.path.join(example_antimony_dir, "Example_InitialConditions.csv")
    with open(init_cond_path, "w") as f:
        f.write("Species,InitialCondition,Units,Comment\n")
        f.write("A_comp1,10.0,,Initial amount of A\n")
        f.write("B_comp1,0.0,,Initial amount of B\n")
    print("  Created file: antimony_models/Example/Example_InitialConditions.csv")

    run_script_path = os.path.join(example_scripts_dir, "Example_run.py")
    with open(run_script_path, "w") as f:
        f.write(textwrap.dedent("""\
            import tellurium as te
            import numpy as np
            import matplotlib.pyplot as plt
            import os
            import time

            MODEL_NAME = "Example"

            def run_simulation():
                current_dir = os.path.dirname(__file__)
                if os.path.basename(current_dir) == "scripts":
                    project_root = os.path.normpath(os.path.join(current_dir, ".."))
                else:
                    project_root = os.path.normpath(os.path.join(current_dir, "..", ".."))
                plot_path = os.path.normpath(os.path.join(project_root, "results", MODEL_NAME))
                os.makedirs(plot_path, exist_ok=True)
                plot_name = os.path.join(plot_path, os.path.basename(__file__).replace(".py", ".png"))

                from framework.antimony_utils import load_antimony_files, archive_antimony_snapshot
                full_model_text = load_antimony_files(MODEL_NAME, project_root)
                if not full_model_text.strip():
                    print("No model content loaded. Generate the model first: python Example_generate.py")
                    return

                print("Loading model into Tellurium...")
                try:
                    r = te.loada(full_model_text)
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Model Text:")
                    print(full_model_text)
                    return

                sbml_content = r.getSBML()
                archive_dir = archive_antimony_snapshot(MODEL_NAME, project_root, sbml_content=sbml_content)
                print(f"Archive and SBML written to: {archive_dir}")

                r.setIntegrator("cvode")
                print("Running simulation...")
                t0 = time.perf_counter()
                result = r.simulate(0, 100, 100)
                elapsed = time.perf_counter() - t0
                print(f"Simulation time: {elapsed:.3f} s")

                plt.figure(figsize=(8, 6))
                time_points = result[:, 0]
                for i in range(1, result.shape[1]):
                    plt.plot(time_points, result[:, i], label=result.colnames[i])
                plt.xlabel("Time")
                plt.ylabel("Concentration")
                plt.title("Simulation Results")
                plt.legend()
                plt.savefig(plot_name)
                print(f"Plot saved to: {plot_name}")
                plt.show()

            if __name__ == "__main__":
                run_simulation()
        """))
    print("  Created file: scripts/Example/Example_run.py")

    print("\nProject scaffolded successfully!")
    print("Note: Because PyAntiGen is installed in your Python environment,")
    print("you DO NOT need a copy of the 'framework' folder here. You can simply")
    print("import it directly (e.g., 'from framework.pyantigen import PyAntiGen')")
    print("from any script.")

if __name__ == "__main__":
    create_project()
