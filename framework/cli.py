import os
import argparse
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

    # Create a boilerplate script
    main_script_path = os.path.join(project_dir, "scripts", f"{project_dir}_main.py")
    with open(main_script_path, "w") as f:
        f.write(textwrap.dedent(f"""\
            \"\"\"
            Model Builder: {project_dir}
            \"\"\"
            import os
            import sys
            import subprocess
            
            # Ensure the project root and the PyAntiGen root are in sys.path so we can import modules
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
            
            from framework.pyantigen import PyAntiGen
            from modules.Basic.ma_reaction import BasicMAReaction

            if __name__ == "__main__":
                Isotopes = ['']
                
                # Initialize the PyAntiGen model
                model = PyAntiGen(name="{project_dir}", isotopes=Isotopes)
                
                # Instantiate your modules here
                BasicMAReaction(model)

                print(f"Reactions generated: {{model.counter}}")
                print(f"Rules generated: {{len(model.rules)}}")
                
                # Write and Convert to Antimony
                model.generate(__file__)
                
                # Automatically execute the generated Antimony model simulation
                script_path = os.path.join(os.path.dirname(__file__), "Antimony_{project_dir}_main.py")
                print("\\nModel generated successfully.")
                print("You can specify custom parameters in `antimony_models/{project_dir}_main_parameters.csv`.")
                print(f"Running simulation now via: python {{os.path.basename(script_path)}}")
                print("-" * 50)
                subprocess.call(["python", script_path])
        """))
    print(f"  Created file: scripts/{project_dir}_main.py")

    # Create dummy parameter configuration
    param_csv_path = os.path.join(project_dir, "antimony_models", f"{project_dir}_main_parameters.csv")
    with open(param_csv_path, "w") as f:
        f.write("Parameter,Value,Comment\n")
        f.write("k_A_to_B,0.1,Default rate constant for A to B\n")
        f.write("V_comp1,1.0,Default compartment volume\n")
        f.write("A_comp1,10.0,Initial amount of A\n")
        f.write("B_comp1,0.0,Initial amount of B\n")
    print(f"  Created file: antimony_models/{project_dir}_main_parameters.csv")

    # Create Antimony simulation script
    antimony_script_path = os.path.join(project_dir, "scripts", f"Antimony_{project_dir}_main.py")
    with open(antimony_script_path, "w") as f:
        f.write(textwrap.dedent(f"""\
            import tellurium as te
            import numpy as np
            import matplotlib.pyplot as plt
            import os
            import csv
            import time

            def run_simulation():
                current_dir = os.path.dirname(__file__)
                plot_path = os.path.normpath(os.path.join(current_dir, '..', 'results'))
                plot_name = os.path.join(plot_path, os.path.basename(__file__).replace('.py', '.png'))
                reactions_file = os.path.basename(__file__).replace('.py', '_all_reactions.txt')
                params_file = os.path.basename(__file__).replace('Antimony_', '').replace('.py', '_parameters.csv')
                rules_file = os.path.basename(__file__).replace('.py', '_rules.txt')
                
                # Load model files
                model_path = os.path.normpath(os.path.join(current_dir, '..', 'antimony_models', reactions_file))
                param_path = os.path.normpath(os.path.join(current_dir, '..', 'antimony_models', params_file))
                rules_path = os.path.normpath(os.path.join(current_dir, '..', 'antimony_models', rules_file))
                
                # Read the model files
                try:
                    with open(model_path, 'r') as f:
                        model_reactions = f.read()
                except FileNotFoundError:
                    print(f"Could not find {{model_path}}. Make sure to generate the model first.")
                    return

                model_parameters_list = []
                try:
                    with open(param_path, 'r', newline='', encoding='utf-8-sig') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            val = row['Value']
                            try:
                                float(val)
                                line = f"{{row['Parameter']}} = {{val}}"
                            except ValueError:
                                line = f"{{row['Parameter']}} := {{val}}"
                            if row.get('Comment'):
                                line += f" # {{row['Comment']}}"
                            model_parameters_list.append(line)
                    model_parameters = '\\n'.join(model_parameters_list)
                except FileNotFoundError:
                    print(f"Could not find {{param_path}}. Using empty parameters.")
                    model_parameters = ""
                    
                try:
                    with open(rules_path, 'r') as f:
                        model_rules = f.read()
                except FileNotFoundError:
                    model_rules = ""
                
                # Combine model components
                full_model_text = model_reactions + '\\n' + model_parameters + '\\n' + model_rules
                
                print("Loading model into Tellurium...")
                try:
                    r = te.loada(full_model_text)
                except Exception as e:
                    print(f"Error loading model: {{e}}")
                    print("Model Text:")
                    print(full_model_text)
                    return

                # Export SBML
                sbml_filename = os.path.basename(__file__).replace('.py','.xml')
                sbml_path = os.path.normpath(os.path.join(current_dir, '..', 'SBML_models', sbml_filename))
                os.makedirs(os.path.dirname(sbml_path), exist_ok=True)
                sbml_content = r.getSBML()
                with open(sbml_path, 'w') as f:
                    f.write(sbml_content)
                print(f"SBML exported to: {{sbml_path}}")
                
                # Run Simulation
                r.setIntegrator('cvode')
                print("Running simulation...")
                t0 = time.perf_counter()
                
                # Example generic simulation: Simulate from t=0 to 100 with 100 points
                result = r.simulate(0, 100, 100)
                elapsed = time.perf_counter() - t0
                print(f"Simulation time: {{elapsed:.3f}} s")
                
                # Plot results
                plt.figure(figsize=(8, 6))
                time_points = result[:, 0]
                for i in range(1, result.shape[1]):
                    plt.plot(time_points, result[:, i], label=result.colnames[i])
                plt.xlabel('Time')
                plt.ylabel('Concentration')
                plt.title('Simulation Results')
                plt.legend()
                plt.savefig(plot_name)
                print(f"Plot saved to: {{plot_name}}")
                plt.show()

            if __name__ == "__main__":
                run_simulation()
        """))
    print(f"  Created file: scripts/Antimony_{project_dir}_main.py")

    print("\nProject scaffolded successfully!")
    print("Note: Because PyAntiGen is installed in your Python environment,")
    print("you DO NOT need a copy of the 'framework' folder here. You can simply")
    print("import it directly (e.g., 'from framework.pyantigen import PyAntiGen')")
    print("from any script.")

if __name__ == "__main__":
    create_project()
