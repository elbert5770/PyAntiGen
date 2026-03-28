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
                f.write(textwrap.dedent(f"""\
                    \"\"\"
                    Basic module with a single MA reaction A -> B.
                    \"\"\"

                    from framework.module_base import PyAntiGenModule

                    class BasicMAReaction(PyAntiGenModule):
                        \"\"\"
                        Creates a simple MA reaction A -> B.
                        \"\"\"
                        def build(self):
                            Compartments = ['Comp1']
                            for Comp in Compartments:
                                # Define reaction properties
                                Reaction_name = f"Basic_A_to_B_{{Comp}}"
                                Reactants = f"[A_{{Comp}}]"
                                Products = f"[B_{{Comp}}]"
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

    # Copy Example template exactly into scripts/Example/
    template_dir = os.path.join(framework_dir, "template")
    example_template = os.path.join(template_dir, "Example")
    example_scripts_dir = os.path.join(project_dir, "scripts", "Example")
    if os.path.isdir(example_template):
        shutil.copytree(example_template, example_scripts_dir)
        print("  Copied scripts/Example/ (from template)")
    else:
        os.makedirs(example_scripts_dir, exist_ok=True)
        print("  Created folder: scripts/Example/ (template not found)")

    # Copy Example data files into project data/
    template_data = os.path.join(template_dir, "data")
    project_data_dir = os.path.join(project_dir, "data")
    if os.path.isdir(template_data):
        for name in os.listdir(template_data):
            src = os.path.join(template_data, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(project_data_dir, name))
                print(f"  Copied data/{name}")

    # Example model dirs and parameter/initial-condition CSVs (for generate output and run)
    example_antimony_dir = os.path.join(project_dir, "antimony_models", "Example")
    example_generated_dir = os.path.join(project_dir, "generated", "Example")
    example_results_dir = os.path.join(project_dir, "results", "Example")
    os.makedirs(example_antimony_dir, exist_ok=True)
    os.makedirs(example_generated_dir, exist_ok=True)
    os.makedirs(example_results_dir, exist_ok=True)

    param_csv_path = os.path.join(example_antimony_dir, "Example_parameters.csv")
    with open(param_csv_path, "w") as f:
        f.write("Parameter,Value,Units,Comment\n")
        f.write("k_A_to_B,0.1,,Default rate constant for A to B\n")
        f.write("V_Comp1,1.0,,Default compartment volume\n")
    print("  Created file: antimony_models/Example/Example_parameters.csv")

    init_cond_path = os.path.join(example_antimony_dir, "Example_InitialConditions.csv")
    with open(init_cond_path, "w") as f:
        f.write("Species,InitialCondition,Units,Comment\n")
        f.write("A_Comp1,0.0,,Initial amount of A\n")
        f.write("B_Comp1,0.0,,Initial amount of B\n")
    print("  Created file: antimony_models/Example/Example_InitialConditions.csv")

    # Project-named folder: same structure as Example with Example Modules copied in
    project_scripts_dir = os.path.join(project_dir, "scripts", project_dir)
    project_modules_dir = os.path.join(project_scripts_dir, "Modules")
    os.makedirs(project_scripts_dir, exist_ok=True)
    os.makedirs(project_modules_dir, exist_ok=True)
    print(f"  Created folder: scripts/{project_dir}/")
    example_modules_src = os.path.join(example_template, "Modules")
    if os.path.isdir(example_modules_src):
        for name in os.listdir(example_modules_src):
            src = os.path.join(example_modules_src, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(project_modules_dir, name))
                print(f"  Copied scripts/{project_dir}/Modules/{name}")
        print(f"  Created folder: scripts/{project_dir}/Modules/ (Example modules)")
    else:
        print(f"  Created folder: scripts/{project_dir}/Modules/ (empty, template not found)")

    project_antimony_dir = os.path.join(project_dir, "antimony_models", project_dir)
    project_generated_dir = os.path.join(project_dir, "generated", project_dir)
    project_results_dir = os.path.join(project_dir, "results", project_dir)
    os.makedirs(project_antimony_dir, exist_ok=True)
    os.makedirs(project_generated_dir, exist_ok=True)
    os.makedirs(project_results_dir, exist_ok=True)

    # Copy Example scripts into project folder (MODEL_NAME derived from folder name at runtime)
    if os.path.isdir(example_template):
        for base in ("Example_generate", "Example_run", "Example_optimize"):
            src = os.path.join(example_template, base + ".py")
            if os.path.isfile(src):
                dst_name = project_dir + base[7:]  # "Example_foo" -> "{project_dir}_foo"
                dst = os.path.join(project_scripts_dir, dst_name + ".py")
                shutil.copy2(src, dst)
                print(f"  Created file: scripts/{project_dir}/{dst_name}.py")
        
        # Also copy paths.py utility
        paths_src = os.path.join(example_template, "paths.py")
        if os.path.isfile(paths_src):
            paths_dst = os.path.join(project_scripts_dir, "paths.py")
            shutil.copy2(paths_src, paths_dst)
            print(f"  Created file: scripts/{project_dir}/paths.py")

    print("\nProject scaffolded successfully!")
    print("Note: Because PyAntiGen is installed in your Python environment,")
    print("you DO NOT need a copy of the 'framework' folder here. You can simply")
    print("import it directly (e.g., 'from framework.pyantigen import PyAntiGen')")
    print("from any script.")

if __name__ == "__main__":
    create_project()
