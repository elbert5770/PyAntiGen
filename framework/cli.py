import os
import argparse
import textwrap

def create_project():
    parser = argparse.ArgumentParser(description="Initialize a new PyAntiGen project structure.")
    parser.add_argument("project_name", help="Name of the directory to create for the new project")
    args = parser.parse_args()

    project_dir = args.project_name

    if os.path.exists(project_dir):
        print(f"Error: Directory '{project_dir}' already exists.")
        return

    # Directories to scaffold
    directories = [
        "scripts",
        "modules",
        "data",
        "antimony_models",
        "generated",
        "results",
        "SBML_models"
    ]

    print(f"Creating PyAntiGen project '{project_dir}'...")

    # Create directories
    for d in directories:
        path = os.path.join(project_dir, d)
        os.makedirs(path)
        # Create an __init__.py in modules so it's a python package
        if d == "modules":
            with open(os.path.join(path, "__init__.py"), "w") as f:
                pass
        print(f"  Created folder: {d}/")

    # Create a boilerplate script
    main_script_path = os.path.join(project_dir, "scripts", f"{project_dir}_main.py")
    with open(main_script_path, "w") as f:
        f.write(textwrap.dedent(f"""\
            \"\"\"
            Model Builder: {project_dir}
            \"\"\"
            from framework.pyantigen import PyAntiGen
            # from modules.my_module import MyModule

            if __name__ == "__main__":
                Isotopes = ['']
                
                # Initialize the PyAntiGen model
                model = PyAntiGen(name="{project_dir}", isotopes=Isotopes)
                
                # Instantiate your modules here
                # MyModule(model)

                print(f"Reactions generated: {{model.counter}}")
                print(f"Rules generated: {{len(model.rules)}}")
                
                # Write and Convert to Antimony
                model.generate(__file__)
        """))
    print(f"  Created file: scripts/{project_dir}_main.py")

    print("\nProject scaffolded successfully!")
    print("Note: Because PyAntiGen is installed in your Python environment,")
    print("you DO NOT need a copy of the 'framework' folder here. You can simply")
    print("import it directly (e.g., 'from framework.pyantigen import PyAntiGen')")
    print("from any script.")

if __name__ == "__main__":
    create_project()
