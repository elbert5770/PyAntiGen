import sys
import os

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from framework.pyantigen import PyAntiGen
from modules.Abeta.APP_reactions_Lin2022 import APP_reactions_Lin2022
from modules.Abeta.Abeta_aggregation_Lin2022 import Abeta_aggregation_Lin2022
from modules.Tissue_Flows.cns_flows_Lin2022 import cns_flows_Lin2022
from modules.Antibody.Antibody_PK_Lin2022 import Antibody_PK_Lin2022
from modules.Abeta.Abeta_antibody_binding_Lin2022 import Abeta_antibody_binding_Lin2022
from modules.Abeta.Abeta_ADCP_clearance_Lin2022 import Abeta_ADCP_clearance_Lin2022

if __name__ == "__main__":
    Isotopes = ['']
    model = PyAntiGen(name="Lin2022_test", isotopes=Isotopes)
    
    SpeciesList_Abeta_peptides = ['AB42']
    No_Isotope_SpeciesList = ['Antibody', 'FcR']

    Antibody_PK_Lin2022(model, Species='Antibody', No_Isotope_SpeciesList=No_Isotope_SpeciesList)

    for Species in SpeciesList_Abeta_peptides:
        APP_reactions_Lin2022(model, Species=Species, No_Isotope_SpeciesList=No_Isotope_SpeciesList)
        Abeta_aggregation_Lin2022(model, Species=Species, No_Isotope_SpeciesList=No_Isotope_SpeciesList)
        cns_flows_Lin2022(model, Species=Species, No_Isotope_SpeciesList=No_Isotope_SpeciesList)
        Abeta_antibody_binding_Lin2022(model, Species=Species, No_Isotope_SpeciesList=No_Isotope_SpeciesList)
        Abeta_ADCP_clearance_Lin2022(model, Species=Species, No_Isotope_SpeciesList=No_Isotope_SpeciesList)

    print(f"Reactions generated: {model.counter}")
    print(f"Rules generated: {len(model.rules)}")
    
    model.generate(__file__)
    
    # Parameter extraction (skip if Lin2022_extracted.txt not present)
    txt_path = os.path.join(project_root, "Lin2022_extracted.txt")
    if os.path.isfile(txt_path):
        import csv
        params = []
        in_table_2 = False
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("--- TABLE 2 ---"):
                    in_table_2 = True
                    continue
                if in_table_2 and line.startswith("---"):
                    break
                if in_table_2 and "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        desc_col = parts[1].strip()
                        val_col = parts[2].strip()
                        comment = parts[3].strip() if len(parts) > 3 else ""
                        if desc_col == "Parameter description":
                            continue
                        val_str = val_col.split()[0]
                        if val_str == "Dose":
                            continue
                        words = desc_col.replace(',', ' ').split()
                        symbols = []
                        for w in reversed(words):
                            if w.startswith('k') or w == 'konPP':
                                symbols.insert(0, w)
                            else:
                                if len(symbols) > 0:
                                    break
                        if not symbols:
                            symbols = [words[-1].strip(',')]
                        for sym in symbols:
                            params.append({'Parameter': sym, 'Value': val_str, 'Comment': comment})
        csv_file = os.path.basename(__file__).replace('.py', '_parameters.csv')
        csv_path = os.path.join(project_root, 'antimony_models', f"Antimony_{csv_file}")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Parameter', 'Value', 'Comment'])
            writer.writeheader()
            writer.writerows(params)
        print(f"Parameters extracted to {csv_path}")
