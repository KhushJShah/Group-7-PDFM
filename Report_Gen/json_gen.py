import json
import os
import shutil


def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)


data_to_save = \
    {
        # -----------------------------------------------------------------------------------------------------------------------
        "Version":
            """7""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2025""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Spring""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Analyze and predict migration trends of low-income families in FEMA Region 9 in response to climate change impacts, focusing on identifying spatial and temporal patterns to inform equitable relocation strategies.""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            The primary goal of this project is to predict trends in low-income populations (e.g., displaced individuals) in Region 9(California, Arizona, Nevada) and identify optimal relocation strategies using geographical data. The problem involves:
            Modeling spatial and temporal dynamics of low-income groups.
            Incorporating complex relationships between regions (e.g., migration flows, resource availability).

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            There are primarily 3 datasets that we are going to work with:
            1. Health analytics
            2. Unemployment
            3. Poverty
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            Utilizing GNNs will help in undertaking the consideration of spatial features, while the time-series incorporation to it will help in temporal analysis.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            The project will proceed in several phases:
            **Data Collection and Preprocessing**: Collect geospatial data (e.g., county-level maps, housing costs, employment rates) and demographic data (e.g., income levels, population density). Represent data as both tabular datasets (for traditional models) and graph structures (for GNNs). For graphs:Nodes: Counties or regions. Preprocess data to ensure consistency across formats.
            **Modeling Approaches**: Traditional Models: Use regression-based models (linear regression, random forests) as a baseline for predicting trends. Graph Neural Networks: Experiment with suitable GNN architectures such as Graph Convolutional Networks (GCNs) for static spatial relationships.

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            This is a rough timeline for the project:
             **Weeks 1-2**: Familiarization with project requirements, intellectual humility constructs, and open-source tools.
             **Weeks 3-8**: Design, develop, and build a working POC for 1 region.
             **Weeks 9-12**: Upscaling it for the other regions
             **Weeks 13-16**: Extrapolate the missing regions.
             **Weeks 17-18**: Final reporting, documentation, and presentation of outcomes.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            This project is suitable for a team of 2 student, given the scope and need for interdisciplinary collaboration.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            Potential challenges for this project include:
             **Limited Training Data**: Finding appropriate datasets for Spatial analysis may be a limiting factor.
            """,

        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Dr. Michael Mann and Khush Shah",
        "Proposed by email": "khushjayant.shah@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gmail.com",
        "github_repo": "https://github.com/amir-jafari/Capstone",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy('json_gen.py', output_file_path)
print(f"Data saved to {output_file_path}")
