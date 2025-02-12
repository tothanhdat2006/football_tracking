import os

from dotenv import load_dotenv
from roboflow import Roboflow

def load_dataset(workspace_name: str = "roboflow-jvuqo", project_name: str = "football-players-detection-3zvbc", version_number: int = 12) -> str:
    """
    Load dataset from Roboflow if not exist

    Args:
    - workspace_name: str - name of the workspace
    - project_name: str - name of the project
    - version_number: int - version number of the project

    Returns:
    - dataset: str - path to the dataset
    - project: str - project name
    """
    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')
        load_dotenv('../.env.local')
        ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(workspace_name).project(project_name)
        version = project.version(version_number)
        dataset = version.download("yolo11")
        return dataset, project
    else:
        return './datasets'
