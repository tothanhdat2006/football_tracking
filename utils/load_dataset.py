import os

from dotenv import load_dotenv
from roboflow import Roboflow

def get_roboflow_api_key() -> str:
    """
    Get Roboflow API Key from .env file

    Returns:
    - ROBOFLOW_API_KEY: str - Roboflow API Key
    """
    load_dotenv('../.env.local')
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
    return ROBOFLOW_API_KEY

def load_football_dataset(workspace_name: str = "roboflow-jvuqo", project_name: str = "football-players-detection-3zvbc", version_number: int = 12) -> str:
    """
    Load players, referees and ball detetction dataset from Roboflow if not exist

    Args:
    - workspace_name: str - name of the workspace
    - project_name: str - name of the project
    - version_number: int - version number of the project

    Returns:
    - dataset: str - path to the dataset
    - project: str - project name
    """
    if not os.path.exists('./datasets/football'):
        os.makedirs('./datasets/football')
        ROBOFLOW_API_KEY = get_roboflow_api_key()
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(workspace_name).project(project_name)
        version = project.version(version_number)
        dataset = version.download("yolo11")
        return dataset, project
    else:
        return './datasets/football'

def load_field_dataset(workspace_name: str = "roboflow-jvuqo", project_name: str = "football-field-detection-f07vi", version_number: int = 15) -> str:
    """
    Load field keypoints detetction dataset from Roboflow if not exist

    Args:
    - workspace_name: str - name of the workspace
    - project_name: str - name of the project
    - version_number: int - version number of the project

    Returns:
    - dataset: str - path to the dataset
    - project: str - project name
    """
    if not os.path.exists('./datasets/field'):
        os.makedirs('./datasets/field')
        ROBOFLOW_API_KEY = get_roboflow_api_key()
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(workspace_name).project(project_name)
        version = project.version(version_number)
        dataset = version.download("yolov8")
        return dataset, project
    else:
        return './datasets/field'
