import os
from typing import Optional

from sagemaker.estimator import Estimator


REGISTRY = os.environ.get("AWS_REGISTRY")


class KermutEstimator(Estimator):
    def __init__(
        self,
        role: str,
        hyperparameters: Optional[dict] = None,
        image_uri=f"{REGISTRY}/kermut-groth:latest",
        instance_type: str = "ml.g5.2xlarge",
        instance_count: int = 1,
        **kwargs,
    ):
        super().__init__(
            image_uri=image_uri,
            instance_type=instance_type,
            instance_count=instance_count,
            role=role,
            tags=[
                {
                    "Key": "Application",
                    "Value": "Protein property prediction model training job",
                },
            ],
            hyperparameters=hyperparameters,
            script_mode=True,
            **kwargs,
        )
