"""Clustering router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user
from database.models import User
from database.pg_operations import load_dataset_as_dataframe, save_model_run, write_audit_log
from models.clustering import run_kmeans_clustering

router = APIRouter(prefix="/cluster", tags=["clustering"])


class ClusterRequest(BaseModel):
    dataset_id: int
    feature_columns: list[str]
    performance_columns: list[str]
    random_state: int = 42


class ClusterResponse(BaseModel):
    run_id: int
    clustered_rows: int
    inertia: float
    cluster_counts: dict[str, int]
    profiles: list[dict]
    interpretations: list[dict]


@router.post("/run", response_model=ClusterResponse)
def cluster(
    body: ClusterRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    data = load_dataset_as_dataframe(db, body.dataset_id)
    if data is None or data.empty:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")

    result = run_kmeans_clustering(
        data=data,
        feature_columns=body.feature_columns,
        performance_columns=body.performance_columns,
        random_state=body.random_state,
    )

    cluster_counts = result.assignments["Cluster Label"].value_counts().to_dict()
    profiles = result.profiles.to_dict(orient="records")
    interpretations = result.interpretations.to_dict(orient="records")

    db_run = save_model_run(
        db=db,
        dataset_id=body.dataset_id,
        run_type="clustering",
        feature_columns=result.feature_columns,
        target_source=None,
        train_rows=result.clustered_rows,
        test_rows=None,
        cv_folds=None,
        performance_summary=f"K-Means clustering: {result.clustered_rows} rows, inertia={result.inertia:.2f}",
        run_config={"random_state": body.random_state, "performance_columns": body.performance_columns},
        results=[],
    )

    write_audit_log(
        db, action="TRAIN_CLUSTERING", user_id=current_user.id,
        resource_type="model_run", resource_id=str(db_run.id),
    )

    return ClusterResponse(
        run_id=db_run.id,
        clustered_rows=result.clustered_rows,
        inertia=result.inertia,
        cluster_counts={str(k): int(v) for k, v in cluster_counts.items()},
        profiles=profiles,
        interpretations=interpretations,
    )
