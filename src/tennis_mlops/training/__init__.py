"""Entraînement : jeu de données, découpage temporel, pipelines, évaluation, suivi MLflow, registre.

Une méthode = une responsabilité = une task Prefect (voir `tennis_mlops.flows.train`) :
`DatasetBuilder.build` → `TemporalSplitter.split` → `PipelineFactory.create` → `Trainer.fit` →
`Evaluator.evaluate` → `Reporter.log_candidate` → registre (`tennis_mlops.registry`).
"""
